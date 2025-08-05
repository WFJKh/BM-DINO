# =============================================================
# EM-DINO 5th-rev — Prompt + ROI version (supports ablation switches)
# =============================================================
# Overview:
# 1. Learnable prompt token (L_p × D) — controlled by Config.use_prompt
# 2. Token-wise ROI pooling — controlled by Config.use_roi & Config.use_mask
# 3. Optional mask branch — remove it to get RGB-only variant
# 4. Adjustable freeze/unfreeze strategy — Config.freeze_backbone, Config.full_ft
# 5. Just change Config or CLI args for ablation experiments
# =============================================================

import os, math, argparse, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from transformers import Dinov2Model, Dinov2Config, get_cosine_schedule_with_warmup


# ---------------------------
# 1. Config
# ---------------------------
class Config:
    # --- data -----------------------------------------------
    img_dir = 'Blastocyst_Dataset/Images'
    mask_dir = 'Blastocyst_Dataset/Masks'
    train_csv = 'datasets/train.csv'
    val_csv = 'datasets/val.csv'
    test_csv = 'testset_filenames.csv'

    # --- training -------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    num_workers = 8
    epochs = 300
    lr = 1e-5
    weight_decay = 1e-3
    dropout = 0.3

    # --- focal loss -----------------------------------------
    gamma_focal = 2.0
    alpha_exp = torch.tensor([1.26, 1.26, 0.54, 0.22, 1.72])
    alpha_icm = torch.tensor([0.05, 0.21, 3.55, 0.19])
    alpha_te = torch.tensor([0.17, 0.36, 2.98, 0.49])

    # --- ablation switches ----------------------------------
    use_prompt = True   # False -> "–Prompt" variant
    prompt_len = 8      # L_p
    use_roi = True      # False -> "–ROI" variant
    use_mask = True     # False -> RGB-only variant
    freeze_backbone = True  # False -> "+Full FT" variant
    full_ft = False     # True -> unfreeze all

    # Freeze except last k layers (only valid when freeze_backbone=True & full_ft=False)
    k_unfreeze = 4

    # --- save path ------------------------------------------
    best_model = 'dino_prompt_roi.pth'


# ---------------------------
# 2. Data transforms
# ---------------------------
class BasicTF:
    def __init__(self, is_mask=False):
        if is_mask:
            self.tf = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.NEAREST),
                transforms.ToTensor()
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, img):
        return self.tf(img)


tf_img = BasicTF(False)
_tf_mask = BasicTF(True)  # lazy wrapper below to handle Config


def mask_tf(img):
    return _tf_mask(img if Config.use_mask else Image.new('RGB', img.size))


# ---------------------------
# 3. Dataset / Collate_fn
# ---------------------------
class BlastocystDS(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, header=None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname, exp, icm, te = self.df.iloc[idx][:4]
        img = Image.open(os.path.join(Config.img_dir, fname)).convert('RGB')
        mask_name = os.path.splitext(fname)[0] + '_mask.png'
        mask_path = os.path.join(Config.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('RGB') if (Config.use_mask and os.path.exists(mask_path)) else Image.new(
            'RGB', img.size)
        return tf_img(img), mask_tf(mask), int(exp), int(icm), int(te)


def collate_fn(batch):
    imgs, masks, exps, icms, tes = zip(*batch)
    return (torch.stack(imgs), torch.stack(masks),
            torch.tensor(exps), torch.tensor(icms), torch.tensor(tes))


# ---------------------------
# 4. Focal Loss
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # can be float or 1-D tensor

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)

        if isinstance(self.alpha, torch.Tensor):
            at = self.alpha.to(logits.device)[targets]  # sample-wise weight
        else:
            at = self.alpha if self.alpha is not None else 1.0

        loss = at * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ---------------------------
# 5. Model with Prompt & ROI
# ---------------------------
class MultiTaskDINOv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = Config
        # 1) backbone
        self.backbone = Dinov2Model.from_pretrained('dinov2-base')
        feat_dim = self.backbone.config.hidden_size  # 768

        # 2) learnable prompt tokens (optional)
        if self.cfg.use_prompt:
            self.prompt = nn.Parameter(torch.randn(1, self.cfg.prompt_len, feat_dim))

        # 3) mask branch (optional)
        if self.cfg.use_mask:
            self.mask_cnn = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(64, feat_dim)
            )

        # 4) heads
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.exp_head = nn.Linear(feat_dim, 5)
        self.icm_head = nn.Linear(feat_dim, 4)
        self.te_head = nn.Linear(feat_dim, 4)

        # 5) freezing strategy
        if not self.cfg.full_ft and self.cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if self.cfg.k_unfreeze > 0:
                # Compatible with encoder.layer or encoder.layers
                layers = getattr(self.backbone.encoder, 'layer', None) \
                         or getattr(self.backbone.encoder, 'layers')
                for blk in layers[-self.cfg.k_unfreeze:]:
                    for p in blk.parameters():
                        p.requires_grad = True

    # -------- ROI pooling utility ----------
    def patch_roi_pool(self, patch_tokens, masks):
        """
        patch_tokens: B × N × D  (N=14*14)
        masks       : B × C × 224 × 224  (C=1 or 3)
        Returns 1 × D ROI averaged feature per sample
        """
        B, N, D = patch_tokens.shape
        h = w = int(math.sqrt(N))  # 14
        masks_ds = nn.functional.interpolate(masks.float(), size=h, mode='nearest')
        masks_ds = masks_ds[:, 0]  # take 1st channel -> B × h × h
        masks_ds = masks_ds.flatten(1).bool()  # B × N
        area = masks_ds.sum(1, keepdim=True).clamp(min=1)
        roi_feat = (patch_tokens * masks_ds.unsqueeze(-1)).sum(1) / area
        return roi_feat  # B × D

    # ---------------- forward ----------------
    def forward(self, imgs, masks):
        B = imgs.size(0)

        # (1) backbone: tokens = [CLS] + patches
        tokens = self.backbone(imgs).last_hidden_state  # B × (1+N) × D
        cls_tok = tokens[:, 0, :]  # B × D
        patch_tok = tokens[:, 1:, :]  # B × N × D

        #(2) Prompt tokens
        if self.cfg.use_prompt:
            # broadcast prompt to batch, then mean over L_p
            prompt_feat = self.prompt.mean(dim=1).expand(B, -1)  # B × D
            fused = prompt_feat

        # (3) ROI vs. GAP
        if self.cfg.use_roi and self.cfg.use_mask:
            roi_feat = self.patch_roi_pool(patch_tok, masks)  # B × D
        else:
            roi_feat = patch_tok.mean(1)  # B × D

        # (4) optional mask CNN
        # fused = tok + roi_feat
        fused = fused + roi_feat
        if self.cfg.use_mask:
            mask_feat = self.mask_cnn(masks)  # B × D
            fused = fused + mask_feat

        fused = self.dropout(fused)

        # (5) task heads
        return (self.exp_head(fused),
                self.icm_head(fused),
                self.te_head(fused))


# ---------------------------
# 6. Train / Val / Test
# ---------------------------

def run_epoch(model, loader, crit_exp, crit_icm, crit_te, optim_=None):
    is_train = optim_ is not None
    model.train() if is_train else model.eval()
    total_loss, correct_e, correct_i, correct_t, tot = 0, 0, 0, 0, 0
    # criterion = FocalLoss(Config.gamma_focal, Config.alpha_focal)

    for img, mask, exp, icm, te in loader:
        img, mask = img.to(Config.device), mask.to(Config.device)
        exp, icm, te = exp.to(Config.device), icm.to(Config.device), te.to(Config.device)
        if is_train:
            optim_.zero_grad()
        o_e, o_i, o_t = model(img, mask)
        loss = (crit_exp(o_e, exp) +
                crit_icm(o_i, icm) +
                crit_te(o_t, te))
        if is_train:
            loss.backward();
            optim_.step()
        total_loss += loss.item()
        correct_e += (o_e.argmax(1) == exp).sum().item()
        correct_i += (o_i.argmax(1) == icm).sum().item()
        correct_t += (o_t.argmax(1) == te).sum().item()
        tot += exp.size(0)
    return total_loss / len(loader), (correct_e / tot, correct_i / tot, correct_t / tot)


# ---------------------------
# 7. Main
# ---------------------------

def main():
    tr_ds = BlastocystDS(Config.train_csv)
    va_ds = BlastocystDS(Config.val_csv)
    te_ds = BlastocystDS(Config.test_csv)
    tr_ld = DataLoader(tr_ds, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers,
                       collate_fn=collate_fn)
    va_ld = DataLoader(va_ds, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers,
                       collate_fn=collate_fn)
    te_ld = DataLoader(te_ds, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers,
                       collate_fn=collate_fn)

    model = MultiTaskDINOv2().to(Config.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    crit_exp = FocalLoss(Config.gamma_focal, Config.alpha_exp)
    crit_icm = FocalLoss(Config.gamma_focal, Config.alpha_icm)
    crit_te = FocalLoss(Config.gamma_focal, Config.alpha_te)
    optim_ = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.lr,
                         weight_decay=Config.weight_decay)
    sched_ = get_cosine_schedule_with_warmup(optim_, int(0.1 * len(tr_ld) * Config.epochs), len(tr_ld) * Config.epochs)

    best = 0
    for ep in range(1, Config.epochs + 1):
        # tr_loss, _         = run_epoch(model, tr_ld, optim_)
        # va_loss, va_metrics= run_epoch(model, va_ld)
        tr_loss, _ = run_epoch(model, tr_ld, crit_exp, crit_icm, crit_te, optim_)
        va_loss, va_metrics = run_epoch(model, va_ld, crit_exp, crit_icm, crit_te)

        mean_acc = np.mean(va_metrics)
        if mean_acc > best:
            best = mean_acc
            torch.save(model.state_dict(), Config.best_model)
        sched_.step()
        # if ep % 20 == 0 or ep==1:
        print(
            f"Ep {ep:03d} | TrainL {tr_loss:.4f} | ValL {va_loss:.4f} | EXP {va_metrics[0]:.3f} ICM {va_metrics[1]:.3f} TE {va_metrics[2]:.3f}")

    # ----- Test -----
    model.load_state_dict(torch.load(Config.best_model, map_location=Config.device))
    _, te_metrics = run_epoch(model, te_ld, crit_exp, crit_icm, crit_te)
    print(f"📊 Test Acc -> EXP {te_metrics[0]:.4f} | ICM {te_metrics[1]:.4f} | TE {te_metrics[2]:.4f}")


if __name__ == '__main__':
    torch.manual_seed(42);
    random.seed(42);
    np.random.seed(42)
    main()