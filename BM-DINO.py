import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from transformers import Dinov2Model


# ======================================
#   Mask-segmentation + DINOv2   v4
#   No LR scheduler, no EXP/ICM/TE weight self-weighting
# ======================================

# ==========================
# 1. Hyper-parameters & Config
# ==========================
class Config:
    # Data paths
    img_dir        = 'Blastocyst_Dataset/Images'
    mask_dir       = 'Blastocyst_Dataset/Masks'
    train_csv      = 'datasets/train.csv'
    val_csv        = 'datasets/val.csv'
    test_csv       = 'testset_filenames.csv'

    # Model saving
    best_model_path = 'best_dinov2_model.pth'

    # Training settings
    device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size     = 32
    num_workers    = 4
    epochs         = 300
    learning_rate  = 1e-4
    weight_decay   = 1e-3
    dropout        = 0.3    # 0.5

    # Data augmentation
    elastic_alpha  = 20
    elastic_sigma  = 5
    blur_prob      = 0.5


# ==========================
# 2. Data Augmentation & Preprocessing
# ==========================
class BasicTransform:
    """Simplified data augmentation"""
    def __init__(self, is_train=True, apply_to_mask=False):
        self.is_train = is_train
        self.apply_to_mask = apply_to_mask

        if not apply_to_mask:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

    def __call__(self, img: Image.Image):
        return self.transform(img)


# ==========================
# 3. Dataset Definition
# ==========================
class BlastocystDataset(Dataset):
    def __init__(self, csv_path, img_dir, mask_dir):
        self.data = pd.read_csv(csv_path, header=None)
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = str(row[0]).strip()
        exp = int(row[1])
        icm = int(row[2])
        te  = int(row[3])

        # Load image
        img_path = os.path.join(self.img_dir, filename)
        pil_img = Image.open(img_path).convert('RGB')

        # Load mask
        mask_filename = os.path.splitext(filename)[0] + "_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_filename)
        if os.path.exists(mask_path):
            pil_mask = Image.open(mask_path).convert('RGB')
        else:
            pil_mask = Image.new('RGB', pil_img.size, (0, 0, 0))

        return pil_img, pil_mask, exp, icm, te  # Return labels separately

    def __len__(self):
        return len(self.data)


# ==========================
# 4. DINOv2 Backbone
# ==========================
class MultiTaskDINOv2(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained model
        # self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        local_model_path = "dinov2-base"  # Local path must contain config.json, pytorch_model.bin, etc.
        self.backbone = Dinov2Model.from_pretrained(local_model_path)

        # Freeze most layers, only fine-tune last few
        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in self.backbone.encoder.layer[-4:]:
            for param in block.parameters():
                param.requires_grad = True

        feat_dim = 768

        # Mask processing module
        self.mask_processor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, feat_dim)
        )

        # Classification heads
        self.exp_head = nn.Linear(feat_dim, 5)
        self.icm_head = nn.Linear(feat_dim, 4)
        self.te_head = nn.Linear(feat_dim, 4)

        # Dropout
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x, mask):
        # Process image
        outputs = self.backbone(x)
        img_feat = outputs.last_hidden_state[:, 0, :]

        # Process mask
        mask_feat = self.mask_processor(mask)

        # Fuse features
        fused_feat = img_feat + mask_feat
        fused_feat = self.dropout(fused_feat)

        # Classification predictions
        exp_logits = self.exp_head(fused_feat)
        icm_logits = self.icm_head(fused_feat)
        te_logits = self.te_head(fused_feat)

        return exp_logits, icm_logits, te_logits


# ==========================
# 5. Training Function
# ==========================
def train(model, train_loader, val_loader):
    optimizer = optim.AdamW(model.parameters(),
                           lr=Config.learning_rate,
                           weight_decay=Config.weight_decay)

    # Simple cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(Config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            imgs = batch[0].to(Config.device)
            masks = batch[1].to(Config.device)
            exp_labels = batch[2].to(Config.device)
            icm_labels = batch[3].to(Config.device)
            te_labels = batch[4].to(Config.device)

            optimizer.zero_grad()

            exp_pred, icm_pred, te_pred = model(imgs, masks)

            # Compute loss
            loss_exp = criterion(exp_pred, exp_labels)
            loss_icm = criterion(icm_pred, icm_labels)
            loss_te = criterion(te_pred, te_labels)
            loss = loss_exp + loss_icm + loss_te

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        val_loss, val_acc_exp, val_acc_icm, val_acc_te = validate(model, val_loader, criterion)
        avg_val_acc = (val_acc_exp + val_acc_icm + val_acc_te) / 3

        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), Config.best_model_path)
            print(f"✅ Saved best model (Epoch {epoch+1}, avg acc: {avg_val_acc:.4f})")

        print(f"Epoch [{epoch+1}/{Config.epochs}] | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"EXP Acc: {val_acc_exp:.4f} | ICM Acc: {val_acc_icm:.4f} | TE Acc: {val_acc_te:.4f}")

    return best_val_acc


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct_exp, correct_icm, correct_te = 0, 0, 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch[0].to(Config.device)
            masks = batch[1].to(Config.device)
            exp_labels = batch[2].to(Config.device)
            icm_labels = batch[3].to(Config.device)
            te_labels = batch[4].to(Config.device)

            exp_pred, icm_pred, te_pred = model(imgs, masks)

            # Compute loss
            loss_exp = criterion(exp_pred, exp_labels)
            loss_icm = criterion(icm_pred, icm_labels)
            loss_te = criterion(te_pred, te_labels)
            loss = loss_exp + loss_icm + loss_te
            val_loss += loss.item()

            # Compute accuracy
            _, exp_pred = torch.max(exp_pred, 1)
            _, icm_pred = torch.max(icm_pred, 1)
            _, te_pred = torch.max(te_pred, 1)

            correct_exp += (exp_pred == exp_labels).sum().item()
            correct_icm += (icm_pred == icm_labels).sum().item()
            correct_te += (te_pred == te_labels).sum().item()
            total += exp_labels.size(0)

    val_loss /= len(val_loader)
    acc_exp = correct_exp / total
    acc_icm = correct_icm / total
    acc_te = correct_te / total

    return val_loss, acc_exp, acc_icm, acc_te


def evaluate_testset(model, test_loader):
    model.eval()
    correct_exp, correct_icm, correct_te = 0, 0, 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch[0].to(Config.device)
            masks = batch[1].to(Config.device)
            exp_labels = batch[2].to(Config.device)
            icm_labels = batch[3].to(Config.device)
            te_labels = batch[4].to(Config.device)

            exp_pred, icm_pred, te_pred = model(imgs, masks)

            _, exp_pred = torch.max(exp_pred, 1)
            _, icm_pred = torch.max(icm_pred, 1)
            _, te_pred = torch.max(te_pred, 1)

            correct_exp += (exp_pred == exp_labels).sum().item()
            correct_icm += (icm_pred == icm_labels).sum().item()
            correct_te += (te_pred == te_labels).sum().item()
            total += exp_labels.size(0)

    acc_exp = round(correct_exp / total * 100, 2)
    acc_icm = round(correct_icm / total * 100, 2)
    acc_te = round(correct_te / total * 100, 2)

    print("📊 Test-set evaluation results:")
    print(f"  EXP accuracy: {acc_exp}%")
    print(f"  ICM accuracy: {acc_icm}%")
    print(f"  TE  accuracy: {acc_te}%")


# ==========================
# 6. Main Entry Point
# ==========================
def main():
    # Create data transforms
    img_train_tf = BasicTransform(is_train=True, apply_to_mask=False)
    mask_train_tf = BasicTransform(is_train=True, apply_to_mask=True)
    img_val_tf = BasicTransform(is_train=False, apply_to_mask=False)
    mask_val_tf = BasicTransform(is_train=False, apply_to_mask=True)

    # Create datasets
    train_ds = BlastocystDataset(Config.train_csv, Config.img_dir, Config.mask_dir)
    val_ds = BlastocystDataset(Config.val_csv, Config.img_dir, Config.mask_dir)
    test_ds = BlastocystDataset(Config.test_csv, Config.img_dir, Config.mask_dir)

    # Create data loaders
    def collate_fn(batch):
        imgs = []
        masks = []
        exps = []
        icms = []
        tes = []

        for item in batch:
            imgs.append(img_train_tf(item[0]))
            masks.append(mask_train_tf(item[1]))
            exps.append(item[2])
            icms.append(item[3])
            tes.append(item[4])

        return (
            torch.stack(imgs),
            torch.stack(masks),
            torch.tensor(exps, dtype=torch.long),
            torch.tensor(icms, dtype=torch.long),
            torch.tensor(tes, dtype=torch.long)
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn
    )

    # Initialize model
    model = MultiTaskDINOv2().to(Config.device)
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # # Train model
    # print("Start training...")
    # best_val_acc = train(model, train_loader, val_loader)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(Config.best_model_path))
    evaluate_testset(model, test_loader)

    print(f"\n🎉 Training complete, best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()