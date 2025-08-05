# compute_profile.py
# -----------------------------------------------------------
import torch, time, os
from Albation import MultiTaskDINOv2, Config   # ← change to your actual path

# ---------- 1. Build model ----------
model = MultiTaskDINOv2().to(Config.device)
model.eval()

# ---------- 2. Count parameters ----------
n_params = sum(p.numel() for p in model.parameters())        # total parameters
n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'# Params  : {n_params/1e6:.2f} M')
print(f'└─ Train. : {n_train/1e6:.2f} M  (after freezing policy)')

# ---------- 3. Estimate FLOPs ----------
try:
    from ptflops import get_model_complexity_info

    def input_constructor(input_res):
        bs  = input_res[0]           # batch size
        c,h,w = input_res[1:]
        dummy_img  = torch.randn(bs, c, h, w).to(Config.device)
        dummy_mask = torch.zeros(bs, 3, h, w).to(Config.device)
        # return kwargs, aligned with forward(imgs=..., masks=...)
        return dict(imgs=dummy_img, masks=dummy_mask)

    macs, _ = get_model_complexity_info(
        model,
        (1, 3, 224, 224),           # (batch, C, H, W)
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        input_constructor=input_constructor
    )
    flops = macs * 2               # 1 MAC = 2 FLOPs
    print(f'FLOPs     : {flops/1e9:.2f} G')
except ImportError:
    print('ptflops not installed, skipping FLOPs estimation. pip install ptflops')

# ---------- 4. Single-image inference memory & latency ----------
dummy_x   = torch.randn(1, 3, 224, 224).to(Config.device)
dummy_msk = torch.zeros(1, 3, 224, 224).to(Config.device)    # placeholder mask
torch.cuda.reset_peak_memory_stats(Config.device)

with torch.no_grad():
    tic = time.time()
    _ = model(dummy_x, dummy_msk)
    torch.cuda.synchronize(Config.device)
    toc = time.time()

if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated(Config.device) / (1024**3)
    print(f'GPU Mem   : {peak_mem:.2f} GB  (batch=1)')
print(f'Latency   : {(toc - tic)*1e3:.2f} ms')



# benchmark_mask_gen.py
# ------------------------------------------------------------
import os, random, time, psutil, tracemalloc
import numpy as np, cv2
from fvcore.nn import FlopCountAnalysis, flop_count_table
from fenge import (  # ← change to the .py file that stores your functions
    preprocess_image,
    improved_embryo_mask,
    improved_segment_watershed,
    identify_te_region,
    identify_icm_by_subtraction,
)

IMG_DIR = "Blastocyst_Dataset/Images"
SAMPLE_N = 100                  # sampling 100 images is enough for statistics

# ---------- 1. Sample images ----------
all_imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('png','jpg','jpeg'))]
sampled  = random.sample(all_imgs, min(SAMPLE_N, len(all_imgs)))

# ---------- 2. Runtime & memory ----------
t_total = 0
tracemalloc.start()
for fname in sampled:
    img_path = os.path.join(IMG_DIR, fname)
    tic = time.perf_counter()

    img_rgb, gray  = preprocess_image(img_path)
    emb_mask       = improved_embryo_mask(gray)
    ws_labels      = improved_segment_watershed(gray, emb_mask)
    te_mask        = identify_te_region(ws_labels, gray)
    icm_mask       = identify_icm_by_subtraction(emb_mask, te_mask)

    t_total += (time.perf_counter() - tic)

cur, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"[CPU]   Avg latency : {t_total/len(sampled)*1e3:.2f} ms / img")
print(f"[CPU]   Peak memory : {peak/1024/1024:.1f}  MB")

# ---------- 3. FLOPs (per image) ----------
# Build a forward function and let fvcore count python-call FLOPs
def forward_fn(img_path):
    img_rgb, gray  = preprocess_image(img_path)
    emb_mask       = improved_embryo_mask(gray)
    ws_labels      = improved_segment_watershed(gray, emb_mask)
    _              = identify_te_region(ws_labels, gray)
    _              = identify_icm_by_subtraction(emb_mask, _)

img_path0 = os.path.join(IMG_DIR, sampled[0])
flop_analyser = FlopCountAnalysis(forward_fn, (img_path0,))
print(flop_count_table(flop_analyser))