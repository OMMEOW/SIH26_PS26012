"""
Multi-scale fine-tune of the existing building-footprint model.

Same architecture and loss as src/train.py. Differences:
  * starts from an existing checkpoint (--init-checkpoint) instead of ImageNet
  * training pool = original 0a4c40 train tiles (7.2 cm) + multi-scale tiles
    from tile_multiscale.py; each epoch draws a fixed number of samples so that
    every resolution group gets a set share
  * stronger photometric augmentation (JPEG, blur, colour shifts, downscale)
    to look more like satellite / web-map screenshots
  * two validation sets: the original 0a4c40 val split (guard against
    forgetting) and the held-out multi-scale val rows; best checkpoint is
    chosen on their mean IoU and written to a NEW file, so best_model.pt
    is untouched.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_val_transform
from src.models.segmentation import build_model, DiceBCELoss
from src.train import get_device, run_epoch


def strong_train_transform():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.75, 1.33), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.4),
        A.RandomGamma(p=0.2),
        A.ToGray(p=0.05),
        A.Downscale(scale_range=(0.5, 0.9), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.Sharpen(p=0.15),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.15),
        A.ImageCompression(quality_range=(40, 90), p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class PairListDataset(Dataset):
    """(image_path, mask_path) list -> (image, mask, name)."""

    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        m = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        a = self.transform(image=img, mask=m)
        mask = a["mask"]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return a["image"], mask, Path(ip).name


def pairs_in(root):
    root = Path(root)
    return [(p, root / "masks" / p.name) for p in sorted((root / "images").glob("*.png"))]


def gsd_group(name):
    m = re.search(r"_g([0-9.]+)_", name)
    return f"g{m.group(1)}" if m else "g7.2_0a4c40"


def subsample(pairs, n, seed=0, balance=True):
    if len(pairs) <= n:
        return pairs
    rng = random.Random(seed)
    if not balance:
        return rng.sample(pairs, n)
    by = {}
    for p in pairs:
        by.setdefault(gsd_group(p[0].name), []).append(p)
    k = n // len(by)
    out = []
    for g in sorted(by):
        out += rng.sample(by[g], min(k, len(by[g])))
    return out


def main():
    ap = argparse.ArgumentParser()
    root = os.path.expanduser("~/My Stuff/SIH_PS26012")
    ap.add_argument("--base-train", default=f"{root}/data/processed/train")
    ap.add_argument("--base-val", default=f"{root}/data/processed/val")
    ap.add_argument("--ms-root", default=f"{root}/data/processed/ms")
    ap.add_argument("--init-checkpoint", default=f"{root}/checkpoints/best_model.pt")
    ap.add_argument("--out-checkpoint", default=f"{root}/checkpoints/best_model_ms.pt")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--samples-per-epoch", type=int, default=6000)
    ap.add_argument("--val-per-set", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--native-share", type=float, default=0.4,
                    help="share of each epoch drawn at 7.2 cm (half original scene, half new scenes)")
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(0)
    print(f"Using device: {device}", flush=True)

    base = pairs_in(args.base_train)
    ms = pairs_in(Path(args.ms_root) / "train")
    pool = base + ms
    groups = [gsd_group(p[0].name) for p in pool]
    cnt = Counter(groups)
    coarse = sorted(g for g in cnt if g not in ("g7.2", "g7.2_0a4c40"))
    share = {"g7.2_0a4c40": args.native_share / 2, "g7.2": args.native_share / 2}
    for g in coarse:
        share[g] = (1 - args.native_share) / len(coarse)
    weights = [share[g] / cnt[g] for g in groups]
    print("train pool:", dict(sorted(cnt.items())), flush=True)
    print("epoch share:", {k: round(v, 3) for k, v in sorted(share.items())}, flush=True)

    train_ds = PairListDataset(pool, strong_train_transform())
    sampler = WeightedRandomSampler(weights, num_samples=args.samples_per_epoch, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)

    vt = get_val_transform()
    val_sets = {
        "orig_val": subsample(pairs_in(args.base_val), args.val_per_set, balance=False),
        "ms_val": subsample(pairs_in(Path(args.ms_root) / "val"), args.val_per_set),
    }
    val_loaders = {k: DataLoader(PairListDataset(v, vt), batch_size=args.batch_size, num_workers=0)
                   for k, v in val_sets.items()}
    print("val:", {k: len(v) for k, v in val_sets.items()}, flush=True)

    model = build_model(architecture="Unet", encoder="resnet34", encoder_weights=None,
                        in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)

    def evaluate(tag):
        res = {}
        for k, ld in val_loaders.items():
            _, iou, _ = run_epoch(model, ld, criterion, optimizer, device, train=False)
            res[k] = iou
        print(f"{tag}: " + " ".join(f"{k}={v:.4f}" for k, v in res.items())
              + f" mean={np.mean(list(res.values())):.4f}", flush=True)
        return float(np.mean(list(res.values())))

    best = evaluate("Epoch 0 (initial checkpoint)")
    torch.save(model.state_dict(), args.out_checkpoint)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ti, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        sched.step()
        score = evaluate(f"Epoch {ep}/{args.epochs} ({time.time() - t0:.0f}s) train_loss={tl:.4f} train_iou={ti:.4f} |")
        if score > best:
            best = score
            torch.save(model.state_dict(), args.out_checkpoint)
            print(f"  -> saved {args.out_checkpoint} (mean val IoU {score:.4f})", flush=True)
    print(f"Done. Best mean val IoU {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
