"""
Training loop for the building-footprint segmentation model.

Device selection supports CUDA, Apple Silicon MPS, and CPU fallback.
NOTE: torch.cuda.amp mixed-precision is CUDA-only, so this version skips
AMP on MPS/CPU devices (full fp32) rather than trying to adapt the CUDA
GradScaler/autocast API, which does not map cleanly onto MPS.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BuildingFootprintDataset
from src.data.transforms import get_train_transform, get_val_transform
from src.models.segmentation import build_model, DiceBCELoss, pixel_iou_batch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0
    total_pos_px = 0
    total_px = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device).float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            total_pos_px += int(masks.sum().item())
            total_px += int(masks.numel())

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_iou += pixel_iou_batch(logits, masks)
            n_batches += 1

    pos_pct = 100.0 * total_pos_px / total_px if total_px > 0 else 0.0
    return total_loss / max(n_batches, 1), total_iou / max(n_batches, 1), pos_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.expanduser(
        "~/My Stuff/SIH_PS26012/data/processed"))
    ap.add_argument("--checkpoint-dir", default=os.path.expanduser(
        "~/My Stuff/SIH_PS26012/checkpoints"))
    ap.add_argument("--architecture", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_ds = BuildingFootprintDataset(
        os.path.join(args.data_root, "train"), transform=get_train_transform(args.tile_size))
    val_ds = BuildingFootprintDataset(
        os.path.join(args.data_root, "val"), transform=get_val_transform(args.tile_size))
    print(f"train tiles: {len(train_ds)}, val tiles: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=False,
                               persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=False,
                             persistent_workers=(args.num_workers > 0))

    model = build_model(architecture=args.architecture, encoder=args.encoder,
                         encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_iou, train_pos_pct = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_iou, val_pos_pct = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} ({dt:.1f}s) "
              f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} train_pos%={train_pos_pct:.3f} | "
              f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_pos%={val_pos_pct:.3f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved new best checkpoint (val_iou={val_iou:.4f}) to {ckpt_path}")

    print(f"Training done. Best val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
