"""
Orchestrates baseline-vs-regularized comparison over a validation set and
dumps a results table (CSV) + aggregate summary.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from src.eval.metrics import (
    pixel_iou,
    best_match_polygon_iou,
    boundary_f1,
    mean_hausdorff,
    mean_vertex_count,
)
from src.models.segmentation import build_model
from src.postprocess.regularize import mask_to_baseline_polygons, mask_to_regularized_polygons

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str, architecture: str, encoder: str, device):
    model = build_model(architecture=architecture, encoder=encoder,
                         encoder_weights=None, in_channels=3, classes=1).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model, device, image_rgb: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    x = image_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (prob > threshold).astype(np.uint8)


def gt_polys_from_mask(mask: np.ndarray):
    return mask_to_baseline_polygons(mask, min_area=10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", default=os.path.expanduser(
        "~/My Stuff/SIH_PS26012/data/processed/val"))
    ap.add_argument("--checkpoint", default=os.path.expanduser(
        "~/My Stuff/SIH_PS26012/checkpoints/best_model.pt"))
    ap.add_argument("--architecture", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-csv", default=os.path.expanduser(
        "~/My Stuff/SIH_PS26012/results/compare_results.csv"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all val tiles")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    model = load_model(args.checkpoint, args.architecture, args.encoder, device)

    images_dir = Path(args.val_dir) / "images"
    masks_dir = Path(args.val_dir) / "masks"
    fnames = sorted(os.listdir(images_dir))
    if args.limit:
        fnames = fnames[: args.limit]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows = []

    for fname in fnames:
        image_bgr = cv2.imread(str(images_dir / fname))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = (cv2.imread(str(masks_dir / fname), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

        pred_mask = predict_mask(model, device, image_rgb, args.threshold)

        baseline_polys = mask_to_baseline_polygons(pred_mask)
        regularized_polys = mask_to_regularized_polygons(pred_mask)
        gt_polys = gt_polys_from_mask(gt_mask)

        row = {
            "file": fname,
            "pixel_iou": pixel_iou(pred_mask, gt_mask),
            "baseline_poly_iou": best_match_polygon_iou(baseline_polys, gt_polys),
            "regularized_poly_iou": best_match_polygon_iou(regularized_polys, gt_polys),
            "baseline_boundary_f1": boundary_f1(pred_mask, gt_mask),
            "regularized_boundary_f1": boundary_f1(pred_mask, gt_mask),  # same mask source; polygon-level diff is in vertex/IoU
            "baseline_mean_vertices": mean_vertex_count(baseline_polys),
            "regularized_mean_vertices": mean_vertex_count(regularized_polys),
            "baseline_hausdorff": mean_hausdorff(baseline_polys, gt_polys),
            "regularized_hausdorff": mean_hausdorff(regularized_polys, gt_polys),
            "n_baseline_polys": len(baseline_polys),
            "n_regularized_polys": len(regularized_polys),
            "n_gt_polys": len(gt_polys),
        }
        rows.append(row)

    if not rows:
        print("No validation tiles found -- nothing to compare.")
        return

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print()
    print("=== Aggregate summary ===")
    for key in ("pixel_iou", "baseline_poly_iou", "regularized_poly_iou",
                "baseline_mean_vertices", "regularized_mean_vertices"):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
        if vals:
            print(f"  {key}: mean={np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
