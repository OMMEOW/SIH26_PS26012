"""Dataset-level pixel IoU per resolution group on held-out tiles, for one or more checkpoints.
Usage: python -m src.eval_gsd --dir data/processed/ms/test --ckpts checkpoints/best_model.pt checkpoints/best_model_ms.pt"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from src.data.transforms import get_val_transform
from src.models.segmentation import build_model
from src.train import get_device
from src.train_multiscale import PairListDataset, gsd_group, pairs_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--per-group", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = get_device()
    pairs = pairs_in(args.dir)
    by = defaultdict(list)
    for p in pairs:
        by[gsd_group(p[0].name)].append(p)
    rng = random.Random(0)
    sel = []
    for g in sorted(by):
        sel += rng.sample(by[g], min(args.per_group, len(by[g])))
    ld = DataLoader(PairListDataset(sel, get_val_transform()), batch_size=8, num_workers=0)
    results = {}
    for ck in args.ckpts:
        m = build_model(architecture="Unet", encoder="resnet34", encoder_weights=None,
                        in_channels=3, classes=1).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev))
        m.eval()
        inter, union = defaultdict(int), defaultdict(int)
        with torch.no_grad():
            for x, y, names in ld:
                pr = (torch.sigmoid(m(x.to(dev))) > 0.5).cpu().numpy()[:, 0]
                gt = y.numpy()[:, 0] > 0.5
                for p, t, n in zip(pr, gt, names):
                    g = gsd_group(n)
                    inter[g] += int((p & t).sum())
                    union[g] += int((p | t).sum())
        results[ck] = {g: round(inter[g] / max(union[g], 1), 4) for g in sorted(inter)}
        print(ck, results[ck], flush=True)
    if args.out:
        json.dump({"groups": {g: len(v) for g, v in by.items()}, "iou": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
