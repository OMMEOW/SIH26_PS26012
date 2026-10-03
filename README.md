# GeoParcelAI -- PS 26012: AI-Based Automated Urban Parcel Mapping
## SIH 2026 -- Ministry of Rural Development / Dept. of Land Resources

Preliminary parcel candidate generation from drone orthomosaic imagery, for
field verification -- not automated legal adjudication. Flagship module:
CNN building-footprint segmentation -> polygon regularization.

## Pipeline

```
Drone orthomosaic tile (512x512 RGB)
        |
        v
 U-Net (ResNet-34 encoder) segmentation  -->  binary building mask
        |
        v
 Baseline polygon extraction (raw contours -- noisy, stair-stepped)
        |
        v
 Rectilinear regularization (morphology + minimum-rotated-rect snapping)
        |
        v
 Regularized parcel candidate polygons
```

## Dataset

- Source: Open Cities AI Challenge (data.source.coop), Dar es Salaam region.
- 4 scenes, ~4.8GB of GeoTIFF imagery + GeoJSON building-footprint labels.
- Tiled into 512x512 image/mask pairs: 24,926 tiles total across all 4 scenes
  (train run used 6,677 tiles from one scene; a larger second run can use all).
- **Critical fix**: labels are WGS84 (EPSG:4326); raster is UTM (EPSG:32737).
  Labels are reprojected to the raster CRS before rasterization -- without
  this, masks are silently empty.

## Model

- Architecture: U-Net, ResNet-34 encoder (ImageNet-pretrained), via
  `segmentation_models_pytorch`.
- Loss: combined Dice + BCE-with-logits (`DiceBCELoss`), robust to the
  class imbalance of building vs. background pixels.
- Trained on Apple Silicon (MPS backend) -- see `src/train.py`.

## Results (epoch-1 checkpoint, 150 val tiles -- early, training still running)

| Metric | Baseline (raw contours) | Regularized |
|---|---|---|
| Pixel IoU vs GT | 0.644 | 0.644 (same mask; regularization is polygon-level) |
| Polygon IoU vs GT (best-match) | 0.201 | **0.244** (+22% relative) |
| Mean vertex count | 100.0 | **8.0** (12.5x simplification) |

Headline finding: regularization cuts polygon complexity by >12x (noisy
stair-stepped contours -> near-rectangular candidates) while *improving*
polygon-IoU against ground truth, not trading accuracy for simplicity.

These numbers are from the epoch-1 checkpoint (val_iou=0.586) as a sanity
check that the full eval pipeline works end-to-end on real data. Training
continues for 10 epochs total; final numbers will be better -- see `train.log`
for the per-epoch curve and re-run `src/eval/compare.py` against
`checkpoints/best_model.pt` once training finishes for the final table.

## Honest scoping

- Trained on a subset of one region (Dar es Salaam, Open Cities AI Challenge)
  in a 2-day timeline -- metrics reflect an early checkpoint, not a
  production-ready model.
- Regularization is a deterministic morphology + snapping heuristic, not a
  second learned model -- described accurately as post-processing.
- Output is explicitly "preliminary parcel candidates for field verification,"
  matching the problem statement's own framing -- not a substitute for a
  licensed surveyor's cadastral determination.
- Known limitation: Open Cities imagery is African urban building stock;
  generalization to other regions/architectural styles is untested here.

## Running it

```bash
# 1. Tile raw scenes (already done for the 4 downloaded scenes)
python -m src.data.tile_dataset

# 2. Train
python -m src.train --epochs 10 --batch-size 8

# 3. Evaluate baseline vs regularized
python -m src.eval.compare --checkpoint checkpoints/best_model.pt

# 4. Interactive demo
streamlit run demo_app.py
```

## Future work

- Scale to more regions/scenes for better generalization.
- Boundary-aware loss term to sharpen edge localization.
- Active-learning loop: field-verification corrections feed back into
  retraining.
- Full topology validation (no self-intersections, parcel adjacency rules)
  before candidates reach a human reviewer.
# SIH26_PS26012
