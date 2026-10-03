"""
Scene-level inference: run the trained segmentation model over a window of a
large GeoTIFF orthomosaic, stitch tile predictions, extract baseline and
regularized polygons, score against ground-truth labels (if given), and export
georeferenced GeoJSON parcel candidates (raster CRS + WGS84).

Usage:
    python -m src.infer_scene --tif scene.tif --labels scene.geojson \
        --col 0 --row 0 --size 8192 --out-dir results/scene_infer
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from shapely.affinity import affine_transform

from src.data.tile_dataset import rasterize_labels
from src.eval.metrics import best_match_polygon_iou, mean_vertex_count, pixel_iou
from src.models.segmentation import build_model
from src.postprocess.regularize import mask_to_baseline_polygons, mask_to_regularized_polygons

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_window(model, device, rgb: np.ndarray, tile: int = 512, batch: int = 8) -> np.ndarray:
    h, w, _ = rgb.shape
    prob = np.zeros((h, w), dtype=np.float32)
    coords = [(r, c) for r in range(0, h - tile + 1, tile) for c in range(0, w - tile + 1, tile)]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        x = np.stack([(rgb[r:r + tile, c:c + tile].astype(np.float32) / 255.0 - MEAN) / STD for r, c in chunk])
        x = torch.from_numpy(x.transpose(0, 3, 1, 2)).float().to(device)
        p = torch.sigmoid(model(x))[:, 0].cpu().numpy()
        for (r, c), pi in zip(chunk, p):
            prob[r:r + tile, c:c + tile] = pi
    return prob


def polys_to_geo(polys, transform):
    # pixel (col,row) -> map (x,y): x = a*col + b*row + c ; y = d*col + e*row + f
    params = [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]
    return [affine_transform(p, params) for p in polys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--col", type=int, default=0)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--size", type=int, default=8192)
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--architecture", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", default="results/scene_infer")
    ap.add_argument("--preview-scale", type=int, default=4)
    ap.add_argument("--min-area-m2", type=float, default=4.0,
                    help="drop candidate polygons smaller than this (slivers/noise)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device()
    model = build_model(args.architecture, args.encoder, encoder_weights=None).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with rasterio.open(args.tif) as src:
        win = Window(args.col, args.row, args.size, args.size)
        rgb = np.transpose(src.read([1, 2, 3], window=win), (1, 2, 0))
        wt = src.window_transform(win)
        crs = src.crs
        gt = rasterize_labels(args.labels, crs, (args.size, args.size), wt) if args.labels else None

    prob = predict_window(model, device, rgb)
    pred = (prob > args.threshold).astype(np.uint8)

    px_area = abs(wt.a * wt.e)  # m^2 per pixel
    min_px = args.min_area_m2 / px_area
    base = mask_to_baseline_polygons(pred, min_area=min_px)
    reg = mask_to_regularized_polygons(pred, min_area=min_px)

    report = {"window": [args.col, args.row, args.size], "crs": str(crs),
              "pixel_size_m": abs(wt.a), "min_area_m2": args.min_area_m2,
              "n_baseline": len(base), "n_regularized": len(reg),
              "baseline_mean_vertices": mean_vertex_count(base),
              "regularized_mean_vertices": mean_vertex_count(reg)}
    if gt is not None:
        gt_polys = mask_to_baseline_polygons(gt, min_area=min_px)
        report.update({"pixel_iou": pixel_iou(pred, gt), "n_gt": len(gt_polys),
                       "baseline_poly_iou": best_match_polygon_iou(base, gt_polys),
                       "regularized_poly_iou": best_match_polygon_iou(reg, gt_polys)})

    geo = polys_to_geo(reg, wt)
    gdf = gpd.GeoDataFrame({"id": range(len(geo)), "area_m2": [g.area for g in geo],
                            "status": ["candidate - needs field verification"] * len(geo)},
                           geometry=geo, crs=crs)
    gdf.to_file(os.path.join(args.out_dir, "parcel_candidates_utm.geojson"), driver="GeoJSON")
    gdf.to_crs("EPSG:4326").to_file(os.path.join(args.out_dir, "parcel_candidates_wgs84.geojson"), driver="GeoJSON")

    s = args.preview_scale
    small = cv2.resize(rgb, (args.size // s, args.size // s), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(args.out_dir, "window_rgb.jpg"), cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(args.out_dir, "window_pred.png"), cv2.resize(pred * 255, (args.size // s, args.size // s), interpolation=cv2.INTER_NEAREST))
    if gt is not None:
        cv2.imwrite(os.path.join(args.out_dir, "window_gt.png"), cv2.resize(gt * 255, (args.size // s, args.size // s), interpolation=cv2.INTER_NEAREST))
    to_list = lambda ps: [[[round(x / s, 1), round(y / s, 1)] for x, y in list(p.exterior.coords)] for p in ps]
    json.dump({"scale": s, "regularized": to_list(reg), "baseline": to_list(base)},
              open(os.path.join(args.out_dir, "polygons_preview.json"), "w"))
    json.dump(report, open(os.path.join(args.out_dir, "report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
