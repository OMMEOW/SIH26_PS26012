"""Inference core shared by the Streamlit demo (demo_app.py) and the Gradio Space (app.py):
model loading, scale-aware tiled inference, polygon scaling/drawing, GeoTIFF metadata and GeoJSON export."""
from __future__ import annotations

import os

import cv2
import numpy as np

from src.postprocess.regularize import polygon_to_pixel_array

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(checkpoint_path: str, architecture: str, encoder: str):
    import torch
    from src.models.segmentation import build_model

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    model = build_model(architecture=architecture, encoder=encoder,
                         encoder_weights=None, in_channels=3, classes=1).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


TILE = 512
TRAIN_GSD_CM = 7.2  # ground resolution of the training imagery (Open Cities, Dar es Salaam)


def tiled_predict(model, device, rgb: np.ndarray, tile: int = TILE, overlap: int = 64, batch: int = 8) -> np.ndarray:
    """Sliding-window inference at the image's own resolution (no squashing).
    Overlapping 512 px tiles; probabilities averaged where tiles overlap."""
    import torch

    h, w, _ = rgb.shape
    H, W = max(h, tile), max(w, tile)
    pad = np.pad(rgb, ((0, H - h), (0, W - w), (0, 0)), mode="reflect") if (H, W) != (h, w) else rgb
    step = tile - overlap
    ys = list(range(0, H - tile + 1, step)) or [0]
    xs = list(range(0, W - tile + 1, step)) or [0]
    if ys[-1] != H - tile:
        ys.append(H - tile)
    if xs[-1] != W - tile:
        xs.append(W - tile)
    coords = [(y, x) for y in ys for x in xs]
    prob = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    with torch.no_grad():
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            xb = np.stack([(pad[y:y + tile, x:x + tile].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                           for y, x in chunk])
            xb = torch.from_numpy(xb.transpose(0, 3, 1, 2)).float().to(device)
            pb = torch.sigmoid(model(xb))[:, 0].cpu().numpy()
            for (y, x), p in zip(chunk, pb):
                prob[y:y + tile, x:x + tile] += p
                cnt[y:y + tile, x:x + tile] += 1
    return (prob / np.maximum(cnt, 1))[:h, :w], len(coords)


def run_inference(model, device, image_rgb: np.ndarray, threshold: float, gsd_cm: float,
                  work_gsd_cm: float, max_side: int = 4096):
    """Resample the image to the model's working resolution, run tiled inference,
    and return (mask at working scale, scale factor, number of tiles)."""
    f = gsd_cm / work_gsd_cm
    h, w, _ = image_rgb.shape
    f = min(f, max_side / max(h, w)) if f > 1 else f
    f = max(f, 0.25)
    if abs(f - 1) < 0.02:  # e.g. a 7.22 cm GeoTIFF: not worth resampling
        f = 1.0
    work = image_rgb if abs(f - 1) < 1e-3 else cv2.resize(
        image_rgb, (max(1, round(w * f)), max(1, round(h * f))),
        interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
    prob, n_tiles = tiled_predict(model, device, work)
    return (prob > threshold).astype(np.uint8), f, n_tiles


def scale_polys(polys, f):
    from shapely.affinity import scale
    if abs(f - 1) < 1e-3:
        return polys
    return [scale(p, xfact=1 / f, yfact=1 / f, origin=(0, 0)) for p in polys]


def read_geotiff_meta(source):
    """Return {'transform', 'crs', 'gsd_cm'} for a georeferenced GeoTIFF, else None."""
    try:
        import rasterio
    except ImportError:
        return None
    try:
        data = source.getvalue() if hasattr(source, "getvalue") else open(source, "rb").read()
        with rasterio.MemoryFile(data) as mf, mf.open() as ds:
            if ds.crs is None or ds.transform.is_identity:
                return None
            if not ds.crs.is_projected:
                return None  # degrees per pixel cannot be turned into cm/pixel reliably
            return {"transform": ds.transform, "crs": ds.crs.to_string(), "gsd_cm": abs(ds.transform.a) * 100.0}
    except Exception:
        return None


def polygons_to_geojson(polys, geo=None):
    from shapely.geometry import mapping
    from shapely.affinity import affine_transform
    feats = []
    for i, p in enumerate(polys):
        if geo is not None:
            t = geo["transform"]
            p = affine_transform(p, [t.a, t.b, t.d, t.e, t.c, t.f])
        feats.append({"type": "Feature", "properties": {"id": i, "status": "candidate - needs field verification"},
                      "geometry": mapping(p)})
    gj = {"type": "FeatureCollection", "features": feats}
    if geo is not None:
        gj["crs"] = {"type": "name", "properties": {"name": geo["crs"]}}
    return gj


def draw_polygons(image_rgb: np.ndarray, polygons, color=(45, 212, 191), thickness=3) -> np.ndarray:
    out = image_rgb.copy()
    for poly in polygons:
        pts = polygon_to_pixel_array(poly)
        cv2.polylines(out, [pts.astype(np.int32)], isClosed=True, color=color, thickness=thickness)
    return out
