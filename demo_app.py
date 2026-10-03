"""
SIH 2026 demo app -- PS 26012 (AI-based cadastral parcel mapping from drone imagery).

Streamlit app wired to the real project pipeline:
  src.models.segmentation.build_model          -- CNN segmentation model
  src.postprocess.regularize                   -- baseline + regularized polygon extraction
  src.eval.metrics                             -- pixel IoU, polygon IoU, mean vertex count

Run:
    cd "~/My Stuff/SIH_PS26012"
    streamlit run demo_app.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.postprocess.regularize import (
    mask_to_baseline_polygons,
    mask_to_regularized_polygons,
    polygon_to_pixel_array,
)
from src.eval.metrics import pixel_iou, mean_vertex_count, best_match_polygon_iou

st.set_page_config(page_title="GeoParcelAI -- PS 26012 Demo", layout="wide")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@st.cache_resource
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


st.title("GeoParcelAI -- AI-Based Cadastral Parcel Mapping (PS 26012)")
st.caption(
    "Drone or satellite image -> CNN building-footprint segmentation -> rectilinear "
    "polygon regularization. Parcel candidates are decision-support for field "
    "verification, not automated legal adjudication."
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Public deployment: hide server-side options (checkpoint path etc.) and fail loudly instead of
# showing a placeholder mask. Set GEOPARCEL_PUBLIC=1 in the hosting environment.
PUBLIC = os.environ.get("GEOPARCEL_PUBLIC", "0") == "1"
MAX_PIXELS = int(os.environ.get("GEOPARCEL_MAX_PIXELS", 40_000_000))  # reject larger uploads
Image.MAX_IMAGE_PIXELS = MAX_PIXELS * 2  # PIL refuses decompression bombs beyond this
SAMPLES = {
    "Satellite map crop (Indian city, ~35 cm/pixel)": ("satellite_test_crop.png", "Satellite / map screenshot (~35 cm per pixel)", None),
    "Drone tile, Dar es Salaam (7 cm/pixel, with ground truth)": (
        "drone_tile_0a4c40_7680_2048.png", "Drone tile (~7 cm per pixel)", "ground_truth_mask_0a4c40_7680_2048.png"),
}
# checkpoint paths are relative to this app's folder
MODELS = {
    "Multi-scale (drone + satellite, recommended)": os.path.join("checkpoints", "best_model_ms.pt"),
    "Drone-only (original, 7 cm)": os.path.join("checkpoints", "best_model.pt"),
}
IMAGE_TYPES = {
    "Satellite / map screenshot (~35 cm per pixel)": 35.0,
    "Drone tile (~7 cm per pixel)": TRAIN_GSD_CM,
    "Custom": None,
}

def _on_sample():
    s = st.session_state.get("sample")
    if s in SAMPLES:
        st.session_state["image_type"] = SAMPLES[s][1]


with st.sidebar:
    st.header("Image type")
    st.session_state.setdefault("image_type", list(IMAGE_TYPES)[0])
    image_type = st.radio("What are you uploading?", list(IMAGE_TYPES), key="image_type",
                          help="Buildings must be shown to the model at roughly the scale it learned. "
                               "Picking the image type sets that scale.")
    if IMAGE_TYPES[image_type] is None:
        gsd_cm = st.number_input(
            "Ground resolution of your image (cm per pixel)", min_value=2.0, max_value=200.0, value=TRAIN_GSD_CM,
            step=0.5, help="Drone orthomosaics: 2-10 cm/pixel. Satellite-map screenshots at city-block zoom: 30-60 cm/pixel.")
    else:
        gsd_cm = IMAGE_TYPES[image_type]
    st.divider()
    st.header("Model")
    model_name = st.selectbox("Model", list(MODELS), index=0)
    threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.05)
    checkpoint_path, architecture, encoder = MODELS[model_name], "Unet", "resnet34"
    work_gsd_cm = TRAIN_GSD_CM
    if not PUBLIC:
        with st.expander("Advanced"):
            checkpoint_path = st.text_input("Checkpoint path", value=MODELS[model_name])
            architecture = st.selectbox("Architecture", ["Unet", "DeepLabV3Plus"], index=0)
            encoder = st.text_input("Encoder", value="resnet34")
            work_gsd_cm = st.number_input("Model working resolution (cm per pixel)", min_value=2.0, max_value=100.0,
                                          value=TRAIN_GSD_CM, step=0.5,
                                          help="Images are resampled to this resolution before tiled inference.")
    st.divider()
    st.header("Ground truth (optional)")
    gt_mask_file = st.file_uploader("GT mask (same tile, binary PNG)", type=["png"])

uploaded = st.file_uploader("Upload a drone or satellite image (tile or larger area)", type=["png", "jpg", "jpeg", "tif", "tiff"])
sample = st.selectbox("…or try a sample image", ["—"] + list(SAMPLES), index=0, key="sample", on_change=_on_sample)

source, source_name, geo = None, None, None
if uploaded is not None:
    source, source_name = uploaded, uploaded.name
elif sample in SAMPLES:
    source_name = SAMPLES[sample][0]
    source = os.path.join(APP_DIR, "demo_inputs", source_name)
    if SAMPLES[sample][2] and gt_mask_file is None:
        gt_mask_file = os.path.join(APP_DIR, "demo_inputs", SAMPLES[sample][2])

if source is not None:
    try:
        image = Image.open(source)
        if image.width * image.height > MAX_PIXELS:
            st.error(f"Image is {image.width}×{image.height} px; the limit here is {MAX_PIXELS / 1e6:.0f} megapixels. "
                     "Crop it or run the pipeline locally (see the GitHub README).")
            st.stop()
        image = image.convert("RGB")
    except Image.DecompressionBombError:
        st.error(f"Image is larger than the {MAX_PIXELS / 1e6:.0f} megapixel limit.")
        st.stop()
    except Exception as e:
        st.error(f"Could not read this image ({e}).")
        st.stop()
    image_rgb = np.array(image)

    # GeoTIFF: take the ground resolution and georeference from the file itself
    if source_name and source_name.lower().endswith((".tif", ".tiff")):
        geo = read_geotiff_meta(source)
        if geo is not None:
            gsd_cm = geo["gsd_cm"]
            st.caption(f"GeoTIFF detected: {geo['crs']} · {gsd_cm:.1f} cm/pixel (overrides the image-type setting)")

    try:
        ckpt = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(APP_DIR, checkpoint_path)
        model, device = load_model(ckpt, architecture, encoder)
        import time
        t0 = time.time()
        with st.spinner("Segmenting buildings…"):
            work_mask, f, n_tiles = run_inference(model, device, image_rgb, threshold, gsd_cm, work_gsd_cm)
        elapsed = time.time() - t0
        model_ran = True
    except Exception as e:
        if PUBLIC:
            st.error("The model could not run on this image. Please try a smaller image or one of the samples.")
            print(f"[geoparcel] inference error: {e!r}", flush=True)
            st.stop()
        st.warning(f"Couldn't load/run the trained model ({e}). Showing a placeholder mask so the UI can still be demoed.")
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        work_mask, f, n_tiles, elapsed = (gray > np.percentile(gray, 70)).astype(np.uint8), 1.0, 0, 0.0
        model_ran = False

    h0, w0 = image_rgb.shape[:2]
    pred_mask = work_mask if work_mask.shape[:2] == (h0, w0) else cv2.resize(work_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    # polygons are extracted at the model's working scale, then mapped back to the image
    baseline_polys = scale_polys(mask_to_baseline_polygons(work_mask), f)
    regularized_polys = scale_polys(mask_to_regularized_polygons(work_mask), f)

    if model_ran:
        st.caption(f"Processed at {gsd_cm / f:.1f} cm/pixel working resolution (x{f:.2f}) · {n_tiles} tiles of 512 px · {elapsed:.1f} s")
    if max(h0, w0) > 1024 and abs(gsd_cm - TRAIN_GSD_CM) < 1e-6:
        st.info("This is a large image. If it is a satellite or map screenshot rather than a ~7 cm drone tile, "
                "pick that image type in the sidebar so buildings are shown to the model at the right scale.")
    coarse_limit = 60.0 if "Multi-scale" in model_name else 15.0
    if gsd_cm > coarse_limit:
        st.warning(f"This image is coarser than the imagery this model was trained on (up to ~{coarse_limit:.0f} cm/pixel); "
                   "expect missed buildings. Outputs are candidates for field verification.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Input image")
        st.image(image_rgb, use_container_width=True)
    with col2:
        st.subheader("Raw segmentation mask")
        st.image(pred_mask * 255, use_container_width=True, clamp=True)
    with col3:
        st.subheader("Regularized polygons")
        st.image(draw_polygons(image_rgb, regularized_polys), use_container_width=True)

    st.divider()
    st.subheader("Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline polygon count", len(baseline_polys))
    m2.metric("Regularized polygon count", len(regularized_polys))
    m3.metric(
        "Mean vertex count (baseline -> regularized)",
        f"{mean_vertex_count(baseline_polys):.1f} -> {mean_vertex_count(regularized_polys):.1f}",
    )

    if gt_mask_file is not None:
        gt_mask = np.array(Image.open(gt_mask_file).convert("L"))
        gt_mask = (cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0])) > 127).astype(np.uint8)
        gt_polys = mask_to_baseline_polygons(gt_mask, min_area=10.0)
        c1, c2 = st.columns(2)
        c1.metric("Pixel IoU vs. ground truth", f"{pixel_iou(pred_mask, gt_mask):.4f}")
        c2.metric("Regularized polygon IoU vs. GT", f"{best_match_polygon_iou(regularized_polys, gt_polys):.4f}")

    if model_ran and regularized_polys:
        gj = polygons_to_geojson(regularized_polys, geo)
        stem = os.path.splitext(os.path.basename(source_name or "image"))[0]
        st.download_button(
            "Download parcel candidates (GeoJSON)", data=json.dumps(gj), file_name=f"{stem}_parcel_candidates.geojson",
            mime="application/geo+json",
            help=("Coordinates are in the GeoTIFF's CRS." if geo else
                  "This image has no georeference, so coordinates are image pixels (x right, y down)."))

    if not model_ran:
        st.info(
            "This run used a placeholder threshold mask, not your trained model -- "
            "fix the checkpoint path in the sidebar before recording the demo video."
        )
else:
    st.info("Upload a drone orthomosaic tile or a satellite / map screenshot, and pick its image type in the sidebar. "
            "Large images are processed in overlapping 512-pixel tiles.")
