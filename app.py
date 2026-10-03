"""
GeoParcel AI - Gradio web app (used for the public Hugging Face Space).

Same pipeline as the Streamlit demo (demo_app.py): scale-aware tiled U-Net inference ->
rectilinear polygon regularisation -> GeoJSON export. Run locally with:  python app.py
Model weights are expected in checkpoints/ (download from the GitHub release weights-v1).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from src.app_core import (TRAIN_GSD_CM, draw_polygons, load_model, polygons_to_geojson, read_geotiff_meta,
                          run_inference, scale_polys)
from src.eval.metrics import best_match_polygon_iou, mean_vertex_count, pixel_iou
from src.postprocess.regularize import mask_to_baseline_polygons, mask_to_regularized_polygons

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.environ.get("GEOPARCEL_CKPT_DIR", os.path.join(APP_DIR, "checkpoints"))
MAX_PIXELS = int(os.environ.get("GEOPARCEL_MAX_PIXELS", 40_000_000))
Image.MAX_IMAGE_PIXELS = MAX_PIXELS * 2

MODELS = {
    "Multi-scale (drone + satellite, recommended)": "best_model_ms.pt",
    "Drone-only (original, 7 cm)": "best_model.pt",
}
IMAGE_TYPES = {
    "Satellite / map screenshot (~35 cm per pixel)": 35.0,
    "Drone tile (~7 cm per pixel)": TRAIN_GSD_CM,
    "Custom": None,
}
SAT, DRONE = list(IMAGE_TYPES)[0], list(IMAGE_TYPES)[1]
DEMO = os.path.join(APP_DIR, "demo_inputs")
EXAMPLES = [
    [os.path.join(DEMO, "satellite_test_crop.png"), SAT, None],
    [os.path.join(DEMO, "drone_tile_0a4c40_7680_2048.png"), DRONE, os.path.join(DEMO, "ground_truth_mask_0a4c40_7680_2048.png")],
    [os.path.join(DEMO, "drone_tile_0a4c40_18944_23552.png"), DRONE, os.path.join(DEMO, "ground_truth_mask_0a4c40_18944_23552.png")],
]

_models, _lock = {}, threading.Lock()


def get_model(name):
    with _lock:
        if name not in _models:
            _models[name] = load_model(os.path.join(CKPT_DIR, MODELS[name]), "Unet", "resnet34")
        return _models[name]


def segment(image_path, image_type, custom_gsd, model_name, threshold, gt_path, progress=gr.Progress()):
    if not image_path:
        raise gr.Error("Upload an image or pick one of the examples.")
    try:
        im = Image.open(image_path)
        if im.width * im.height > MAX_PIXELS:
            raise gr.Error(f"Image is {im.width}x{im.height} px; the limit here is {MAX_PIXELS / 1e6:.0f} megapixels. "
                           "Crop it, or run the pipeline locally (see the GitHub README).")
        rgb = np.array(im.convert("RGB"))
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Could not read this image ({e}).")

    gsd = IMAGE_TYPES[image_type] if IMAGE_TYPES[image_type] is not None else float(custom_gsd)
    notes = []
    geo = read_geotiff_meta(image_path) if image_path.lower().endswith((".tif", ".tiff")) else None
    if geo is not None:
        gsd = geo["gsd_cm"]
        notes.append(f"GeoTIFF detected ({geo['crs']}, {gsd:.1f} cm/pixel); the file's own resolution is used and the "
                     "GeoJSON is in its coordinate system.")

    progress(0.1, desc="Loading model")
    model, device = get_model(model_name)
    progress(0.3, desc="Segmenting buildings")
    t0 = time.time()
    work_mask, f, n_tiles = run_inference(model, device, rgb, threshold, gsd, TRAIN_GSD_CM)
    elapsed = time.time() - t0
    progress(0.8, desc="Regularising polygons")
    h0, w0 = rgb.shape[:2]
    mask = work_mask if work_mask.shape[:2] == (h0, w0) else cv2.resize(work_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    base = scale_polys(mask_to_baseline_polygons(work_mask), f)
    reg = scale_polys(mask_to_regularized_polygons(work_mask), f)

    lines = [f"**{len(reg)} parcel candidates** · mean vertices per polygon "
             f"{mean_vertex_count(base):.1f} (raw contour) → {mean_vertex_count(reg):.1f} (regularised)",
             f"Processed at {gsd / f:.1f} cm/pixel working resolution (×{f:.2f}) · {n_tiles} tiles · {elapsed:.1f} s"]
    if gt_path:
        gt = np.array(Image.open(gt_path).convert("L"))
        gt = (cv2.resize(gt, (w0, h0)) > 127).astype(np.uint8)
        lines.append(f"vs ground truth: pixel IoU **{pixel_iou(mask, gt):.3f}** · regularised polygon IoU "
                     f"**{best_match_polygon_iou(reg, mask_to_baseline_polygons(gt, min_area=10.0)):.3f}**")
    limit = 60.0 if "Multi-scale" in model_name else 15.0
    if gsd > limit:
        notes.append(f"This image is coarser than this model's training data (up to ~{limit:.0f} cm/pixel); expect missed buildings.")
    md = "\n\n".join(lines + [f"> {n}" for n in notes])

    out_json = None
    if reg:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_json = os.path.join(tempfile.mkdtemp(), f"{stem}_parcel_candidates.geojson")
        with open(out_json, "w") as fh:
            json.dump(polygons_to_geojson(reg, geo), fh)
    return draw_polygons(rgb, reg), (mask * 255).astype(np.uint8), md, out_json


CSS = ".gradio-container {max-width: 1400px !important}"
with gr.Blocks(title="GeoParcel AI · PS 26012") as demo:
    gr.Markdown(
        "# GeoParcel AI — building footprints → parcel candidates\n"
        "SIH 2026 · PS 26012 (Ministry of Rural Development, Department of Land Resources). "
        "Upload a drone orthomosaic tile or a satellite-map screenshot, or pick an example below. A U-Net segments building "
        "footprints, the polygons are regularised into right-angled shapes, and you can download them as GeoJSON.\n\n"
        "Outputs are **preliminary parcel candidates for field verification**, not a legal land record. "
        "[Code, training and evaluation on GitHub](https://github.com/OMMEOW/SIH26_PS26012)")
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            image_in = gr.File(label="Image (PNG, JPG or GeoTIFF)", file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                               type="filepath", height=120)
            image_type = gr.Radio(list(IMAGE_TYPES), value=SAT, label="What are you uploading?",
                                  info="Sets the scale at which buildings are shown to the model.")
            custom_gsd = gr.Number(value=TRAIN_GSD_CM, label="Custom ground resolution (cm per pixel)", visible=False,
                                   minimum=2, maximum=200)
            model_name = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="Model")
            threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Prediction threshold")
            gt_in = gr.File(label="Ground-truth mask (optional, binary PNG)", file_types=[".png"], type="filepath", height=80)
            run = gr.Button("Detect buildings", variant="primary")
        with gr.Column(scale=2):
            with gr.Row():
                overlay = gr.Image(label="Regularised parcel candidates", interactive=False)
                mask_out = gr.Image(label="Raw segmentation mask", interactive=False)
            summary = gr.Markdown()
            geojson = gr.File(label="Download parcel candidates (GeoJSON)")
    gr.Examples(EXAMPLES, inputs=[image_in, image_type, gt_in], label="Examples")
    image_type.change(lambda t: gr.update(visible=IMAGE_TYPES[t] is None), image_type, custom_gsd)
    run.click(segment, [image_in, image_type, custom_gsd, model_name, threshold, gt_in],
              [overlay, mask_out, summary, geojson], concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
                                   server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860))),
                                   max_file_size="25mb", css=CSS)
