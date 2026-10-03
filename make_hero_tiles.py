
import sys, os
sys.path.insert(0, os.path.expanduser("~/My Stuff/SIH_PS26012"))

import cv2
import numpy as np
import torch
from src.models.segmentation import build_model
from src.postprocess.regularize import (
    mask_to_baseline_polygons, mask_to_regularized_polygons, polygon_to_pixel_array
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = build_model(architecture="Unet", encoder="resnet34", encoder_weights=None).to(device)
ckpt = os.path.expanduser("~/My Stuff/SIH_PS26012/checkpoints/best_model.pt")
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

hero_files = ["0a4c40_1024_5120.png", "0a4c40_10240_3584.png", "0a4c40_10752_11776.png", "0a4c40_13312_1536.png"]
val_dir = os.path.expanduser("~/My Stuff/SIH_PS26012/data/processed/val")
out_dir = os.path.expanduser("~/My Stuff/SIH_PS26012/results/hero_tiles")
os.makedirs(out_dir, exist_ok=True)

for fname in hero_files:
    img_path = os.path.join(val_dir, "images", fname)
    if not os.path.exists(img_path):
        print("missing:", fname)
        continue
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    x = image_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8)

    baseline_polys = mask_to_baseline_polygons(mask)
    regularized_polys = mask_to_regularized_polygons(mask)

    def draw(polys, color):
        out = image_rgb.copy()
        for p in polys:
            pts = polygon_to_pixel_array(p)
            cv2.polylines(out, [pts.astype(np.int32)], True, color, 2)
        return out

    panel_input = image_rgb.copy()
    panel_mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB)
    panel_baseline = draw(baseline_polys, (255, 60, 60))
    panel_regularized = draw(regularized_polys, (60, 200, 60))

    grid = np.concatenate([panel_input, panel_mask, panel_baseline, panel_regularized], axis=1)
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(out_dir, f"hero_{fname}")
    cv2.imwrite(out_path, grid_bgr)
    print(f"saved {out_path} | baseline_verts={np.mean([len(p.exterior.coords)-1 for p in baseline_polys]) if baseline_polys else 0:.1f} "
          f"regularized_verts={np.mean([len(p.exterior.coords)-1 for p in regularized_polys]) if regularized_polys else 0:.1f}")

print("HERO_DONE_MARKER")
