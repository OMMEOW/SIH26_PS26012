"""Hugging Face Space entry point for GeoParcel AI.

Fetches the app code from the GitHub repo at a pinned commit and the model weights from the
GitHub release (MD5-checked), then runs the repo's app.py. To deploy a new version, change REF.
"""
import hashlib
import os
import runpy
import sys
import urllib.request

REPO = "OMMEOW/SIH26_PS26012"
REF = os.environ.get("GEOPARCEL_REF", "__REF__")
WEIGHTS = f"https://github.com/{REPO}/releases/download/weights-v1"
CKPTS = {"best_model_ms.pt": "3e66c7ade418a33cbf02f403c79837ae", "best_model.pt": "870e697f60a50139edb226444df9d590"}
FILES = [
    "app.py", "src/__init__.py", "src/app_core.py", "src/eval/__init__.py", "src/eval/metrics.py",
    "src/models/__init__.py", "src/models/segmentation.py", "src/postprocess/__init__.py", "src/postprocess/regularize.py",
    "demo_inputs/satellite_test_crop.png",
    "demo_inputs/drone_tile_0a4c40_7680_2048.png", "demo_inputs/ground_truth_mask_0a4c40_7680_2048.png",
    "demo_inputs/drone_tile_0a4c40_18944_23552.png", "demo_inputs/ground_truth_mask_0a4c40_18944_23552.png",
]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geoparcel")


def fetch(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    os.replace(tmp, dest)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


for f in FILES:
    dest = os.path.join(ROOT, f)
    if not os.path.exists(dest):
        fetch(f"https://raw.githubusercontent.com/{REPO}/{REF}/{f}", dest)
for name, digest in CKPTS.items():
    dest = os.path.join(ROOT, "checkpoints", name)
    if not os.path.exists(dest) or md5(dest) != digest:
        print(f"downloading {name}", flush=True)
        fetch(f"{WEIGHTS}/{name}", dest)
        if md5(dest) != digest:
            raise SystemExit(f"checksum mismatch for {name}")

os.chdir(ROOT)
sys.path.insert(0, ROOT)
runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
