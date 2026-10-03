
import os, time, requests, sys

BASE = os.path.expanduser("~/My Stuff/SIH_PS26012/data/raw/opencities/train_tier_1/dar")
os.makedirs(BASE, exist_ok=True)

SELECTED_SCENES = ["0a4c40", "353093", "b15fce", "a017f9"]
BASE_URL = "https://data.source.coop/open-cities/ai-challenge/train_tier_1/dar"

def log(msg):
    with open(os.path.expanduser("~/My Stuff/SIH_PS26012/download.log"), "a") as f:
        f.write(msg + "\n")

def stream_download(url, dest_path, expected_size=None):
    if os.path.exists(dest_path) and expected_size and os.path.getsize(dest_path) == expected_size:
        log(f"  already OK: {os.path.basename(dest_path)} ({expected_size/1e6:.1f} MB)")
        return True
    t0 = time.time()
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        written = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
    ok = (expected_size is None) or (written == expected_size)
    dt = time.time() - t0
    log(f"  {chr(39)}OK{chr(39) if ok else chr(39)+chr(32)+chr(39)}: {os.path.basename(dest_path)} {written/1e6:.1f} MB in {dt:.1f}s ({written/1e6/max(dt,0.1):.1f} MB/s)" if ok else f"  SIZE MISMATCH: {os.path.basename(dest_path)}")
    return ok

log(f"=== download run started {time.ctime()} ===")
for scene in SELECTED_SCENES:
    scene_dir = os.path.join(BASE, scene)
    labels_dir = os.path.join(BASE, f"{scene}-labels")
    os.makedirs(scene_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    tif_url = f"{BASE_URL}/{scene}/{scene}.tif"
    geojson_url = f"{BASE_URL}/{scene}-labels/{scene}.geojson"

    head = requests.head(tif_url, timeout=20)
    expected = int(head.headers.get("Content-Length", 0))
    log(f"[{scene}] downloading tif ({expected/1e6:.1f} MB expected)...")
    stream_download(tif_url, os.path.join(scene_dir, f"{scene}.tif"), expected)

    log(f"[{scene}] downloading geojson...")
    stream_download(geojson_url, os.path.join(labels_dir, f"{scene}.geojson"))

log(f"=== download run finished {time.ctime()} ===")
log("DONE_MARKER")
