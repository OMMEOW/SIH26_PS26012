import sys, os, time
sys.path.insert(0, os.path.expanduser("~/My Stuff/SIH_PS26012"))
from src.data.tile_dataset import tile_scene, find_metadata
from pathlib import Path

raw_region_dir = Path(os.path.expanduser("~/My Stuff/SIH_PS26012/data/raw/opencities/train_tier_1/dar"))
out_images = Path(os.path.expanduser("~/My Stuff/SIH_PS26012/data/processed/_all2/images"))
out_masks = Path(os.path.expanduser("~/My Stuff/SIH_PS26012/data/processed/_all2/masks"))

remaining_scenes = ["353093", "b15fce", "a017f9"]
total = 0
for scene_id in remaining_scenes:
    tif_path, geojson_path = find_metadata(raw_region_dir, scene_id)
    t0 = time.time()
    n = tile_scene(tif_path, geojson_path, out_images, out_masks, tile_size=512, stride=512)
    print(f"{scene_id}: {n} tiles in {time.time()-t0:.1f}s", flush=True)
    total += n
print("TOTAL new tiles:", total, flush=True)
print("TILE_DONE_MARKER", flush=True)
