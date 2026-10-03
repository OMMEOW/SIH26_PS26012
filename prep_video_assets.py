
import os, shutil, random, tarfile, json
import numpy as np, cv2, rasterio, geopandas as gpd
from rasterio.enums import Resampling
from rasterio.features import rasterize
from affine import Affine

P = os.path.expanduser("~/My Stuff/SIH_PS26012")
A = os.path.join(P, "video_assets")
if os.path.exists(A): shutil.rmtree(A)
os.makedirs(A)

# code + checkpoint + results
shutil.copytree(os.path.join(P, "src"), os.path.join(A, "src"), ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(os.path.join(P, "demo_app.py"), A)
os.makedirs(os.path.join(A, "checkpoints"))
shutil.copy(os.path.join(P, "checkpoints", "best_model.pt"), os.path.join(A, "checkpoints"))
shutil.copytree(os.path.join(P, "results"), os.path.join(A, "results"))
shutil.copy(os.path.join(P, "train.log"), A)
print("copied code/ckpt/results", flush=True)

# val subset
vi = os.path.join(P, "data/processed/val/images"); vm = os.path.join(P, "data/processed/val/masks")
files = sorted(os.listdir(vi)); random.Random(7).shuffle(files)
hero = ["0a4c40_1024_5120.png","0a4c40_10240_3584.png","0a4c40_10752_11776.png","0a4c40_13312_1536.png"]
pick = list(dict.fromkeys(hero + files[:500]))
for sub in ("images","masks"): os.makedirs(os.path.join(A,"val",sub))
for f in pick:
    shutil.copy(os.path.join(vi,f), os.path.join(A,"val/images",f))
    shutil.copy(os.path.join(vm,f), os.path.join(A,"val/masks",f))
print("val tiles:", len(pick), flush=True)

# scene overview (decimated) + label overlay
tif = os.path.join(P, "data/raw/opencities/train_tier_1/dar/0a4c40/0a4c40.tif")
gj  = os.path.join(P, "data/raw/opencities/train_tier_1/dar/0a4c40-labels/0a4c40.geojson")
with rasterio.open(tif) as src:
    for name, fac in (("overview", 16), ("mid", 4)):
        if name == "overview":
            win = rasterio.windows.Window(0, 0, src.width, src.height)
        else:
            # 8192x8192 window around a dense area near hero tile 1024_5120 region
            win = rasterio.windows.Window(2048, 0, 8192, 8192)
        oh, ow = int(win.height // fac), int(win.width // fac)
        img = src.read([1,2,3], window=win, out_shape=(3, oh, ow), resampling=Resampling.average)
        img = np.transpose(img, (1,2,0))
        cv2.imwrite(os.path.join(A, f"scene_{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        wt = src.window_transform(win) * Affine.scale(win.width/ow, win.height/oh)
        gdf = gpd.read_file(gj).to_crs(src.crs)
        m = rasterize([(g,1) for g in gdf.geometry if g is not None and not g.is_empty], out_shape=(oh,ow), transform=wt, fill=0, dtype=np.uint8)
        cv2.imwrite(os.path.join(A, f"scene_{name}_labels.png"), m*255)
        print(name, img.shape, "label%", round(100*m.mean(),2), flush=True)
    meta = {"crs": str(src.crs), "res_m": src.res[0], "width": src.width, "height": src.height}
json.dump(meta, open(os.path.join(A,"scene_meta.json"),"w"))

with tarfile.open(os.path.join(P, "video_assets.tar"), "w") as t:
    t.add(A, arcname="video_assets")
print("size MB", os.path.getsize(os.path.join(P,"video_assets.tar"))/1e6, flush=True)
print("PREP_DONE", flush=True)
