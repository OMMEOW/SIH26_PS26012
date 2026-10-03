"""
Multi-scale tiling: resample drone scenes to a set of target ground resolutions
(cm/pixel) and cut 512x512 image/mask tiles at each one.

Why: the base model only ever saw ~7 cm/pixel drone imagery, so buildings in a
30-50 cm/pixel satellite or map screenshot look nothing like what it learned.
Showing it the same neighbourhoods at coarser resolutions teaches it what a
building looks like when it is 10 pixels wide instead of 100.

Image windows are read with average resampling (GDAL uses the scene overviews),
labels are rasterised directly on the coarse grid (reprojected WGS84 -> UTM,
same CRS fix as tile_dataset.py), no-data tiles are skipped via the alpha band.

Output names: {scene}_g{gsd_cm}_{top}_{left}.png   (top/left on the resampled grid)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box

TILE = 512


def load_labels(geojson_path, crs):
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    return gdf


def tile_scene_at(src, gdf, scene, gsd_cm, out_dir_for_row, stride, min_valid=0.6):
    native = src.res[0]
    s = (gsd_cm / 100.0) / native            # source pixels per output pixel
    W, H = int(src.width / s), int(src.height / s)
    tr = src.transform * Affine.scale(s)       # transform of the resampled grid
    src_tile = TILE * s
    n = 0
    sindex = gdf.sindex
    for top in range(0, H - TILE + 1, stride):
        out_dir = out_dir_for_row(top / H)
        if out_dir is None:
            continue
        for left in range(0, W - TILE + 1, stride):
            win = Window(left * s, top * s, src_tile, src_tile)
            arr = src.read([1, 2, 3, 4], window=win, out_shape=(4, TILE, TILE),
                           resampling=Resampling.average)
            alpha = arr[3]
            if (alpha > 0).mean() < min_valid or arr[:3].max() == 0:
                continue
            img = np.transpose(arr[:3], (1, 2, 0)).copy()
            img[alpha == 0] = 0
            ttr = tr * Affine.translation(left, top)
            x0, y0 = ttr * (0, 0)
            x1, y1 = ttr * (TILE, TILE)
            idx = sindex.query(box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
            geoms = list(gdf.geometry.iloc[idx])
            mask = (rasterize([(g, 1) for g in geoms], out_shape=(TILE, TILE), transform=ttr,
                              fill=0, dtype=np.uint8) if geoms else np.zeros((TILE, TILE), np.uint8))
            mask[alpha == 0] = 0
            fname = f"{scene}_g{gsd_cm:g}_{top}_{left}.png"
            cv2.imwrite(str(out_dir / "images" / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / "masks" / fname), mask * 255)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=os.path.expanduser(
        "~/My Stuff/SIH_PS26012/data/raw/opencities/train_tier_1/dar"))
    ap.add_argument("--out-root", default=os.path.expanduser("~/My Stuff/SIH_PS26012/data/processed/ms"))
    ap.add_argument("--scenes", nargs="+", default=["b15fce", "a017f9"])
    ap.add_argument("--gsds", nargs="+", type=float, default=[7.2, 12, 20, 30, 45])
    ap.add_argument("--val-scene", default="a017f9", help="bottom rows of this scene go to val/")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-name", default=None, help="write everything to this split (e.g. test)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    for sp in (["train", "val"] if args.split_name is None else [args.split_name]):
        for sub in ("images", "masks"):
            (out_root / sp / sub).mkdir(parents=True, exist_ok=True)

    for scene in args.scenes:
        tif = Path(args.raw_root) / scene / f"{scene}.tif"
        gj = Path(args.raw_root) / f"{scene}-labels" / f"{scene}.geojson"
        with rasterio.open(tif) as src:
            gdf = load_labels(gj, src.crs)
            for g in args.gsds:
                stride = TILE if g < 15 else (384 if g < 25 else 256)

                def route(frac_row, scene=scene):
                    if args.split_name is not None:
                        return out_root / args.split_name
                    if scene == args.val_scene:
                        # leave a gap of one tile-height between train and val rows
                        if frac_row >= 1 - args.val_frac:
                            return out_root / "val"
                        if frac_row >= 1 - args.val_frac - 0.03:
                            return None
                    return out_root / "train"

                n = tile_scene_at(src, gdf, scene, g, route, stride)
                print(f"{scene} @ {g:g} cm: {n} tiles", flush=True)


if __name__ == "__main__":
    main()
