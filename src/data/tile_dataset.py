"""
Tile large GeoTIFF scenes + their GeoJSON building-footprint labels into
fixed-size image/mask PNG pairs for training.

Critical fix: Open Cities AI Challenge GeoJSON labels are in WGS84 lon/lat
(EPSG:4326), while the GeoTIFF raster is in a projected UTM CRS (meters).
Labels MUST be reprojected to the raster's CRS before rasterize() or every
mask comes out silently empty.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window


def find_metadata(region_dir: Path, scene_id: str):
    tif_path = region_dir / scene_id / f"{scene_id}.tif"
    geojson_path = region_dir / f"{scene_id}-labels" / f"{scene_id}.geojson"
    if not tif_path.exists():
        raise FileNotFoundError(f"missing tif: {tif_path}")
    if not geojson_path.exists():
        raise FileNotFoundError(f"missing geojson: {geojson_path}")
    return tif_path, geojson_path


def rasterize_labels(geojson_path: Path, raster_crs, out_shape, transform) -> np.ndarray:
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        return np.zeros(out_shape, dtype=np.uint8)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs) != str(raster_crs):
        gdf = gdf.to_crs(raster_crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)
    mask = rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=False,
    )
    return mask


def tile_scene(tif_path: Path, geojson_path: Path, out_images_dir: Path, out_masks_dir: Path,
                tile_size: int = 512, stride: int = 512, min_building_px: int = 0) -> int:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height
        full_mask = rasterize_labels(geojson_path, src.crs, (height, width), src.transform)
        n_pos_px = int(full_mask.sum())
        print(f"  [{tif_path.stem}] raster {width}x{height}, label pixels={n_pos_px} "
              f"({100.0 * n_pos_px / (width * height):.3f}% positive)")
        if n_pos_px == 0:
            print(f"  WARNING: {tif_path.stem} produced an ENTIRELY EMPTY mask -- check CRS handling.")

        count = 0
        for top in range(0, height - tile_size + 1, stride):
            for left in range(0, width - tile_size + 1, stride):
                window = Window(left, top, tile_size, tile_size)
                img = src.read([1, 2, 3], window=window)
                img = np.transpose(img, (1, 2, 0))
                if img.max() == 0:
                    continue
                mask_tile = full_mask[top:top + tile_size, left:left + tile_size]
                if min_building_px and int(mask_tile.sum()) < min_building_px:
                    continue
                fname = f"{tif_path.stem}_{top}_{left}.png"
                cv2.imwrite(str(out_images_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out_masks_dir / fname), mask_tile * 255)
                count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=os.path.expanduser(
        "~/My Stuff/SIH_PS26012/data/raw/opencities/train_tier_1"))
    ap.add_argument("--out-root", default=os.path.expanduser(
        "~/My Stuff/SIH_PS26012/data/processed"))
    ap.add_argument("--region", default="dar")
    ap.add_argument("--scenes", nargs="+", default=["0a4c40", "353093", "b15fce", "a017f9"])
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_region_dir = Path(args.raw_root) / args.region
    out_root = Path(args.out_root)

    tmp_images = out_root / "_all" / "images"
    tmp_masks = out_root / "_all" / "masks"

    total = 0
    for scene_id in args.scenes:
        tif_path, geojson_path = find_metadata(raw_region_dir, scene_id)
        n = tile_scene(tif_path, geojson_path, tmp_images, tmp_masks,
                        tile_size=args.tile_size, stride=args.stride)
        print(f"  -> {n} tiles from {scene_id}")
        total += n

    print(f"Total tiles: {total}")

    all_files = sorted(os.listdir(tmp_images))
    random.Random(args.seed).shuffle(all_files)
    n_val = max(1, int(len(all_files) * args.val_fraction))
    val_files = set(all_files[:n_val])

    for split in ("train", "val"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "masks").mkdir(parents=True, exist_ok=True)

    for fname in all_files:
        split = "val" if fname in val_files else "train"
        os.replace(tmp_images / fname, out_root / split / "images" / fname)
        os.replace(tmp_masks / fname, out_root / split / "masks" / fname)

    print(f"Split: {len(all_files) - n_val} train / {n_val} val")


if __name__ == "__main__":
    main()
