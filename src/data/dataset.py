"""
Dataset that reads tiled image/mask PNG pairs written by tile_dataset.py
(expects <root>/images/*.png and <root>/masks/*.png with matching filenames).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class BuildingFootprintDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.images_dir = Path(root) / "images"
        self.masks_dir = Path(root) / "masks"
        self.filenames = sorted(p.name for p in self.images_dir.glob("*.png"))
        if not self.filenames:
            raise RuntimeError(f"No tiles found under {self.images_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = cv2.imread(str(self.images_dir / fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks_dir / fname), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0) if hasattr(mask, "unsqueeze") else mask[None, ...]
        else:
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
            mask = mask[None, ...]

        return image, mask, fname
