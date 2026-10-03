"""
Model factory + loss functions for building-footprint segmentation.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(architecture: str = "Unet", encoder: str = "resnet34",
                 encoder_weights: str | None = "imagenet", in_channels: int = 3,
                 classes: int = 1):
    arch_map = {
        "Unet": smp.Unet,
        "UnetPlusPlus": smp.UnetPlusPlus,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "DeepLabV3": smp.DeepLabV3,
    }
    if architecture not in arch_map:
        raise ValueError(f"Unknown architecture: {architecture}. Choices: {list(arch_map)}")
    model_cls = arch_map[architecture]
    model = model_cls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    return model


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE-with-logits loss. Robust to class imbalance
    (buildings are usually a small fraction of pixels)."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth))
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


@torch.no_grad()
def pixel_iou_batch(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    inter = (preds * targets).sum().item()
    union = ((preds + targets) >= 1).float().sum().item()
    return inter / union if union > 0 else 1.0
