"""Segmentation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def seg_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Cross-entropy segmentation loss.

    Parameters
    ----------
    logits : (B, C, H, W)
    target : (B, H, W) long
    """
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Soft Dice loss for class imbalance."""
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    intersection = (probs * target_oh).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    dice = (2 * intersection + 1) / (union + 1)
    return 1 - dice.mean()
