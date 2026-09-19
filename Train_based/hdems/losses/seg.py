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
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft Dice loss for class imbalance.

    ``mask`` : optional (B, H, W) bool — restrict Dice to those pixels (e.g. the
    event pixels), so empty background does not dominate the overlap terms.
    """
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target.clamp(0, num_classes - 1), num_classes)
    target_oh = target_oh.permute(0, 3, 1, 2).float()
    if mask is not None:
        m = mask.unsqueeze(1).to(probs.dtype)
        probs = probs * m
        target_oh = target_oh * m
    intersection = (probs * target_oh).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    dice = (2 * intersection + 1) / (union + 1)
    return 1 - dice.mean()
