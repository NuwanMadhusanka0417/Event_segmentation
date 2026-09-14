"""Segmentation metrics (shared by eval and ridge fitting)."""

from __future__ import annotations

import torch


def mean_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    valid: torch.Tensor | None = None,
) -> float:
    """Per-class IoU for classes ``1..num_classes-1``, averaged."""
    if valid is None:
        valid = target >= 0
    ious: list[float] = []
    for cls in range(1, num_classes):
        pred_c = pred == cls
        tgt_c = target == cls
        union = (pred_c | tgt_c) & valid
        if union.any():
            inter = (pred_c & tgt_c) & valid
            ious.append(inter.sum().float().item() / union.sum().float().item())
    return sum(ious) / max(len(ious), 1)
