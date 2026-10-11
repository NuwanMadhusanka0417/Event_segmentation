"""Optical flow losses."""

from __future__ import annotations

import torch


def epe_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """End-point error loss.

    Parameters
    ----------
    pred, target : (B, 2, H, W)
    valid : optional (B, 1, H, W) mask
    """
    diff = pred - target
    epe = torch.sqrt(diff[:, 0].pow(2) + diff[:, 1].pow(2) + 1e-8)
    if valid is not None:
        epe = epe * valid.squeeze(1)
        return epe.sum() / valid.sum().clamp_min(1)
    return epe.mean()


def multi_scale_flow_loss(
    preds: list[torch.Tensor],
    target: torch.Tensor,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Weighted sum of EPE at multiple pyramid levels."""
    if weights is None:
        weights = [0.5, 0.7, 1.0, 1.0][: len(preds)]
    loss = torch.tensor(0.0, device=preds[0].device)
    for pred, w in zip(preds, weights):
        tgt = torch.nn.functional.interpolate(
            target, size=pred.shape[-2:], mode="bilinear", align_corners=True
        )
        loss = loss + w * epe_loss(pred, tgt)
    return loss
