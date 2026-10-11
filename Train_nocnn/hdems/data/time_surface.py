"""Per-polarity accumulative time surfaces and pyramid builder."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def events_to_time_surface(
    events: torch.Tensor,
    height: int,
    width: int,
    *,
    polarity: bool = True,
    decay: float = 0.8,
) -> torch.Tensor:
    """Convert events to accumulative time surface(s).

    Parameters
    ----------
    events : (N, 4)  columns [t, x, y, p]
    height, width : sensor resolution

    Returns
    -------
    surface : (C, H, W) float32  C=2 if polarity else 1
    """
    C = 2 if polarity else 1
    surface = torch.zeros(C, height, width, dtype=torch.float32)
    if events.numel() == 0:
        return surface

    t = events[:, 0].to(torch.float64)
    x = events[:, 1].long()
    y = events[:, 2].long()
    t0 = t.min()
    val = (decay ** (t - t0)).to(torch.float32)             # per-event contribution

    c = events[:, 3].long().clamp_(0, 1) if polarity else torch.zeros_like(x)
    inb = (x >= 0) & (x < width) & (y >= 0) & (y < height)   # drop out-of-bounds

    flat_idx = (c[inb] * height + y[inb]) * width + x[inb]
    surface.view(-1).index_add_(0, flat_idx, val[inb])       # scatter-accumulate
    return surface


def build_pyramid(
    surface: torch.Tensor,
    levels: int = 4,
) -> list[torch.Tensor]:
    """Build 4-level spatial pyramid via average pooling.

    Parameters
    ----------
    surface : (C, H, W)

    Returns
    -------
    list of (C, H_l, W_l) tensors, finest first
    """
    pyramid = [surface]
    current = surface.unsqueeze(0)
    for _ in range(levels - 1):
        current = F.avg_pool2d(current, kernel_size=2, stride=2)
        pyramid.append(current.squeeze(0))
    return pyramid
