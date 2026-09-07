"""Data augmentation transforms."""

from __future__ import annotations

import torch


def random_flip(
    surface: torch.Tensor,
    flow: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Random horizontal flip of surface and flow."""
    if torch.rand(1).item() > 0.5:
        surface = torch.flip(surface, dims=[-1])
        if flow is not None:
            flow = torch.flip(flow, dims=[-1])
            flow[0] = -flow[0]
    return surface, flow


def normalize_surface(surface: torch.Tensor) -> torch.Tensor:
    """Zero-mean unit-variance normalisation per channel."""
    mean = surface.mean(dim=(-2, -1), keepdim=True)
    std = surface.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (surface - mean) / std
