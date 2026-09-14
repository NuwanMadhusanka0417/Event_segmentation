"""Residual flow after ego compensation."""

from __future__ import annotations

import torch


def residual_flow(optical_flow: torch.Tensor, rigid_bg: torch.Tensor) -> torch.Tensor:
    return optical_flow - rigid_bg
