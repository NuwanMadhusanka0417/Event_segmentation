"""Segmentation head and motion embedding (~0.5M params)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    """Motion embedding -> per-pixel class logits."""

    def __init__(
        self,
        d: int = 1024,
        embedding_dim: int = 32,
        num_classes: int = 16,
    ) -> None:
        super().__init__()
        in_ch = d * 2
        self.embed = nn.Sequential(
            nn.Conv2d(in_ch, embedding_dim * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim * 4, embedding_dim, 3, padding=1),
        )
        self.classifier = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        phi : (B, d, H, W) complex

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        x = torch.cat([phi.real, phi.imag], dim=1)
        return self.classifier(self.embed(x))
