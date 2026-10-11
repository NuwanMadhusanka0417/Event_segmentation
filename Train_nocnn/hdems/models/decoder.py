"""Coarse-to-fine trained decoder (~1M params)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowDecoder(nn.Module):
    """Slim GRU + conv decoder for coarse-to-fine flow refinement.

    Input: concatenated real/imag parts of bundled field Phi at each level.
    Output: optical flow (B, 2, H, W).
    """

    def __init__(
        self,
        d: int = 1024,
        hidden_channels: int = 64,
        gru_layers: int = 1,
    ) -> None:
        super().__init__()
        in_ch = d * 2  # real + imag
        self.proj = nn.Conv2d(in_ch, hidden_channels, 3, padding=1)
        self.gru = nn.GRU(
            hidden_channels,
            hidden_channels,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )

    def forward(
        self,
        phi: torch.Tensor,
        flow_coarse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        phi : (B, d, H, W) complex bundled field
        flow_coarse : optional (B, 2, H, W) upsampled coarse flow

        Returns
        -------
        flow : (B, 2, H, W)
        """
        B, d, H, W = phi.shape
        x = torch.cat([phi.real, phi.imag], dim=1)
        x = F.relu(self.proj(x))

        if flow_coarse is not None:
            flow_up = F.interpolate(flow_coarse, size=(H, W), mode="bilinear", align_corners=True)
            x = x + self.proj(torch.cat([flow_up, torch.zeros(B, self.proj.in_channels - 2, H, W, device=x.device)], dim=1))[:, :x.shape[1]]

        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, -1).contiguous()
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.gru(x_seq)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return self.head(out)
