"""Rank-r analytic VSA encoder (zero-parameter, frozen)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdems.vsa.fpe import fpe, make_base_phases
from hdems.vsa.kernel import eigen_basis, separable_approx


class VSAEncoder(nn.Module):
    """Analytic VFA encoder: time surface -> complex descriptor field F.

    Parameters
    ----------
    d : hypervector dimension
    patch_size, sigma_k, rank : VFA kernel params
    separable_terms : rank-1 factors per filter

    Output shape: (B, d, H, W) complex64
    """

    def __init__(
        self,
        d: int = 1024,
        patch_size: int = 21,
        sigma_k: float = 1.5,
        rank: int = 64,
        separable_terms: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.d = d
        self.patch_size = patch_size
        self.rank = rank

        filters, energy = eigen_basis(patch_size, sigma_k, rank)
        self.register_buffer("filters", filters)
        self.energy_fraction = energy

        self.separable = separable_approx(filters, separable_terms)
        pad = patch_size // 2
        self.pad = pad

        phases = make_base_phases(d, seed=seed)
        self.register_buffer("phases_x", phases)
        self.register_buffer("phases_y", make_base_phases(d, seed=seed + 1))

        for p in self.parameters():
            p.requires_grad = False

    def _conv_separable(self, x: torch.Tensor, terms: list) -> torch.Tensor:
        """Apply separable rank-1 filter to x: (B, 1, H, W)."""
        out = torch.zeros_like(x)
        for col, row in terms:
            tmp = F.conv2d(x, row.view(1, 1, 1, -1), padding=(0, self.pad))
            out = out + F.conv2d(tmp, col.view(1, 1, -1, 1), padding=(self.pad, 0))
        return out

    def forward(self, surface: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        surface : (B, C, H, W) float32 time surface

        Returns
        -------
        F : (B, d, H, W) complex64
        """
        B, C, H, W = surface.shape
        assert surface.dtype == torch.float32

        # Project through rank-r spatial filters, then FPE encode
        channels = []
        for r in range(self.rank):
            feat = torch.zeros(B, 1, H, W, device=surface.device)
            for c in range(C):
                feat = feat + self._conv_separable(
                    surface[:, c : c + 1], self.separable[r]
                )
            code = fpe(self.phases_x, feat.squeeze(1)) * fpe(
                self.phases_y, feat.squeeze(1)
            )
            channels.append(code)

        F_field = torch.stack(channels, dim=1)  # (B, rank, d) — need (B, d, H, W)
        # Sum rank channels into d-dim field via bundling across rank
        F_out = torch.zeros(B, self.d, H, W, dtype=torch.complex64, device=surface.device)
        for r in range(self.rank):
            F_out = F_out + channels[r].permute(0, 2, 1).reshape(B, self.d, H, W)
        return F_out
