"""Hierarchical bundled matching field (zero-parameter)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hdems.vsa.field import bundled_field, query
from hdems.vsa.fpe import make_base_phases


class HierarchicalMatcher(nn.Module):
    """Coarse-to-fine bundled matching across pyramid levels.

    Each level uses M=7 separable field; range compounds geometrically.
    """

    def __init__(
        self,
        d: int = 1024,
        M: int = 7,
        pyramid_levels: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.d = d
        self.M = M
        self.pyramid_levels = pyramid_levels
        self.register_buffer("phx", make_base_phases(d, seed=seed))
        self.register_buffer("phy", make_base_phases(d, seed=seed + 1))

    def forward(
        self,
        pyramid: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        pyramid : list of (B, d, H_l, W_l) complex fields, finest first

        Returns
        -------
        phi_pyramid : list of (B, d, H_l, W_l) bundled fields
        """
        return [bundled_field(F, self.phx, self.phy, M=self.M) for F in pyramid]

    def query_displacement(
        self,
        F: torch.Tensor,
        Phi: torch.Tensor,
        vx: float,
        vy: float,
    ) -> torch.Tensor:
        """Query cost at displacement (vx, vy). Returns (B, H, W)."""
        return query(F, Phi, self.phx, self.phy, (vx, vy))
