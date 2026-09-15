"""Continuous motion codebooks via FPE binding."""

from __future__ import annotations

import torch

from vsa_motionseg.vsa.fpe import bind, fpe, make_base_phases


class MotionCodebook:
    def __init__(
        self,
        d: int,
        flow_range: float = 10.0,
        n_bins: int = 32,
        seed: int = 0,
    ) -> None:
        self.d = d
        self.flow_range = flow_range
        self.n_bins = n_bins
        self.bin_width = 2 * flow_range / n_bins
        self.phases_mx = make_base_phases(d, seed=seed + 20)
        self.phases_my = make_base_phases(d, seed=seed + 21)

    def quantize(self, v: torch.Tensor) -> torch.Tensor:
        v = v.clamp(-self.flow_range, self.flow_range)
        return torch.round((v + self.flow_range) / self.bin_width) * self.bin_width - self.flow_range

    def encode_flow(self, vx: torch.Tensor, vy: torch.Tensor) -> torch.Tensor:
        """Return (..., d) complex motion hypervectors."""
        qx = self.quantize(vx)
        qy = self.quantize(vy)
        mx = fpe(self.phases_mx, qx)
        my = fpe(self.phases_my, qy)
        return bind(mx, my)

    def motion_hypervector(
        self,
        event_hv: torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor,
    ) -> torch.Tensor:
        """Q = F bound Mx(vx) bound My(vy)."""
        mx = fpe(self.phases_mx, self.quantize(vx))
        my = fpe(self.phases_my, self.quantize(vy))
        return bind(bind(event_hv, mx), my)

    def to(self, device: torch.device) -> "MotionCodebook":
        self.phases_mx = self.phases_mx.to(device)
        self.phases_my = self.phases_my.to(device)
        return self
