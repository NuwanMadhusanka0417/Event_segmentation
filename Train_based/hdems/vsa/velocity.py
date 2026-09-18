"""Encode residual velocity as hypervectors and combine with the event field.

Two selectable choices:
  axis_combine  : how to fuse the vx and vy velocity hypervectors  -> bind | bundle
  event_combine : how to fuse the velocity code with the event Phi  -> bind | bundle | concat

FPE with Gaussian bases gives topological similarity (similar velocities -> similar
codes). Note FPE(0) = all-ones = the binding identity, so a zero-residual (static
background) pixel leaves Phi unchanged under bind, and only movers get a motion stamp.
"""

from __future__ import annotations

import torch

from hdems.vsa.fpe import fpe


def combine_axes(vx_code: torch.Tensor, vy_code: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "bind":
        return vx_code * vy_code           # joint 2-D velocity code (like position Xˣ⊙Yʸ)
    if mode == "bundle":
        return vx_code + vy_code           # superpose the two axis codes
    raise ValueError(f"axis_combine must be bind|bundle, got {mode!r}")


def encode_velocity(
    rvx: torch.Tensor,
    rvy: torch.Tensor,
    phi_vx: torch.Tensor,
    phi_vy: torch.Tensor,
    *,
    vel_bw: float = 6.0,
    axis_combine: str = "bind",
) -> torch.Tensor:
    """(rvx, rvy): (B, H, W) residual velocity -> Mv: (B, d, H, W) complex."""
    speed = torch.sqrt(rvx ** 2 + rvy ** 2)
    vscale = torch.quantile(speed.reshape(-1), 0.95) + 1e-6        # robust scale
    vhx = (rvx / vscale) * vel_bw
    vhy = (rvy / vscale) * vel_bw
    vx_code = fpe(phi_vx, vhx)                                     # (B, H, W, d)
    vy_code = fpe(phi_vy, vhy)
    mv = combine_axes(vx_code, vy_code, axis_combine)             # (B, H, W, d)
    return mv.permute(0, 3, 1, 2)                                 # (B, d, H, W)


def combine_event_velocity(phi: torch.Tensor, mv: torch.Tensor, mode: str) -> torch.Tensor:
    """Fuse event field Phi (B,d,H,W) with velocity code Mv (B,d,H,W) -> real features.

    bind/bundle -> one complex M, returned as [M.real | M.imag]  (B, 2d, H, W)
    concat      -> [Phi.real | Phi.imag | Mv.real | Mv.imag]      (B, 4d, H, W)
    """
    if mode == "bind":
        m = phi * mv
        return torch.cat([m.real, m.imag], dim=1).float()
    if mode == "bundle":
        m = phi + mv
        return torch.cat([m.real, m.imag], dim=1).float()
    if mode == "concat":
        return torch.cat([phi.real, phi.imag, mv.real, mv.imag], dim=1).float()
    raise ValueError(f"event_combine must be bind|bundle|concat, got {mode!r}")
