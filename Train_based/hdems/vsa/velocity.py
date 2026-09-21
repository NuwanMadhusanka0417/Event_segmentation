"""Encode residual velocity as hypervectors and combine with the event field.

Selectable choices:
  event_feature : which event hypervector is fused with velocity   -> phi | f
                  (phi = 7x7 bundled neighbourhood field, f = the VFA descriptor F0 itself;
                   selected in HDEMS, this module just receives it as X)
  axis_combine  : how to fuse the vx and vy velocity hypervectors  -> bind | bundle
  event_combine : how to fuse the velocity code with the event X   -> bind | bundle | bindbundle | concat
                  bindbundle = (X^ o Mv) + Mv, with X^ = X scaled to unit frame RMS

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


EVENT_COMBINES = ("bind", "bundle", "bindbundle", "concat")


def unit_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale each sample so its RMS over (d, H, W) is 1.

    Per FRAME, not per pixel: event-dense pixels stay stronger than empty ones.
    Needed before bundling with Mv, whose FPE components all have |.| = 1 --
    raw F is ~5-11x and Phi ~100-350x larger on event pixels, so an unscaled
    bundle would drown the velocity term.
    """
    rms = x.abs().pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()
    return x / (rms + eps)


def combine_event_velocity(x: torch.Tensor, mv: torch.Tensor, mode: str) -> torch.Tensor:
    """Fuse event hypervector X (B,d,H,W) -- Phi or F -- with velocity code Mv -> real features.

    bind       -> X o Mv                          [real | imag]  (B, 2d, H, W)
    bundle     -> X + Mv                          [real | imag]  (B, 2d, H, W)
    bindbundle -> (X^ o Mv) + Mv,  X^ = unit_rms(X)  [real | imag]  (B, 2d, H, W)
                  == (X^ + 1) o Mv, since FPE(0)-style all-ones is the bind identity:
                  the event pattern tagged with its velocity, plus the velocity itself
    concat     -> [X.real | X.imag | Mv.real | Mv.imag]            (B, 4d, H, W)
    """
    if mode == "bind":
        m = x * mv
    elif mode == "bundle":
        m = x + mv
    elif mode == "bindbundle":
        m = unit_rms(x) * mv + mv
    elif mode == "concat":
        return torch.cat([x.real, x.imag, mv.real, mv.imag], dim=1).float()
    else:
        raise ValueError(f"event_combine must be one of {EVENT_COMBINES}, got {mode!r}")
    return torch.cat([m.real, m.imag], dim=1).float()
