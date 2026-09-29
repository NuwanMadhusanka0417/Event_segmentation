"""Encode residual velocity as hypervectors and combine with the event field.

Selectable choices:
  event_feature : which event hypervector is fused with velocity   -> phi | f
                  (phi = 7x7 bundled neighbourhood field, f = the VFA descriptor F0 itself;
                   selected in HDEMS, this module just receives it as X)
  axis_combine  : how to fuse the vx and vy velocity hypervectors  -> bind | bundle
  event_combine : how to fuse the velocity code with the event X   -> bind | bundle | bindbundle | concat
                  bindbundle = (X^ o Mv) + Mv, with X^ = X scaled to unit frame RMS
  vel_norm      : velocity units of the code                       -> fixed | frame (old)

The motion-first head (head mfcnn) does not fuse X and Mv: they enter separate
branches, plus the explicit ``motion_scalars`` channels.

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


VEL_NORMS = ("fixed", "frame")


def encode_velocity(
    rvx: torch.Tensor,
    rvy: torch.Tensor,
    phi_vx: torch.Tensor,
    phi_vy: torch.Tensor,
    *,
    vel_bw: float = 6.0,
    axis_combine: str = "bind",
    norm: str = "frame",
    unit: float = 0.5,
) -> torch.Tensor:
    """(rvx, rvy): (B, H, W) residual velocity -> Mv: (B, d, H, W) complex.

    norm = fixed : v / unit, one FPE unit = ``unit`` px per interval, the same in
                   every frame. Codes of velocities ``unit`` apart have cosine 0.61.
    norm = frame : v / (95th percentile of this frame's speed) * vel_bw (old).
                   The meaning of a code then changes from frame to frame: on a frame
                   where nothing moves, flow noise is stretched to look like the
                   fastest motion. Measured AUC(moving) 0.83 -> 0.78 with it.
    """
    if norm == "fixed":
        vhx, vhy = rvx / unit, rvy / unit
    elif norm == "frame":
        speed = torch.sqrt(rvx ** 2 + rvy ** 2)
        vscale = torch.quantile(speed.reshape(-1), 0.95) + 1e-6    # robust scale
        vhx = (rvx / vscale) * vel_bw
        vhy = (rvy / vscale) * vel_bw
    else:
        raise ValueError(f"vel_norm must be one of {VEL_NORMS}, got {norm!r}")
    vx_code = fpe(phi_vx, vhx)                                     # (B, H, W, d)
    vy_code = fpe(phi_vy, vhy)
    mv = combine_axes(vx_code, vy_code, axis_combine)             # (B, H, W, d)
    return mv.permute(0, 3, 1, 2)                                 # (B, d, H, W)


# Below this, a frame's background residual is treated as this value, so the
# speed-to-noise channel stays finite when the flow of a still scene is exactly 0.
NOISE_FLOOR_PX = 0.1


MOTION_SCALARS = 6


def motion_scalars(
    residual: torch.Tensor,
    ref_events: torch.Tensor,
    flow: torch.Tensor,
    *,
    unit: float = 0.5,
) -> torch.Tensor:
    """Explicit motion channels for the motion-first head -> (B, 6, H, W).

    residual   : (B, 2, H, W) ego-compensated velocity, SENSOR px per interval
    ref_events : (B, H, W) bool, pixels with events in the reference surface
    flow       : (B, 2, H, W) raw flow (before ego compensation), SENSOR px

    Channels (log-compressed so 0.25 px and 10 px both land in a usable range):
      slog(rx / unit), slog(ry / unit)    direction and speed, fixed units
      log1p(|r| / unit)                   speed, fixed units
      log1p(|r| / noise)                  speed relative to this frame's background:
                                          noise = median |r| over the event pixels
                                          (movers are a minority of them)
      log1p(|flow| / unit)                raw speed, no ego compensation
      log1p(camera / unit)                how fast the CAMERA moves in this frame
                                          (median |flow - r| over the event pixels),
                                          the same value at every pixel

    Why the raw flow as well (measured 2026-09-30, 48 eval frames with a mover):
    AUC(moving vs static) is 0.738 for |raw flow| but 0.700 for |residual|. With a
    still camera the ego fit can lock onto a large mover and wreck the residual
    (a frame at 0.92 raw vs 0.46 residual); with a moving camera the raw flow is
    useless (0.09 raw vs 0.87 residual). The camera channel lets the head learn
    which of the two to trust.
    """
    def slog(v: torch.Tensor) -> torch.Tensor:
        return torch.sign(v) * torch.log1p(v.abs())

    speed = residual.pow(2).sum(1).sqrt()                          # (B, H, W)
    raw = flow.pow(2).sum(1).sqrt()
    ego = (flow - residual).pow(2).sum(1).sqrt()
    noise, camera = [], []
    for b in range(residual.shape[0]):
        ev = ref_events[b]
        noise.append(speed[b][ev].median() if ev.any() else speed.new_tensor(0.0))
        camera.append(ego[b][ev].median() if ev.any() else speed.new_tensor(0.0))
    noise = torch.stack(noise).clamp_min(NOISE_FLOOR_PX).view(-1, 1, 1)
    camera = torch.stack(camera).view(-1, 1, 1).expand_as(speed)
    return torch.stack([
        slog(residual[:, 0] / unit),
        slog(residual[:, 1] / unit),
        torch.log1p(speed / unit),
        torch.log1p(speed / noise),
        torch.log1p(raw / unit),
        torch.log1p(camera / unit),
    ], dim=1).float()


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
