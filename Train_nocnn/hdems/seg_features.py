"""Segmentation head feature preparation (shared CNN + Ridge).

Head I/O contract (``segmentation.SegmentationHead`` / ``RidgeHead`` / ``HDEMS``):

- **Input ``phi``:** ``(B, d, H, W)`` complex64 — bundled field ``Phi`` from
  ``HierarchicalMatcher`` on the encoder output.
- **Input ``surface`` (optional):** ``(B, 2, H, W)`` float32 time surface.
- **Prepared features:** ``(B, D, H, W)`` float32, ``D = 2*d`` (+2 if motion append).
- **Output logits:** ``(B, num_classes, H, W)``; ``argmax(dim=1)`` → class id.
- **Labels ``mask``:** ``(B, H, W)`` int64; ``0`` = background, ``1..K`` = instances.
- **Eval validity:** ``mask >= 0``; fit/event mask: time-surface activity > threshold.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hdems.data.motion_labels import IGNORE_LABEL

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)


def event_valid_mask(surface: torch.Tensor, *, threshold: float = 1e-6) -> torch.Tensor:
    if surface.dim() == 3:
        surface = surface.unsqueeze(0)
    return surface.abs().sum(dim=1) > threshold


def event_pixel_mask(surface: torch.Tensor, *, threshold: float = 1e-6) -> torch.Tensor:
    """Pixels that actually carry events in the window -> (B, H, W) bool.

    Accepts (2,H,W), (B,2,H,W) or the multi-time stack (B,T,2,H,W): every
    non-spatial dimension is reduced, so a pixel is active if ANY frame/polarity
    has activity there. Loss and metrics are restricted to these pixels — the
    rest have no evidence and would otherwise dominate training with trivial
    background.
    """
    if surface.dim() == 3:
        surface = surface.unsqueeze(0)
    b, h, w = surface.shape[0], surface.shape[-2], surface.shape[-1]
    return surface.abs().reshape(b, -1, h, w).sum(dim=1) > threshold


def score_pixel_mask(batch: dict, surface: torch.Tensor) -> torch.Tensor:
    """Pixels that are trained on and scored -> (B, H, W) bool.

    The dataset provides ``score_mask`` = pixels with an event in the last
    dataset.score_window_ms before the label time. The label is the object mask AT
    that time, so older events -- the trail an object leaves behind it -- would be
    labelled "static" although a moving object caused them (the review measured 39%
    of the trail pixels labelled static). Falls back to every event pixel of the
    stack when the dataset gives no mask (score_window_ms 0, or cached shards).
    """
    m = batch.get("score_mask") if isinstance(batch, dict) else None
    if m is not None:
        m = m.to(surface.device).bool()
        return m.unsqueeze(0) if m.dim() == 2 else m       # a single sample (H, W)
    return event_pixel_mask(surface)


def motion_mag_angle(surface: torch.Tensor) -> torch.Tensor:
    if surface.dim() == 3:
        surface = surface.unsqueeze(0)
    device, dtype = surface.device, surface.dtype
    sx = _SOBEL_X.to(device=device, dtype=dtype)
    sy = _SOBEL_Y.to(device=device, dtype=dtype)
    s = surface.mean(dim=1, keepdim=True)
    gx = F.conv2d(s, sx, padding=1)
    gy = F.conv2d(s, sy, padding=1)
    return torch.cat([torch.hypot(gx, gy), torch.atan2(gy, gx)], dim=1)


def phi_to_realimag(phi: torch.Tensor) -> torch.Tensor:
    return torch.cat([phi.real, phi.imag], dim=1).float()


def prepare_seg_features(
    phi: torch.Tensor,
    surface: torch.Tensor | None = None,
    *,
    feature_mean: torch.Tensor | None = None,
    mean_center: bool = True,
    motion_features: bool = True,
) -> torch.Tensor:
    x = phi_to_realimag(phi)
    if motion_features:
        if surface is None:
            raise ValueError("surface required when motion_features=True")
        if surface.dim() == 3:
            surface = surface.unsqueeze(0)
        x = torch.cat([x, motion_mag_angle(surface)], dim=1)
    if mean_center and feature_mean is not None and feature_mean.numel():
        fm = feature_mean.to(device=x.device, dtype=x.dtype)
        x = x - fm.view(1, -1, 1, 1)
    return x


def flatten_valid_features(
    features: torch.Tensor,
    mask: torch.Tensor,
    surface: torch.Tensor,
    *,
    event_threshold: float = 1e-6,
    score_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``score_mask`` (B, H, W): pixels to use (see ``score_pixel_mask``); default =
    every event pixel of ``surface``."""
    if surface.dim() == 3:
        surface = surface.unsqueeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    b, d, h, w = features.shape
    events = (score_mask.to(mask.device).bool() if score_mask is not None
              else event_pixel_mask(surface, threshold=event_threshold))
    # Drop ignore-label pixels (ambiguous object speed, mask boundary band) as well
    # as pixels without events -- a 255 label would otherwise be fitted as a class.
    valid = events & (mask >= 0) & (mask != IGNORE_LABEL)
    x_flat = features.permute(0, 2, 3, 1).reshape(b * h * w, d)
    y_flat = mask.reshape(b * h * w)
    v = valid.reshape(b * h * w)
    return x_flat[v], y_flat[v]
