"""Temporal consistency: carry the moving-probability map from frame to frame.

Why
---
Every frame was segmented on its own, so a mover that is clear in one frame can
vanish in the next when its evidence is weak there (few events, a slow phase of its
motion), and isolated false positives flicker on the background. Objects do not
appear and disappear between two frames 16.7 ms apart. Hua, Yuan & Fermueller 2025
(our benchmark) rely on exactly this: a persistent background map and the previous
segmentation warped into the current frame.

How (causal: only past frames)
------------------------------
The front end's flow at frame t is the displacement of each pixel towards EARLIER
times (reference surface at the label time, targets before it; one flow interval =
12.5 ms). So the previous frame's filtered map can be pulled into the current frame
by BACKWARD warping -- no splatting, no holes:

    P~_t(x) = P_t(x) + beta * valid(x) * ( P~_{t-1}(x + flow_t(x) * dt / interval) - P_t(x) )

dt = time since the previous frame of the same sequence. The state is reset when the
gap exceeds ``max_gap_s`` or the sequence changes; ``valid`` is 0 where the warped
position falls outside the previous map. beta = 0 switches it off.

Measured flow is the TOTAL motion (camera + object), which is what moves a pixel
between frames, so the same warp follows the background and the movers.

Occlusion check: where an object WAS, the background is revealed and its current
flow is background motion, so a plain warp would copy the old "moving" probability
onto it (a ghost trail behind every mover). The previous frame's flow is warped too,
and the previous prediction is only trusted where the two motions agree (constant
velocity over 16.7 ms):

    valid(x) *= exp( -|flow_t(x) - flow_{t-1}(x + d)|^2 / (2 sigma^2) )
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def warp_backward(img: torch.Tensor, disp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``img`` (H, W) or (C, H, W) at x + disp(x) -> (warped, valid (H, W) in [0, 1]).

    ``disp`` (2, H, W) in pixels, [dx, dy]. Bilinear; outside the image -> invalid.
    """
    chans = img if img.dim() == 3 else img[None]
    H, W = chans.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device, dtype=torch.float32),
                            torch.arange(W, device=img.device, dtype=torch.float32), indexing="ij")
    gx = (xx + disp[0].float()) / max(W - 1, 1) * 2.0 - 1.0
    gy = (yy + disp[1].float()) / max(H - 1, 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None]
    both = torch.cat([chans.float(), torch.ones(1, H, W, device=img.device)])[None]
    out = F.grid_sample(both, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    return (out[0] if img.dim() == 2 else out[:-1]), out[-1]


class TemporalFilter:
    """Causal propagation of P(moving) along the measured flow, one state per sequence.

    ``sigma`` (working px per interval): scale of the occlusion check between the
    current flow and the previous frame's flow at the source point; 0 = no check.
    """

    def __init__(self, beta: float, interval_s: float = 0.0125, max_gap_s: float = 0.05,
                 sigma: float = 1.0) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"temporal beta must be in [0, 1), got {beta}")
        self.beta = float(beta)
        self.interval_s = float(interval_s)
        self.max_gap_s = float(max_gap_s)
        self.sigma = float(sigma)
        self.state: dict[Any, tuple[torch.Tensor, torch.Tensor | None, float]] = {}

    def reset(self) -> None:
        self.state.clear()

    def __call__(self, key: Any, prob: torch.Tensor, flow: torch.Tensor | None,
                 ts: float) -> torch.Tensor:
        """prob (H, W) at frame time ``ts`` (s); flow (2, H, W) working px per interval
        towards earlier times (None = no motion known). -> filtered prob (H, W)."""
        out = prob
        flow = None if flow is None else flow.to(prob.device).float()
        prev = self.state.get(key)
        if self.beta > 0 and prev is not None:
            p_prev, f_prev, t_prev = prev
            dt = ts - t_prev
            if 0.0 < dt <= self.max_gap_s and p_prev.shape == prob.shape:
                disp = (torch.zeros(2, *prob.shape, device=prob.device) if flow is None
                        else flow * (dt / self.interval_s))
                if self.sigma > 0 and flow is not None and f_prev is not None:
                    warped, valid = warp_backward(torch.cat([p_prev[None], f_prev]), disp)
                    mismatch = (flow - warped[1:]).pow(2).sum(0)
                    valid = valid * torch.exp(-mismatch / (2.0 * self.sigma ** 2))
                    warped = warped[0]
                else:
                    warped, valid = warp_backward(p_prev, disp)
                out = prob + self.beta * valid * (warped - prob)
        self.state[key] = (out.detach(), None if flow is None else flow.detach(), float(ts))
        return out
