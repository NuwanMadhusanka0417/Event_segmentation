"""Paper-faithful VSA-Flow cost volume and flow estimator (You et al. 2025).

This replaces the single-frame self-similarity path with the paper's actual
construction:

  - **Two-time cost volume** (Eq. 10): match a reference HD descriptor F0 against
    a LATER descriptor Fk -> cos(F0(x), Fk(x+v)) over an MxM window. This is real
    temporal correspondence (optical flow), not intra-frame self-similarity.
  - **Multi-scale doubling-interval fusion** (Eq. 11): pair F0 with F1, F2, F4 at
    scales 0, 1, 2 (avg-pooled), up-sample each cost volume to full resolution,
    and sum -> large-motion coverage within a small MxM window.
  - **Probability-volume flow estimator** (Eq. 12-14): threshold the cost volume
    by alpha*max + (1-alpha)*mean, normalise to a probability P over the MxM
    displacement grid, and take the expected displacement over a flow template.

All functions are parameter-free.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def avg_pool_complex(field: torch.Tensor, k: int) -> torch.Tensor:
    """Average-pool a complex (B, d, H, W) field by factor k (real/imag apart)."""
    if k <= 1:
        return field
    r = F.avg_pool2d(field.real, k, k)
    i = F.avg_pool2d(field.imag, k, k)
    return torch.complex(r, i)


def _cosine_shift(A: torch.Tensor, B: torch.Tensor, a: int, b: int, eps: float = 1e-8) -> torch.Tensor:
    """Per-pixel complex cosine similarity  cos(A(x), B(x + (a, b)))  -> (B, H, W)."""
    Bs = torch.roll(B, shifts=(-a, -b), dims=(2, 3))
    num = (A * Bs.conj()).sum(1).real
    den = A.abs().pow(2).sum(1).sqrt() * Bs.abs().pow(2).sum(1).sqrt() + eps
    return num / den


def cross_cost_volume(A: torch.Tensor, B: torch.Tensor, M: int = 7) -> torch.Tensor:
    """Two-time cost volume C[b, ia, ib, y, x] = cos(A(y,x), B(y+a, x+b)).

    A, B : (B, d, H, W) complex   reference and target descriptor fields
    out  : (B, M, M, H, W) float32
    """
    m = M // 2
    Bn, _, H, W = A.shape
    C = torch.zeros(Bn, M, M, H, W, device=A.device)
    for ia, a in enumerate(range(-m, m + 1)):
        for ib, b in enumerate(range(-m, m + 1)):
            C[:, ia, ib] = _cosine_shift(A, B, a, b)
    return C


def multiscale_cost_volume(
    fields: list[torch.Tensor],
    M: int = 7,
    scales: tuple[int, ...] = (0, 1, 2),
) -> torch.Tensor:
    """Sum of two-time cost volumes across doubling-interval scales (Eq. 10-11).

    fields = [F0, F1, F2, F4] (reference first, then progressively later frames).
    Scale s pairs F0 with fields[s+1] at pooling factor 2**s, then the cost
    volume is bilinearly up-sampled to full resolution and summed.
    """
    F0 = fields[0]
    Bn, _, H, W = F0.shape
    total: torch.Tensor | None = None
    for s in scales:
        k = 2 ** s
        target = fields[min(s + 1, len(fields) - 1)]
        A = avg_pool_complex(F0, k)
        B = avg_pool_complex(target, k)
        C = cross_cost_volume(A, B, M)                       # (Bn, M, M, Hs, Ws)
        if k > 1:
            C = F.interpolate(
                C.reshape(Bn, M * M, C.shape[-2], C.shape[-1]),
                size=(H, W), mode="bilinear", align_corners=False,
            ).reshape(Bn, M, M, H, W)
        total = C if total is None else total + C
    return total


def flow_from_cost(
    C: torch.Tensor,
    M: int = 7,
    alpha: float = 0.3,
    vel_scale: float = 1.0,
    smooth: int = 1,
) -> torch.Tensor:
    """Probability-volume optical-flow estimator (Eq. 12-14), parameter-free.

    C : (B, M, M, H, W)   final cost volume
    -> flow (B, 2, H, W) as [u_x, u_y], expected displacement over the MxM grid.
    """
    Bn, _, _, H, W = C.shape
    Cf = C.reshape(Bn, M * M, H, W)
    if smooth > 1:                                            # optional spatial denoise
        Cf = F.avg_pool2d(Cf, smooth, 1, smooth // 2)
    cmax = Cf.max(1, keepdim=True).values
    cmean = Cf.mean(1, keepdim=True)
    cbar = (Cf - alpha * cmax - (1.0 - alpha) * cmean).clamp_min(0)   # Eq. 12
    prob = cbar / (cbar.sum(1, keepdim=True) + 1e-8)

    m = M // 2
    offs = torch.arange(-m, m + 1, device=C.device, dtype=torch.float32) * vel_scale
    gy = offs.view(M, 1).expand(M, M).reshape(M * M)          # first cost-volume axis (y)
    gx = offs.view(1, M).expand(M, M).reshape(M * M)          # second axis (x)
    uy = (prob * gy.view(1, -1, 1, 1)).sum(1)
    ux = (prob * gx.view(1, -1, 1, 1)).sum(1)
    return torch.stack([ux, uy], dim=1)
