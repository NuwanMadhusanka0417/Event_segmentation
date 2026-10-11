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


def _unit_realimag(Z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Complex (B, d, H, W) -> per-pixel unit-norm [real | imag] (B, 2d, H, W).

    Re<a, b> for complex vectors equals the real dot product of [Re|Im] stacks, so
    the cosine becomes a plain real channel sum on pre-normalised fields.
    """
    n = Z.abs().pow(2).sum(1, keepdim=True).sqrt() + eps
    return torch.cat([Z.real / n, Z.imag / n], dim=1)


def cross_cost_volume(A: torch.Tensor, B: torch.Tensor, M: int = 7) -> torch.Tensor:
    """Two-time cost volume C[b, ia, ib, y, x] = cos(A(y,x), B(y+a, x+b))  (Eq. 10).

    A, B : (B, d, H, W) complex   reference and target descriptor fields
    out  : (B, M, M, H, W) float32

    Both fields are normalised ONCE and B is zero-padded ONCE; every displacement
    is then a slice VIEW of the padded field. The previous version recomputed both
    norms and made a full ``torch.roll`` copy of B for each of the M^2 shifts,
    which made the paper's M = 31 (961 shifts) impractical. Zero padding also
    fixes a border artefact: roll wrapped around, so pixels on the left edge were
    matched against the right edge. Out-of-frame displacements now score 0.
    """
    m = M // 2
    Bn, _, H, W = A.shape
    a_n = _unit_realimag(A)
    b_p = F.pad(_unit_realimag(B), (m, m, m, m))
    C = torch.empty(Bn, M, M, H, W, device=A.device, dtype=a_n.dtype)
    for ia in range(M):
        for ib in range(M):
            C[:, ia, ib] = (a_n * b_p[:, :, ia:ia + H, ib:ib + W]).sum(1)
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


def box_filter(x: torch.Tensor, k: int) -> torch.Tensor:
    """k x k mean, stride 1, zero padding -- same result as
    ``F.avg_pool2d(x, k, 1, k // 2)`` but O(1) per pixel via cumulative sums.

    The paper pools the cost volume with sc = 71. A direct 71x71 average pool over
    all M^2 = 961 displacement channels costs 71^2 operations per value; running
    sums make it two subtractions per value regardless of k.
    """
    if k <= 1:
        return x
    p = k // 2
    x = F.pad(x, (p, p, p, p))
    c = F.pad(x.cumsum(-1), (1, 0))
    x = c[..., k:] - c[..., :-k]
    c = F.pad(x.cumsum(-2), (0, 0, 1, 0))
    x = c[..., k:, :] - c[..., :-k, :]
    return x / float(k * k)


def flow_from_cost(
    C: torch.Tensor,
    M: int = 7,
    alpha: float = 0.3,
    vel_scale: float = 1.0,
    smooth: int = 1,
    *,
    return_confidence: bool = False,
):
    """Probability-volume optical-flow estimator (Eq. 12-14), parameter-free.

    C : (B, M, M, H, W)   final cost volume
    -> flow (B, 2, H, W) as [u_x, u_y], expected displacement over the MxM grid.

    ``return_confidence``: also return (B, 1, H, W) = sum of P^2, how concentrated the
    Eq.12 probability volume is: 1 = one displacement, 1/K = K equally likely ones
    (e.g. along an edge), 0 = a flat cost volume. Measured against GT flow on 12
    EVIMO2 frames (2026-09-30), it is only a WEAK error predictor (Spearman +0.27
    with -EPE; the best of four cost-volume measures -- max-mean was +0.08, and it
    was higher on empty pixels than on event pixels, because the cosine normalises
    away the descriptor strength). Where the events are is given separately.
    """
    Bn, _, _, H, W = C.shape
    Cf = C.reshape(Bn, M * M, H, W)
    if smooth > 1:                    # Eq.12 spatial average pooling, kernel sc, stride 1
        Cf = box_filter(Cf, smooth)
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
    flow = torch.stack([ux, uy], dim=1)
    if return_confidence:
        return flow, prob.pow(2).sum(1, keepdim=True)
    return flow
