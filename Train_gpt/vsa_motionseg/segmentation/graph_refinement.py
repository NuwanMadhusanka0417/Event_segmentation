"""Local VSA spatial refinement."""

from __future__ import annotations

import torch

from vsa_motionseg.vsa.fpe import similarity


def local_vsa_refinement(
    Q: torch.Tensor,
    residual_flow: torch.Tensor,
    confidence: torch.Tensor,
    *,
    lambda_bundle: float = 0.2,
    sigma_flow: float = 2.0,
    sigma_hv: float = 0.3,
    connectivity: int = 8,
) -> torch.Tensor:
    """
    Q: (N, d) complex motion hypervectors for active pixels only — simplified full-grid version.
    For grid: pass (d,H,W) by reshaping internally via caller.
    """
    if Q.dim() == 3:
        d, H, W = Q.shape
        Qf = Q.reshape(d, -1).T
        rf = residual_flow.reshape(2, -1).T
        conf = confidence.reshape(-1)
        out = _refine_grid(Qf, rf, conf, H, W, lambda_bundle, sigma_flow, sigma_hv, connectivity)
        return out.reshape(d, H, W)
    return Q


def _refine_grid(Q, rf, conf, H, W, lam, sf, sh, conn):
    d = Q.shape[1]
    Qn = Q / (Q.abs().norm(dim=1, keepdim=True) + 1e-8)
    out = Q.clone()
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn == 8:
        offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for idx in range(H * W):
        y, x = idx // W, idx % W
        acc = Q[idx]
        wsum = 1.0
        for dy, dx in offs:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            j = ny * W + nx
            df = ((rf[idx] - rf[j]) ** 2).sum()
            af = torch.exp(-df / (sf ** 2))
            hv_sim = similarity(Qn[idx : idx + 1], Qn[j : j + 1], dim=-1).squeeze()
            ah = torch.exp(-(1 - hv_sim) / sh)
            w = af * ah * conf[idx] * conf[j]
            acc = acc + lam * w * Q[j]
            wsum += lam * w
        out[idx] = acc / wsum
    return out
