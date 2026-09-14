"""VSA cost-volume motion matching."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vsa_motionseg.vsa.bundling import cost_volume_explicit


def fuse_multiscale_costs(costs: list[torch.Tensor], alphas: list[float] | None = None) -> torch.Tensor:
    if alphas is None:
        alphas = [1.0 / len(costs)] * len(costs)
    out = sum(a * c for a, c in zip(alphas, costs))
    return out


def cost_to_flow(
    C: torch.Tensor,
    *,
    temperature: float = 0.05,
    delta_t: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    C: (B, M, M, H, W) cosine costs.
    Returns flow (B,2,H,W), prob stats, validity.
    """
    B, M, _, H, W = C.shape
    m = M // 2
    flat = C.view(B, M * M, H, W)
    P = torch.softmax(flat / temperature, dim=1)
    top2 = torch.topk(flat, k=2, dim=1)
    margin = top2.values[:, 0] - top2.values[:, 1]
    entropy = -(P * (P + 1e-8).log()).sum(dim=1)

    offs_a = torch.arange(-m, m + 1, device=C.device, dtype=C.dtype)
    offs_b = torch.arange(-m, m + 1, device=C.device, dtype=C.dtype)
    grid_a, grid_b = torch.meshgrid(offs_a, offs_b, indexing="ij")
    disp_x = grid_a.reshape(-1)
    disp_y = grid_b.reshape(-1)

    vx = (P * disp_x.view(1, -1, 1, 1)).sum(dim=1) / delta_t
    vy = (P * disp_y.view(1, -1, 1, 1)).sum(dim=1) / delta_t
    flow = torch.stack([vx, vy], dim=1)
    return {
        "flow": flow,
        "max_cost": flat.max(dim=1).values,
        "max_prob": P.max(dim=1).values,
        "margin": margin,
        "entropy": entropy,
        "prob": P.view(B, M, M, H, W),
    }


def cost_volume_two_fields(F0: torch.Tensor, F1: torch.Tensor, M: int = 7) -> torch.Tensor:
    """C[b,i,j,y,x] = <F0(y,x), F1(y+b, x+a)> (normalized inner product)."""
    m = M // 2
    B, d, H, W = F0.shape
    F0n = F0 / (F0.abs().pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
    F1n = F1 / (F1.abs().pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
    C = torch.zeros(B, M, M, H, W, device=F0.device)
    for ia, a in enumerate(range(-m, m + 1)):
        for ib, b in enumerate(range(-m, m + 1)):
            F1_shift = torch.roll(F1n, shifts=(-a, -b), dims=(2, 3))
            C[:, ia, ib] = (F0n.conj() * F1_shift).sum(1).real
    return C


def compute_cost_volume_pair(
    F0: torch.Tensor,
    F1: torch.Tensor,
    search_radius: int,
    *,
    active_mask: torch.Tensor | None = None,
    chunked: bool = False,
) -> torch.Tensor:
    """Match F0 at t0 to F1 at t1."""
    M = 2 * search_radius + 1
    if not chunked:
        return cost_volume_two_fields(F0, F1, M=M)
    # Chunked: process rows
    B, d, H, W = F0.shape
    C = torch.zeros(B, M, M, H, W, device=F0.device)
    chunk = max(H // 4, 1)
    for y0 in range(0, H, chunk):
        y1 = min(y0 + chunk, H)
        sl = F0[:, :, y0:y1]
        sl1 = F1[:, :, y0:y1]
        C[:, :, :, y0:y1] = cost_volume_two_fields(sl, sl1, M=M)[:, :, :, : sl.shape[2]]
    if active_mask is not None:
        C = C * active_mask.unsqueeze(1).unsqueeze(1)
    return C


def multitime_cost(
    fields: list[torch.Tensor],
    search_radius: int,
    alphas: list[float] | None = None,
) -> torch.Tensor:
    """fields[0] at t0, fields[k] at progressive times."""
    C0 = cost_volume_explicit(fields[0], M=2 * search_radius + 1)
    costs = [C0]
    for F1 in fields[1:]:
        costs.append(cost_volume_explicit(F1, M=2 * search_radius + 1))
    return fuse_multiscale_costs(costs, alphas)
