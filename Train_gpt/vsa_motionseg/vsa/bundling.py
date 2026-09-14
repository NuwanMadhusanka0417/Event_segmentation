"""Bundled matching field (from Train_based/hdems/vsa/field.py)."""

from __future__ import annotations

import torch

from vsa_motionseg.vsa.fpe import fpe


def bundled_field(
    F: torch.Tensor,
    phx: torch.Tensor,
    phy: torch.Tensor,
    M: int = 7,
) -> torch.Tensor:
    assert F.is_complex()
    m = M // 2
    offs = torch.arange(-m, m + 1, device=F.device, dtype=torch.float32)
    wx = fpe(phx, offs)
    tmp = torch.zeros_like(F)
    for i, a in enumerate(range(-m, m + 1)):
        tmp = tmp + torch.roll(F, shifts=-a, dims=2) * wx[i].view(1, -1, 1, 1)
    wy = fpe(phy, offs)
    out = torch.zeros_like(F)
    for i, b in enumerate(range(-m, m + 1)):
        out = out + torch.roll(tmp, shifts=-b, dims=3) * wy[i].view(1, -1, 1, 1)
    return out


def query(
    F: torch.Tensor,
    Phi: torch.Tensor,
    phx: torch.Tensor,
    phy: torch.Tensor,
    v: tuple[float, float] | torch.Tensor,
) -> torch.Tensor:
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, device=F.device, dtype=torch.float32)
    p = fpe(phx, v[0]) * fpe(phy, v[1])
    return ((F * p.view(1, -1, 1, 1)) * Phi.conj()).sum(1).real


def cost_volume_explicit(
    F: torch.Tensor,
    M: int = 7,
) -> torch.Tensor:
    """C[b, i, j, y, x] = cosine-like inner product <F(y,x), F(y+b, x+a)>."""
    m = M // 2
    B, d, H, W = F.shape
    C = torch.zeros(B, M, M, H, W, device=F.device)
    Fn = F / (F.abs().pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
    for ia, a in enumerate(range(-m, m + 1)):
        for ib, b in enumerate(range(-m, m + 1)):
            F_shift = torch.roll(Fn, shifts=(-a, -b), dims=(2, 3))
            C[:, ia, ib] = (Fn.conj() * F_shift).sum(1).real
    return C
