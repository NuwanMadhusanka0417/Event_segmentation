"""Fractional Power Encoding, binding, bundling, and similarity (from Train_based/hdems/vsa/fpe.py)."""

from __future__ import annotations

import torch


def make_base_phases(
    d: int,
    seed: int = 0,
    kernel: str = "gaussian",
    bandwidth: float = 1.0,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if kernel == "gaussian":
        return torch.randn(d, generator=g) * bandwidth
    if kernel == "uniform":
        return (torch.rand(d, generator=g) * 2 - 1) * torch.pi * bandwidth
    raise ValueError(kernel)


def fpe(phases: torch.Tensor, x: torch.Tensor | float) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=phases.dtype, device=phases.device)
    else:
        x = x.to(device=phases.device, dtype=phases.dtype)
    return torch.exp(1j * x[..., None] * phases)


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def unbind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b.conj()


def bundle(vs: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return vs.sum(dim=dim)


def similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    num = (a * b.conj()).sum(dim).real
    denom = (
        a.abs().pow(2).sum(dim).sqrt() * b.abs().pow(2).sum(dim).sqrt() + 1e-8
    )
    return num / denom


def hamming_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Similarity for bipolar {-1,+1} hypervectors."""
    return (a * b).sum(dim) / a.shape[dim]
