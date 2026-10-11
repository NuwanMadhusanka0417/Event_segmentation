"""Fractional Power Encoding, binding, bundling, and similarity."""

from __future__ import annotations

import torch


def make_base_phases(
    d: int,
    seed: int = 0,
    kernel: str = "gaussian",
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """Random phases defining an FPE base vector.

    The DISTRIBUTION of phases determines the induced kernel (Frady et al. 2021):
      uniform  -> sinc kernel
      gaussian -> Gaussian kernel   <-- use this; matches the VFA smoothing in the paper

    Returns
    -------
    phases : (d,) float32
    """
    g = torch.Generator().manual_seed(seed)
    if kernel == "gaussian":
        return torch.randn(d, generator=g) * bandwidth
    elif kernel == "uniform":
        return (torch.rand(d, generator=g) * 2 - 1) * torch.pi * bandwidth
    raise ValueError(kernel)


def fpe(phases: torch.Tensor, x: torch.Tensor | float) -> torch.Tensor:
    """Fractional power encoding: z(x) = X^x, in the PHASOR domain.

    GOTCHA: never implement this as ifft(fft(X)**x) -- complex powers have branch
    cuts and you will get silent sign errors. Always exponentiate the PHASE.

    Parameters
    ----------
    phases : (d,)
    x      : (*,) scalar positions, or Python float/int

    Returns
    -------
    complex tensor, shape (*x.shape, d)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=phases.dtype, device=phases.device)
    else:
        x = x.to(device=phases.device, dtype=phases.dtype)
    return torch.exp(1j * x[..., None] * phases)


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Binding. In the phasor domain circular convolution IS elementwise product."""
    return a * b


def unbind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inverse binding: bind with the conjugate (valid because base vectors are unitary)."""
    return a * b.conj()


def bundle(vs: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Superposition = sum. Do NOT normalise here; normalise at readout."""
    return vs.sum(dim=dim)


def similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Real part of the normalised Hermitian inner product."""
    num = (a * b.conj()).sum(dim).real
    denom = (
        a.abs().pow(2).sum(dim).sqrt() * b.abs().pow(2).sum(dim).sqrt() + 1e-8
    )
    return num / denom
