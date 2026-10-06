"""Real-valued Fractional Power Encoding (the HRR form of FPE) -- complex -> real.

A coordinate x is encoded with a base vector X raised to the power x. In the Fourier
domain this is a vector of unit phasors

    C(x) = exp(i * beta * x * omega)            omega: fixed random angular frequencies

and the hypervector itself is its inverse FFT, P(x) = IFFT(C(x)). If omega is
HERMITIAN-symmetric -- omega[0] = omega[d/2] = 0 and omega[d-k] = -omega[k] -- every
C(x) is Hermitian too and P(x) is REAL. In 2-D, with two independent bases,

    P(x, y) = X^x ⊛ Y^y        ⊛ = circular convolution
            = IFFT( exp(i * beta * (x * omega_x + y * omega_y)) )

because circular convolution in the vector domain is an element-wise product in the
Fourier domain. The product of Hermitian spectra is Hermitian, so binding keeps real
vectors real.

Similarity: <P(a), P(b)> / d = (1/d) Σ_j cos(beta * (a - b) * omega_j). With omega
uniform in (-pi, pi) (the default, as in the 2-D FPE experiment) this is ~
sin(pi*beta*Δ) / (pi*beta*Δ): 1 at Δ = 0, 0 at Δ = 1/beta, slightly negative beyond.
beta sets the width: beta = 0.0025 -> 0.89 at 100 px, 0.61 at 200 px, ~0 at 400 px.

With d even, the DC and Nyquist components are fixed to 1, so a real d-vector has
d/2 - 1 independent random frequencies (249 for d = 500).

Scaling: IFFT with norm="ortho", so a vector has norm sqrt(d) and its components have
unit variance (convenient as CNN input); cosine similarity is unaffected.
"""

from __future__ import annotations

import math

import torch

OMEGA_DISTRIBUTIONS = ("uniform", "gaussian")


def hermitian_frequencies(d: int, seed: int, dist: str = "uniform") -> torch.Tensor:
    """(d,) fixed random angular frequencies with Hermitian symmetry (real IFFT).

    uniform  -> omega_j ~ U(-pi, pi)   (sinc kernel, as in the 2-D FPE experiment)
    gaussian -> omega_j ~ N(0, 1)      (Gaussian kernel)
    """
    if dist not in OMEGA_DISTRIBUTIONS:
        raise ValueError(f"omega distribution must be one of {OMEGA_DISTRIBUTIONS}, got {dist!r}")
    g = torch.Generator().manual_seed(int(seed))
    n_free = (d - 1) // 2                                  # independent positive frequencies
    if dist == "uniform":
        w = (torch.rand(n_free, generator=g) * 2 - 1) * math.pi
    else:
        w = torch.randn(n_free, generator=g)
    omega = torch.zeros(d)
    omega[1:n_free + 1] = w
    omega[d - n_free:] = -w.flip(0)                        # omega[d-k] = -omega[k]
    return omega                                           # omega[0] (and omega[d/2]) = 0


def fpe_spectrum(omega: torch.Tensor, beta: float, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Fourier-domain code exp(i * beta * x * omega); omega is placed along ``dim``."""
    shape = [1] * (x.dim() + 1)
    shape[dim] = -1
    xe = x.unsqueeze(dim)
    return torch.exp(1j * beta * xe * omega.view(shape).to(x.device))


def to_real(spectrum: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Hermitian spectrum -> real hypervector (inverse FFT, orthonormal scaling).

    The imaginary part of the IFFT of a Hermitian spectrum is zero up to rounding;
    ``check_real`` verifies it (used in the tests, not on every forward pass).
    """
    return torch.fft.ifft(spectrum, dim=dim, norm="ortho").real


def check_real(spectrum: torch.Tensor, dim: int = -1, tol: float = 1e-4) -> float:
    """Largest |imaginary part| of the IFFT relative to the largest |real part|."""
    v = torch.fft.ifft(spectrum, dim=dim, norm="ortho")
    rel = float(v.imag.abs().max() / v.real.abs().max().clamp_min(1e-12))
    if rel > tol:
        raise ValueError(f"spectrum is not Hermitian: imaginary part {rel:.2e} of the real part")
    return rel


def to_spectrum(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Real hypervector -> its Fourier-domain code (inverse of ``to_real``)."""
    return torch.fft.fft(v, dim=dim, norm="ortho")


def bind(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Binding a ⊛ b: circular convolution, done as an element-wise product of spectra.

    Scaled by 1/sqrt(d) relative to the textbook circular convolution, so FPE codes
    stay FPE codes: bind(P(x1), P(x2)) == P(x1 + x2), norm sqrt(d).
    """
    prod = torch.fft.fft(a, dim=dim, norm="ortho") * torch.fft.fft(b, dim=dim, norm="ortho")
    return torch.fft.ifft(prod, dim=dim, norm="ortho").real


def unbind(c: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Circular correlation: recovers a from c = a ⊛ b (exact for FPE codes, |spectrum| = 1)."""
    prod = torch.fft.fft(c, dim=dim, norm="ortho") * torch.fft.fft(b, dim=dim, norm="ortho").conj()
    return torch.fft.ifft(prod, dim=dim, norm="ortho").real


def similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Cosine similarity of real hypervectors."""
    return (a * b).sum(dim) / (a.norm(dim=dim) * b.norm(dim=dim)).clamp_min(1e-12)


class RealFPE2D(torch.nn.Module):
    """Fixed 2-D real FPE: code(x, y) = IFFT(exp(i*beta*(x*omega_x + y*omega_y))).

    omega_x and omega_y are independent Hermitian bases, generated ONCE from fixed
    seeds and stored as buffers, so every pixel, frame and run uses the same bases.
    """

    def __init__(self, d: int, beta: float, seed: int, dist: str = "uniform") -> None:
        super().__init__()
        self.d, self.beta, self.dist = int(d), float(beta), dist
        self.register_buffer("omega_x", hermitian_frequencies(d, seed, dist))
        self.register_buffer("omega_y", hermitian_frequencies(d, seed + 1, dist))

    def phase(self, x: torch.Tensor, y: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """beta * (x*omega_x + y*omega_y), with the d axis inserted at ``dim``."""
        shape = [1] * (x.dim() + 1)
        shape[dim] = -1
        return self.beta * (x.unsqueeze(dim) * self.omega_x.view(shape)
                            + y.unsqueeze(dim) * self.omega_y.view(shape))

    def spectrum(self, x: torch.Tensor, y: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return torch.exp(1j * self.phase(x, y, dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """x, y: same shape -> real codes with the d axis at ``dim``."""
        return to_real(self.spectrum(x, y, dim), dim=dim)
