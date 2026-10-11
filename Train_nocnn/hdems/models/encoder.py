"""Paper-faithful VFA encoder (You et al. 2025, Eq. 2-9).

    D(Δx,Δy)  = X^Δx ⊙ Y^Δy                          spatial code          (Eq. 2)
    K_p       = D_p ∗ G                              VFA HD kernel         (Eq. 6)
    F_p^s     = T_p^s ∗ K_p                          per polarity/scale    (Eq. 7)
    F^s       = F_+^s ∘ R_+  +  F_-^s ∘ R_-          polarity role-binding (Eq. 8)
    F         = Σ_s up(F^s) ∘ R_s                    multi-scale fusion    (Eq. 9)

Kernel (Eq. 6) -- a spatial CONVOLUTION of the code field D with a Gaussian G.
For FPE codes, (D∗G)(p) ≈ D(p) ⊙ ĝ with ĝ_k = exp(-σ²|φ_k|²/2): the kernel keeps
the FULL N×N aperture and attenuates high-frequency channels, which gives the
smooth translation-invariant similarity of a VFA (paper Fig. 1b). The earlier
implementation multiplied D by a Gaussian WINDOW instead, which shrinks the
aperture to ~3σ and discards most of the neighbourhood; it is kept as
``kernel="window"`` only to load old checkpoints.

Polarity (Eq. 7-8) -- each polarity has its own kernel and is bound to a random
role vector before bundling, so a positive edge is not matched to a negative one.
Summing polarities first (the old behaviour) lets them cancel and mix.

Binding is the element-wise product in the phasor domain (FPE/HRR); role vectors
are random unit phasors.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdems.vsa.fpe import make_base_phases


def _gaussian(sigma: float) -> torch.Tensor:
    """Normalised 2-D Gaussian G on a (2⌈3σ⌉+1)² grid."""
    r = max(1, int(math.ceil(3.0 * sigma)))
    c = torch.arange(-r, r + 1, dtype=torch.float32)
    g1 = torch.exp(-c ** 2 / (2.0 * sigma ** 2))
    g = g1[:, None] * g1[None, :]
    return g / g.sum()


def build_hd_kernel(d: int, patch_size: int, sigma: float, seed: int, mode: str) -> torch.Tensor:
    """(d, N, N) complex HD kernel for one polarity."""
    phases_x = make_base_phases(d, seed=seed)
    phases_y = make_base_phases(d, seed=seed + 1)
    n = patch_size // 2
    coords = torch.arange(-n, n + 1, dtype=torch.float32)
    oy, ox = torch.meshgrid(coords, coords, indexing="ij")                     # (N, N)
    D = torch.exp(1j * (ox[None] * phases_x[:, None, None]
                        + oy[None] * phases_y[:, None, None]))                   # Eq. 2
    if mode == "window":                       # legacy: Gaussian WINDOW (not the paper)
        gauss = torch.exp(-(ox ** 2 + oy ** 2) / (2.0 * sigma ** 2))
        return gauss[None] * D
    if mode != "conv":
        raise ValueError(f"encoder.kernel must be conv|window, got {mode!r}")
    # Eq. 6: K = D ∗ G, a depthwise 2-D convolution of every code channel with G
    G = _gaussian(sigma)[None, None]                                            # (1,1,g,g)
    pad = G.shape[-1] // 2
    re = F.conv2d(D.real.unsqueeze(1), G, padding=pad).squeeze(1)
    im = F.conv2d(D.imag.unsqueeze(1), G, padding=pad).squeeze(1)
    return torch.complex(re, im)


def _role(d: int, seed: int) -> torch.Tensor:
    """Random unit-phasor role vector (the key in role-filler binding)."""
    g = torch.Generator().manual_seed(seed)
    return torch.exp(1j * torch.rand(d, generator=g) * 2 * math.pi)


class VSAEncoder(nn.Module):
    """Time surface (B, 2, H, W) -> HD descriptor field F (B, d, H, W) complex64."""

    def __init__(
        self,
        d: int = 1024,
        patch_size: int = 21,
        sigma_k: float = 1.5,
        rank: int = 64,            # kept for config compatibility (unused: full kernel)
        separable_terms: int = 2,  # kept for config compatibility (unused)
        seed: int = 0,
        *,
        kernel: str = "conv",
        polarity_binding: bool = True,
        scales: int = 2,
    ) -> None:
        super().__init__()
        self.d = d
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.kernel_mode = kernel
        self.polarity_binding = bool(polarity_binding)
        self.n_scales = max(1, int(scales))

        # Polarity 0 keeps the historical buffer names (k_real / k_imag).
        K0 = build_hd_kernel(d, patch_size, sigma_k, seed, kernel)
        self.register_buffer("k_real", K0.real.unsqueeze(1).contiguous())
        self.register_buffer("k_imag", K0.imag.unsqueeze(1).contiguous())
        if self.polarity_binding:
            K1 = build_hd_kernel(d, patch_size, sigma_k, seed + 2, kernel)       # K_p, Eq. 7
            self.register_buffer("k1_real", K1.real.unsqueeze(1).contiguous())
            self.register_buffer("k1_imag", K1.imag.unsqueeze(1).contiguous())
            R = torch.stack([_role(d, seed + 10), _role(d, seed + 11)])          # R_+, R_-
            self.register_buffer("role_pol", R)
        if self.n_scales > 1:
            Rs = torch.stack([_role(d, seed + 20 + s) for s in range(self.n_scales)])
            self.register_buffer("role_scale", Rs)                                # R_s, Eq. 9

        for p in self.parameters():
            p.requires_grad = False

    def _conv(self, t: torch.Tensor, re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.conv2d(t, re, padding=self.pad),
                             F.conv2d(t, im, padding=self.pad))

    def _describe(self, surface: torch.Tensor) -> torch.Tensor:
        """One scale: both polarities -> one descriptor field (Eq. 7-8)."""
        if not self.polarity_binding:          # legacy: sum polarities, one kernel
            return self._conv(surface.sum(dim=1, keepdim=True), self.k_real, self.k_imag)
        rp = self.role_pol.view(2, 1, -1, 1, 1)
        f = self._conv(surface[:, 0:1], self.k_real, self.k_imag) * rp[0]
        return f + self._conv(surface[:, 1:2], self.k1_real, self.k1_imag) * rp[1]

    def forward(self, surface: torch.Tensor) -> torch.Tensor:
        """surface: (B, 2, H, W) float32 -> F: (B, d, H, W) complex64."""
        assert surface.dtype == torch.float32
        H, W = surface.shape[-2:]
        out: torch.Tensor | None = None
        t = surface
        for s in range(self.n_scales):
            if s > 0:                          # down-interpolation at ratio 1/2 (Sec. 3.2.3)
                t = F.avg_pool2d(t, 2, ceil_mode=True)
            f = self._describe(t)
            if s > 0:                          # up-interpolate back to scale 0 (Eq. 9)
                f = torch.complex(
                    F.interpolate(f.real, size=(H, W), mode="bilinear", align_corners=False),
                    F.interpolate(f.imag, size=(H, W), mode="bilinear", align_corners=False))
            if self.n_scales > 1:
                f = f * self.role_scale[s].view(1, -1, 1, 1)
            out = f if out is None else out + f
        return out
