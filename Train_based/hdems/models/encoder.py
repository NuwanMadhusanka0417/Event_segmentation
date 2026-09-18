"""Paper-faithful VFA encoder (You et al. 2025, Eq. 2/4/6).

F(x,y) = Σ_{Δx,Δy} T(x+Δx, y+Δy) · K(Δx,Δy),  where the HD kernel is
    K(Δx,Δy) = Gaussian(Δx,Δy) · ( Xᐟᐟ^Δx ⊙ Yᐟᐟ^Δy )
             = Gaussian(Δx,Δy) · exp( i ( Δx·φx + Δy·φy ) ).

FPE is applied to the POSITIONS (Δx,Δy) — building a Gaussian-smoothed
translation-invariant (VFA) kernel — exactly as in the paper. This replaces the
earlier (incorrect) FPE-of-filter-response construction. Implemented as a single
depthwise convolution of the time surface with the d complex kernels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdems.vsa.fpe import make_base_phases


class VSAEncoder(nn.Module):
    """Time surface -> complex descriptor field F, output (B, d, H, W) complex64."""

    def __init__(
        self,
        d: int = 1024,
        patch_size: int = 21,
        sigma_k: float = 1.5,
        rank: int = 64,          # kept for config compatibility (unused: full kernel)
        separable_terms: int = 2,  # kept for config compatibility (unused)
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.d = d
        self.patch_size = patch_size
        self.pad = patch_size // 2

        # FPE base phases for the two axes (Gaussian -> Gaussian kernel similarity).
        phases_x = make_base_phases(d, seed=seed)          # (d,)
        phases_y = make_base_phases(d, seed=seed + 1)
        self.register_buffer("phases_x", phases_x)
        self.register_buffer("phases_y", phases_y)

        # Build the HD kernel K (d, N, N): position code * Gaussian.
        n = self.pad
        coords = torch.arange(-n, n + 1, dtype=torch.float32)
        oy, ox = torch.meshgrid(coords, coords, indexing="ij")     # (N, N) each
        gauss = torch.exp(-(ox ** 2 + oy ** 2) / (2.0 * sigma_k ** 2))  # (N, N)
        # phase[k, iy, ix] = ox·φx[k] + oy·φy[k]
        phase = (ox[None] * phases_x[:, None, None]
                 + oy[None] * phases_y[:, None, None])              # (d, N, N)
        K = gauss[None] * torch.exp(1j * phase)                     # (d, N, N) complex
        # conv2d weights: (out_channels=d, in_channels=1, N, N)
        self.register_buffer("k_real", K.real.unsqueeze(1).contiguous())
        self.register_buffer("k_imag", K.imag.unsqueeze(1).contiguous())

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, surface: torch.Tensor) -> torch.Tensor:
        """surface: (B, C, H, W) float32 -> F: (B, d, H, W) complex64."""
        assert surface.dtype == torch.float32
        # Combine polarity channels into one time surface T, then F = T * K.
        t = surface.sum(dim=1, keepdim=True)                        # (B, 1, H, W)
        f_real = F.conv2d(t, self.k_real, padding=self.pad)         # (B, d, H, W)
        f_imag = F.conv2d(t, self.k_imag, padding=self.pad)
        return torch.complex(f_real, f_imag)
