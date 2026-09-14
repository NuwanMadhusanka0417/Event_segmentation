"""Frozen multi-scale VSA/VFA encoder with polarity and scale role vectors."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vsa_motionseg.vsa.fpe import bind, fpe, make_base_phases
from vsa_motionseg.vsa.kernels import eigen_basis


def separable_approx(filters: torch.Tensor, n_terms: int = 2):
    out = []
    for f in filters:
        U, S, Vh = torch.linalg.svd(f.double())
        terms = [
            (U[:, i] * S[i].sqrt(), Vh[i] * S[i].sqrt())
            for i in range(min(n_terms, S.numel()))
        ]
        out.append(terms)
    return out


class MultiScaleVSAEncoder(nn.Module):
    """Time surface (B,C,H,W) -> complex hypervector field (B,D,H,W)."""

    def __init__(
        self,
        d: int = 1024,
        patch_size: int = 21,
        sigma_k: float = 1.5,
        rank: int = 64,
        separable_terms: int = 2,
        scales: list[int] | None = None,
        seed: int = 42,
        representation: str = "real",
        artifact_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.d = d
        self.patch_size = patch_size
        self.rank = rank
        self.scales = scales or [1, 2, 4]
        self.representation = representation

        filters, energy = eigen_basis(patch_size, sigma_k, rank)
        self.register_buffer("filters", filters)
        self.energy_fraction = energy
        self.separable = separable_approx(filters, separable_terms)
        self.pad = patch_size // 2

        self.register_buffer("phases_x", make_base_phases(d, seed=seed))
        self.register_buffer("phases_y", make_base_phases(d, seed=seed + 1))
        self.register_buffer("role_pos", make_base_phases(d, seed=seed + 2))
        self.register_buffer("role_neg", make_base_phases(d, seed=seed + 3))
        for i, _ in enumerate(self.scales):
            r = make_base_phases(d, seed=seed + 10 + i)
            self.register_buffer(f"role_scale_{i}", r)

        for p in self.parameters():
            p.requires_grad = False

        if artifact_dir:
            self.save_artifacts(Path(artifact_dir))

    def _conv_separable(self, x: torch.Tensor, terms: list) -> torch.Tensor:
        out = torch.zeros_like(x)
        for col, row in terms:
            row_f = row.to(dtype=x.dtype, device=x.device).view(1, 1, 1, -1)
            col_f = col.to(dtype=x.dtype, device=x.device).view(1, 1, -1, 1)
            tmp = F.conv2d(x, row_f, padding=(0, self.pad))
            out = out + F.conv2d(tmp, col_f, padding=(self.pad, 0))
        return out

    def _encode_surface(self, surface: torch.Tensor) -> torch.Tensor:
        B, C, H, W = surface.shape
        F_out = torch.zeros(B, self.d, H, W, dtype=torch.complex64, device=surface.device)
        for r in range(self.rank):
            feat = torch.zeros(B, 1, H, W, device=surface.device)
            for c in range(C):
                feat = feat + self._conv_separable(surface[:, c : c + 1], self.separable[r])
            code = fpe(self.phases_x, feat.squeeze(1)) * fpe(self.phases_y, feat.squeeze(1))
            F_out = F_out + code.permute(0, 3, 1, 2)
        return F_out

    def _bind_polarity(self, F_pos: torch.Tensor, F_neg: torch.Tensor) -> torch.Tensor:
        rp = self.role_pos.view(1, -1, 1, 1)
        rn = self.role_neg.view(1, -1, 1, 1)
        return bind(F_pos, rp) + bind(F_neg, rn)

    def forward(self, surface: torch.Tensor) -> torch.Tensor:
        """
        surface: (B, 2, H, W) positive/negative time surfaces (or C channels).
        """
        B, C, H, W = surface.shape
        pos = surface[:, 0:1] if C >= 1 else surface
        neg = surface[:, 1:2] if C >= 2 else torch.zeros_like(pos)

        F_final = torch.zeros(B, self.d, H, W, dtype=torch.complex64, device=surface.device)
        for i, s in enumerate(self.scales):
            if s == 1:
                sp, sn = pos, neg
            else:
                sp = F.avg_pool2d(pos, s, s)
                sn = F.avg_pool2d(neg, s, s)
            Fp = self._encode_surface(torch.cat([sp, torch.zeros_like(sp)], dim=1))
            Fn = self._encode_surface(torch.cat([torch.zeros_like(sn), sn], dim=1))
            F_scale = self._bind_polarity(Fp, Fn)
            role = getattr(self, f"role_scale_{i}").view(1, -1, 1, 1)
            F_up = bind(F_scale, role)
            if s > 1:
                F_up = F.interpolate(
                    F_up.real, size=(H, W), mode="bilinear", align_corners=False
                ) + 1j * F.interpolate(
                    F_up.imag, size=(H, W), mode="bilinear", align_corners=False
                )
            F_final = F_final + F_up

        if self.representation == "bipolar":
            hv = torch.sign(F_final.real)
            return hv.to(torch.float32)
        if self.representation == "binary":
            return (F_final.real > 0).to(torch.float32)
        return F_final

    def save_artifacts(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "filters": self.filters,
                "phases_x": self.phases_x,
                "phases_y": self.phases_y,
                "role_pos": self.role_pos,
                "role_neg": self.role_neg,
                "scales": self.scales,
                "d": self.d,
                "rank": self.rank,
            },
            directory / "encoder_artifacts.pt",
        )
