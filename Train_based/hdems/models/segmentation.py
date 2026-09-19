"""Segmentation head and motion embedding (~0.5M params)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hdems.seg_features import prepare_seg_features


class SegmentationHead(nn.Module):
    """Motion embedding -> per-pixel class logits."""

    def __init__(
        self,
        d: int = 1024,
        embedding_dim: int = 32,
        num_classes: int = 16,
        *,
        mean_center: bool = False,
        motion_features: bool = False,
    ) -> None:
        super().__init__()
        self.d = d
        self.mean_center = mean_center
        self.motion_features = motion_features
        in_ch = d * 2 + (2 if motion_features else 0)
        self.embed = nn.Sequential(
            nn.Conv2d(in_ch, embedding_dim * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim * 4, embedding_dim, 3, padding=1),
        )
        self.classifier = nn.Conv2d(embedding_dim, num_classes, 1)
        self.register_buffer("feature_mean", torch.zeros(in_ch), persistent=False)

    def forward(
        self,
        phi: torch.Tensor,
        surface: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fm = self.feature_mean if self.mean_center and self.feature_mean.numel() else None
        x = prepare_seg_features(
            phi,
            surface,
            feature_mean=fm,
            mean_center=self.mean_center,
            motion_features=self.motion_features,
        )
        return self.classifier(self.embed(x))


class MotionSegHead(nn.Module):
    """Motion-primary segmentation head.

    The classifier's PRIMARY input is the decoded, ego-compensated residual
    velocity (from ``models.motion``); the HD bundled field ``Phi`` is compressed
    to a small learned context and used only as SECONDARY appearance context.
    This is the opposite emphasis of ``SegmentationHead`` (which feeds the full HV
    and treats motion as an optional 2-channel append), and it puts the motion
    axis back that the cost-volume -> bundle step hid from a raw-HV classifier.
    """

    def __init__(
        self,
        d: int = 1024,
        embedding_dim: int = 32,
        num_classes: int = 16,
        *,
        ctx_dim: int = 32,
        motion_ch: int = 3,
    ) -> None:
        super().__init__()
        self.d = d
        self.motion_ch = motion_ch
        self.ctx = nn.Conv2d(d * 2, ctx_dim, 1)          # HV (real|imag) -> compact context
        in_ch = motion_ch + ctx_dim
        self.embed = nn.Sequential(
            nn.Conv2d(in_ch, embedding_dim * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim * 4, embedding_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(
        self,
        phi: torch.Tensor,
        motion: torch.Tensor | None = None,
        surface: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ctx = self.ctx(torch.cat([phi.real, phi.imag], dim=1).float())
        if motion is None:                               # runnable even if not wired
            b, _, h, w = phi.shape
            motion = torch.zeros(b, self.motion_ch, h, w, device=phi.device)
        x = torch.cat([motion.float(), ctx], dim=1)
        return self.classifier(self.embed(x))


class HVConvHead(nn.Module):
    """HV-as-channels CNN head (the correct way to conv a hypervector field).

    Input is the paper feature tensor ``(B, in_ch, H, W)`` where the hypervector
    is the CHANNEL vector at each grid cell (NOT a reshaped image). A 1x1 conv is
    a per-pixel whole-hypervector template match; the following 3x3 conv adds
    spatial context for coherent segmentation. Trained with ``hdems.train``.
    """

    def __init__(self, in_ch: int, embedding_dim: int = 32, num_classes: int = 16) -> None:
        super().__init__()
        hidden = embedding_dim * 4
        groups = 8 if hidden % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),                 # 1x1: whole-HV template match
            nn.GroupNorm(groups, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, embedding_dim, 3, padding=1),  # 3x3: spatial context
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.net(feats.float()))
