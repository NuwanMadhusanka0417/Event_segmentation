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


class MotionFirstHead(nn.Module):
    """Motion-first HV head (head ``mfcnn``).

    Real-data ablation of the ``cnn`` head (2026-09-29): zeroing the velocity
    input changed 1.3-4.6% of its predictions, zeroing the appearance input
    25%. It had learned what moving objects LOOK like in the training scenes,
    which does not transfer. Here motion is the main input and appearance a
    small context the head must learn to do without:

      motion     : 1x1 readout of the velocity code Mv (fixed units)  -> motion_dim ch
                   + 6 explicit motion channels (``motion_scalars``: residual,
                   raw flow and camera speed)
      appearance : 1x1 projection of unit-RMS X (F or Phi)           -> app_dim ch
                   (8-16), zeroed for a whole sample with probability
                   ``app_dropout`` while training

    X and Mv are NOT fused (no event_combine): fusing them is what let the
    appearance term drown the velocity term.
    """

    def __init__(
        self,
        d: int,
        num_classes: int = 2,
        *,
        embedding_dim: int = 32,
        motion_dim: int = 32,
        app_dim: int = 8,
        app_dropout: float = 0.5,
        scalar_ch: int = 6,
    ) -> None:
        super().__init__()
        self.motion_proj = nn.Conv2d(2 * d, motion_dim, 1)
        self.app_proj = nn.Conv2d(2 * d, app_dim, 1) if app_dim > 0 else None
        self.app_dropout = float(app_dropout)
        hidden = embedding_dim * 4
        groups = 8 if hidden % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(motion_dim + scalar_ch + max(app_dim, 0), hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, embedding_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, mv: torch.Tensor, x: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """mv, x: (B, d, H, W) complex (x already unit-RMS); scalars: (B, 6, H, W)."""
        parts = [self.motion_proj(torch.cat([mv.real, mv.imag], dim=1).float()), scalars.float()]
        if self.app_proj is not None:
            a = self.app_proj(torch.cat([x.real, x.imag], dim=1).float())
            if self.training and self.app_dropout > 0:
                keep = (torch.rand(a.shape[0], 1, 1, 1, device=a.device) >= self.app_dropout)
                a = a * keep.to(a.dtype)
            parts.append(a)
        return self.classifier(self.net(torch.cat(parts, dim=1)))
