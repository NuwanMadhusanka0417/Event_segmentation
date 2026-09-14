"""Closed-form ridge linear readout over HD segmentation features."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from hdems.ridge_fit import RidgeFitResult, load_ridge_weights
from hdems.seg_features import prepare_seg_features


class RidgeHead(nn.Module):
    """Mean-center -> ``X @ W + b`` on CPU; no learnable parameters."""

    def __init__(
        self,
        num_classes: int,
        *,
        mean_center: bool = True,
        motion_features: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mean_center = mean_center
        self.motion_features = motion_features
        self.register_buffer("weight", torch.zeros(0), persistent=True)
        self.register_buffer("bias", torch.zeros(0), persistent=True)
        self.register_buffer("feature_mean", torch.zeros(0), persistent=True)
        self.alpha: float = 1.0
        self.imbalance: str = "balanced"

    @property
    def is_loaded(self) -> bool:
        return self.weight.numel() > 0

    def _set_buffer(self, name: str, tensor: torch.Tensor) -> None:
        t = tensor.detach().float().cpu()
        if name in self._buffers and self._buffers[name] is not None:
            buf = self._buffers[name]
            if buf.shape == t.shape:
                buf.copy_(t)
                return
        self.register_buffer(name, t, persistent=True)

    def set_from_result(self, result: RidgeFitResult) -> None:
        self._set_buffer("weight", result.weight)
        self._set_buffer("bias", result.bias)
        self._set_buffer("feature_mean", result.feature_mean)
        self.alpha = float(result.alpha)
        self.imbalance = str(result.imbalance)
        self.mean_center = bool(result.mean_center)
        self.motion_features = bool(result.motion_features)
        self.num_classes = int(result.num_classes)

    def load(self, path: str | Path) -> None:
        ckpt = load_ridge_weights(str(path))
        self._set_buffer("weight", ckpt["weight"])
        self._set_buffer("bias", ckpt["bias"])
        self._set_buffer("feature_mean", ckpt["feature_mean"])
        self.alpha = float(ckpt.get("alpha", 1.0))
        self.imbalance = str(ckpt.get("imbalance", "balanced"))
        self.mean_center = bool(ckpt.get("mean_center", True))
        self.motion_features = bool(ckpt.get("motion_features", True))
        self.num_classes = int(ckpt.get("num_classes", self.weight.shape[1]))

    def forward(
        self,
        phi: torch.Tensor,
        surface: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.is_loaded:
            raise RuntimeError("RidgeHead.load(path) must be called before inference")
        x = prepare_seg_features(
            phi,
            surface,
            feature_mean=self.feature_mean if self.mean_center else None,
            mean_center=self.mean_center,
            motion_features=self.motion_features,
        )
        b, _, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        with torch.no_grad():
            logits = x_flat.cpu() @ self.weight + self.bias
        logits = logits.reshape(b, h, w, self.num_classes).permute(0, 3, 1, 2)
        return logits.to(device=phi.device, dtype=torch.float32)

    def num_readout_params(self) -> int:
        if not self.is_loaded:
            return 0
        return int(self.weight.numel() + self.bias.numel())
