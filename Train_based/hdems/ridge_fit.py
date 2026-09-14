"""Closed-form ridge readout fitting (streaming + sklearn reference)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


def ova_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    n = y.shape[0]
    t = torch.full((n, num_classes), -1.0, dtype=torch.float64)
    for c in range(num_classes):
        t[y == c, c] = 1.0
    return t


def sample_weights_balanced(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    n = y.shape[0]
    counts = torch.bincount(y.clamp(min=0), minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    w_class = n / (num_classes * counts)
    return w_class[y.clamp(min=0)]


@dataclass
class RidgeFitResult:
    weight: torch.Tensor
    bias: torch.Tensor
    feature_mean: torch.Tensor
    alpha: float
    imbalance: str
    feature_dim: int
    num_classes: int
    mean_center: bool = True
    motion_features: bool = True


def fit_ridge_streaming(
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    num_classes: int,
    alpha: float = 1.0,
    imbalance: Literal["none", "balanced", "subsample_bg"] = "balanced",
    bg_subsample_ratio: float = 0.2,
    seed: int = 0,
    feature_mean: torch.Tensor | None = None,
    mean_center: bool = True,
    motion_features: bool = True,
) -> RidgeFitResult:
    d: int | None = None
    a_mat = None
    b_mat = None
    rng = np.random.default_rng(seed)

    for x_cpu, y_cpu in batches:
        x = x_cpu.to(torch.float64)
        y = y_cpu.long()
        if d is None:
            d = x.shape[1]
            a_mat = torch.zeros(d, d, dtype=torch.float64)
            b_mat = torch.zeros(d, num_classes, dtype=torch.float64)

        if imbalance == "subsample_bg":
            fg = y > 0
            bg = y == 0
            bg_idx = bg.nonzero(as_tuple=True)[0].numpy()
            if bg_idx.size and fg.any():
                keep_n = max(1, int(bg_subsample_ratio * bg_idx.size))
                keep = rng.choice(bg_idx, size=min(keep_n, bg_idx.size), replace=False)
                sel = torch.zeros(y.shape[0], dtype=torch.bool)
                sel[fg] = True
                sel[torch.from_numpy(keep).long()] = True
                x, y = x[sel], y[sel]

        sw = torch.ones(x.shape[0], dtype=torch.float64)
        if imbalance == "balanced":
            sw = sample_weights_balanced(y, num_classes).to(torch.float64)

        wx = x * sw.sqrt().unsqueeze(1)
        t = ova_targets(y, num_classes)
        assert a_mat is not None and b_mat is not None
        a_mat += wx.T @ wx
        b_mat += wx.T @ (t * sw.unsqueeze(1))

    assert d is not None and a_mat is not None and b_mat is not None
    a_mat += float(alpha) * torch.eye(d, dtype=torch.float64)
    weight = torch.linalg.solve(a_mat, b_mat)
    bias = torch.zeros(num_classes, dtype=torch.float64)
    fm = feature_mean if feature_mean is not None else torch.zeros(d)
    return RidgeFitResult(
        weight=weight.float(),
        bias=bias.float(),
        feature_mean=fm.float(),
        alpha=float(alpha),
        imbalance=imbalance,
        feature_dim=d,
        num_classes=num_classes,
        mean_center=mean_center,
        motion_features=motion_features,
    )


def fit_ridge_sklearn(
    x_all: np.ndarray,
    y_all: np.ndarray,
    *,
    num_classes: int,
    alpha: float = 1.0,
    class_weight: str | None = "balanced",
    feature_mean: torch.Tensor | None = None,
) -> RidgeFitResult:
    from sklearn.linear_model import RidgeClassifier

    clf = RidgeClassifier(alpha=alpha, class_weight=class_weight)
    clf.fit(x_all, y_all)
    w = torch.from_numpy(clf.coef_.T.copy()).float()
    b = torch.from_numpy(clf.intercept_.copy()).float()
    d = w.shape[0]
    fm = feature_mean if feature_mean is not None else torch.zeros(d)
    return RidgeFitResult(
        weight=w,
        bias=b,
        feature_mean=fm.float(),
        alpha=alpha,
        imbalance=class_weight or "none",
        feature_dim=d,
        num_classes=num_classes,
    )


def save_ridge_weights(path: str, result: RidgeFitResult, extra: dict | None = None) -> None:
    payload = {
        "weight": result.weight,
        "bias": result.bias,
        "feature_mean": result.feature_mean,
        "alpha": result.alpha,
        "imbalance": result.imbalance,
        "feature_dim": result.feature_dim,
        "num_classes": result.num_classes,
        "mean_center": result.mean_center,
        "motion_features": result.motion_features,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_ridge_weights(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
