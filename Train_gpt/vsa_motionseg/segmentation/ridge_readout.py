"""Optional Ridge regression readout (comparison baseline)."""

from __future__ import annotations

import torch


def fit_ridge(X: torch.Tensor, Y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """X (N,F), Y (N,C). Returns W (F,C) via stable normal equations."""
    X = X.double()
    Y = Y.double()
    XtX = X.T @ X
    reg = alpha * torch.eye(XtX.shape[0], dtype=XtX.dtype)
    W = torch.linalg.solve(XtX + reg, X.T @ Y)
    return W.float()


def predict_ridge(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return X @ W
