"""DBSCAN clustering on dynamic pixels."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import DBSCAN


def dbscan_segments(
    dynamic_mask: torch.Tensor,
    residual_flow: torch.Tensor,
    hv_field: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    H, W = dynamic_mask.shape
    labels = torch.zeros(H, W, dtype=torch.int64)
    ys, xs = torch.where(dynamic_mask)
    if ys.numel() == 0:
        return labels

    d = hv_field.shape[0]
    hv = hv_field[:, ys, xs].T.real.numpy()
    hv = hv / (np.linalg.norm(hv, axis=1, keepdims=True) + 1e-8)
    fl = residual_flow[:, ys, xs].T.numpy()
    pos = torch.stack([ys.float(), xs.float()], dim=1).numpy()

    fw = float(cfg.get("flow_weight", 1.0))
    hw = float(cfg.get("hv_weight", 0.5))
    sw = float(cfg.get("spatial_weight", 0.05))
    X = np.concatenate([fw * fl, hw * hv, sw * pos], axis=1)

    eps = float(cfg.get("flow_threshold", 2.0))
    min_samples = int(cfg.get("min_samples", 5))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    seg = db.labels_ + 1
    seg[db.labels_ < 0] = 0
    for i, (y, x) in enumerate(zip(ys.tolist(), xs.tolist())):
        labels[y, x] = int(seg[i])
    return labels
