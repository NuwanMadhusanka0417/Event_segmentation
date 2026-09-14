"""Event-masked segmentation metrics."""

from __future__ import annotations

import torch


def confusion(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[int, int, int, int]:
    p = pred[valid].bool()
    t = target[valid].bool()
    tp = (p & t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    tn = (~p & ~t).sum().item()
    return tp, fp, fn, tn


def iou(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> float:
    if valid is None:
        valid = torch.ones_like(pred, dtype=torch.bool)
    inter = ((pred > 0) & (target > 0) & valid).sum().float()
    union = ((pred > 0) | (target > 0)) & valid
    u = union.sum().float()
    return (inter / u.clamp_min(1)).item()


def precision_recall_f1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    tp, fp, fn, _ = confusion(pred > 0, target > 0, valid)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"precision": prec, "recall": rec, "f1": f1}


def boundary_f1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, tol: int = 2) -> float:
    pb = _boundary(pred > 0)
    tb = _boundary(target > 0)
    pb, tb = pb & valid, tb & valid
    if not pb.any() and not tb.any():
        return 1.0
    tp = 0
    for y, x in pb.nonzero(as_tuple=False):
        y0, y1 = max(0, y - tol), min(pred.shape[0], y + tol + 1)
        x0, x1 = max(0, x - tol), min(pred.shape[1], x + tol + 1)
        if tb[y0:y1, x0:x1].any():
            tp += 1
    prec = tp / max(pb.sum().item(), 1)
    rec = tp / max(tb.sum().item(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-8)


def _boundary(m: torch.Tensor) -> torch.Tensor:
    er = torch.zeros_like(m)
    er[1:, :] |= m[1:, :] != m[:-1, :]
    er[:, 1:] |= m[:, 1:] != m[:, :-1]
    return er & m


def temporal_stability(ids_t: torch.Tensor, ids_t1: torch.Tensor, valid: torch.Tensor) -> float:
    """Fraction of pixels keeping same non-zero label when both frames valid."""
    m = valid & (ids_t > 0) & (ids_t1 > 0)
    if not m.any():
        return 1.0
    same = (ids_t == ids_t1) & m
    return same.sum().float().item() / m.sum().float().item()


def event_masked_valid(event_count: torch.Tensor, min_count: float = 1.0) -> torch.Tensor:
    return event_count >= min_count
