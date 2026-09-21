"""Instance split + instance-level metric for motion segmentation (Option A).

The binary head answers "is this pixel a moving object?". Instances are then
recovered class-agnostically by connected components on the predicted
foreground, and scored against the GT objects by best-IoU matching (Hungarian
if SciPy is available, otherwise greedy). Class ids are never compared, so an
arbitrary instance numbering cannot penalise a correct segmentation.
"""

from __future__ import annotations

import numpy as np


def connected_components(fg: np.ndarray, min_size: int = 0) -> np.ndarray:
    """Label 4-connected components of a boolean foreground mask -> int map."""
    try:                                              # fast path
        from scipy.ndimage import label as _label
        lab, _ = _label(fg)
    except Exception:                                 # dependency-free fallback
        lab = _label_bfs(fg)
    if min_size > 0:
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        for i, c in zip(ids.tolist(), counts.tolist()):
            if c < min_size:
                lab[lab == i] = 0
    return lab


def _label_bfs(fg: np.ndarray) -> np.ndarray:
    """Iterative 4-connected labelling (used only when SciPy is unavailable)."""
    h, w = fg.shape
    lab = np.zeros((h, w), dtype=np.int64)
    cur = 0
    for sy in range(h):
        for sx in range(w):
            if not fg[sy, sx] or lab[sy, sx]:
                continue
            cur += 1
            stack = [(sy, sx)]
            lab[sy, sx] = cur
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = cur
                        stack.append((ny, nx))
    return lab


def instance_metrics(
    pred_inst: np.ndarray,
    gt_inst: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """Match predicted instances to GT instances by IoU (ids are ignored).

    Returns mean IoU over matched GT instances, plus detection counts at
    ``iou_thresh``. Unmatched GT instances count as IoU 0 (missed).
    """
    if valid is not None:
        pred_inst = np.where(valid, pred_inst, 0)
        gt_inst = np.where(valid, gt_inst, 0)

    gt_ids = [i for i in np.unique(gt_inst) if i > 0]
    pr_ids = [i for i in np.unique(pred_inst) if i > 0]
    if not gt_ids:
        return {"instance_miou": float("nan"), "n_gt": 0, "n_pred": len(pr_ids),
                "matched": 0, "precision": float("nan"), "recall": float("nan")}
    if not pr_ids:
        return {"instance_miou": 0.0, "n_gt": len(gt_ids), "n_pred": 0,
                "matched": 0, "precision": 0.0, "recall": 0.0}

    iou = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float64)
    for gi, g in enumerate(gt_ids):
        gm = gt_inst == g
        gsum = int(gm.sum())
        for pi, p in enumerate(pr_ids):
            pm = pred_inst == p
            inter = int(np.logical_and(gm, pm).sum())
            if inter:
                iou[gi, pi] = inter / (gsum + int(pm.sum()) - inter)

    try:                                              # optimal assignment
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-iou)
        pairs = list(zip(rows.tolist(), cols.tolist()))
    except Exception:                                 # greedy fallback
        pairs, used_g, used_p = [], set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        for gi, pi in order.tolist():
            if iou[gi, pi] <= 0 or gi in used_g or pi in used_p:
                continue
            used_g.add(gi); used_p.add(pi); pairs.append((gi, pi))

    scores = [iou[g, p] for g, p in pairs]
    matched = sum(1 for s in scores if s >= iou_thresh)
    total = sum(scores) / len(gt_ids)                 # unmatched GT -> 0
    return {
        "instance_miou": float(total),
        "n_gt": len(gt_ids),
        "n_pred": len(pr_ids),
        "matched": int(matched),
        "precision": matched / max(len(pr_ids), 1),
        "recall": matched / max(len(gt_ids), 1),
    }


def binary_iou(pred_fg: np.ndarray, gt_fg: np.ndarray,
               valid: np.ndarray | None = None) -> float:
    """Foreground (moving-object) IoU — the headline number for Option A."""
    if valid is not None:
        pred_fg = np.logical_and(pred_fg, valid)
        gt_fg = np.logical_and(gt_fg, valid)
    union = int(np.logical_or(pred_fg, gt_fg).sum())
    if union == 0:
        return float("nan")
    return int(np.logical_and(pred_fg, gt_fg).sum()) / union
