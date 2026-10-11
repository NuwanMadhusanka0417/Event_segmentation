"""Object detection derived from the segmentation output.

A box is the extent of an object's region, so detection needs no extra head:
  GT boxes   <- the EVIMO instance mask (object_id = raw // 1000)
  pred boxes <- connected components of the predicted foreground (motion mode)
                or the predicted class regions (objects mode)

Boxes are matched to GT class-agnostically by IoU, which is the usual protocol
for motion-based detection: the question is "did it find the moving object and
localise it", not "did it guess the right instance number".
"""

from __future__ import annotations

import numpy as np


def masks_to_boxes(inst: np.ndarray, min_area: int = 50) -> dict[int, tuple[int, int, int, int]]:
    """Instance-id map -> {id: (x0, y0, x1, y1)} inclusive pixel bounds."""
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for i in np.unique(inst):
        if i <= 0:
            continue
        ys, xs = np.nonzero(inst == i)
        if ys.size < min_area:
            continue
        boxes[int(i)] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return boxes


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0 + 1), max(0, iy1 - iy0 + 1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    area_b = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    return inter / (area_a + area_b - inter)


def detection_metrics(
    pred: dict[int, tuple[int, int, int, int]],
    gt: dict[int, tuple[int, int, int, int]],
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """Class-agnostic box matching -> precision / recall / mean matched IoU."""
    p_ids, g_ids = list(pred), list(gt)
    if not g_ids:
        return {"det_precision": float("nan"), "det_recall": float("nan"),
                "det_miou": float("nan"), "n_pred": len(p_ids), "n_gt": 0, "tp": 0}
    if not p_ids:
        return {"det_precision": 0.0, "det_recall": 0.0, "det_miou": 0.0,
                "n_pred": 0, "n_gt": len(g_ids), "tp": 0}

    iou = np.zeros((len(g_ids), len(p_ids)))
    for gi, g in enumerate(g_ids):
        for pi, p in enumerate(p_ids):
            iou[gi, pi] = box_iou(gt[g], pred[p])

    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-iou)
        pairs = list(zip(rows.tolist(), cols.tolist()))
    except Exception:                                  # greedy fallback
        pairs, ug, up = [], set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        for gi, pi in order.tolist():
            if iou[gi, pi] <= 0 or gi in ug or pi in up:
                continue
            ug.add(gi); up.add(pi); pairs.append((gi, pi))

    scores = [iou[g, p] for g, p in pairs]
    tp = sum(1 for s in scores if s >= iou_thresh)
    return {
        "det_precision": tp / max(len(p_ids), 1),
        "det_recall": tp / max(len(g_ids), 1),
        "det_miou": float(sum(scores) / len(g_ids)),   # unmatched GT -> 0
        "n_pred": len(p_ids),
        "n_gt": len(g_ids),
        "tp": int(tp),
    }


def draw_boxes(ax, boxes: dict[int, tuple[int, int, int, int]], color: str, label: str | None = None):
    """Overlay boxes on a matplotlib axis (used by the eval panels)."""
    from matplotlib.patches import Rectangle
    for i, (x0, y0, x1, y1) in boxes.items():
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1,
                               fill=False, edgecolor=color, linewidth=1.2))
        ax.text(x0, max(y0 - 2, 0), str(i), color=color, fontsize=6)
    if label:
        ax.plot([], [], color=color, label=label)
