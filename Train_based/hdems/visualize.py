"""Colour the events that belong to moving objects -- one colour per OBJECT.

All renderings are shown ONLY where there are events -- a fully coloured frame
looks impressive but most of it would be untrained guesswork:

  binary  : moving events red, static events grey, no-event pixels black
  objects : each independently moving object in its own colour. The objects come
            from hdems.grouping: the CNN's moving pixels grouped by the motion
            model they fit, so a whole object gets one colour even when its flow
            direction varies (rotation). Without grouping, connected components.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hdems.instances import connected_components

# Distinguishable instance colours (RGB 0-1). Index 0 is never used (background).
_INSTANCE_COLOURS = np.array([
    [0.90, 0.10, 0.10], [0.20, 0.60, 1.00], [0.20, 0.80, 0.30], [1.00, 0.70, 0.10],
    [0.75, 0.30, 0.95], [0.10, 0.85, 0.85], [1.00, 0.45, 0.75], [0.60, 0.85, 0.20],
    [1.00, 0.55, 0.20], [0.45, 0.45, 0.95], [0.95, 0.85, 0.25], [0.35, 0.75, 0.65],
])

_STATIC_GREY = np.array([0.45, 0.45, 0.45])


def _as_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


_SLOW_YELLOW = np.array([0.85, 0.75, 0.25])


def event_image(surface) -> np.ndarray:
    """(T,2,H,W) or (2,H,W) time surface -> per-pixel event activity, contrast-stretched.

    With the paper's 35 ms decay most values are tiny, so a plain grey map was almost
    black; clipping at the 99th percentile of active pixels makes the scene visible.
    """
    s = _as_numpy(surface)
    img = s.reshape(-1, s.shape[-2], s.shape[-1]).sum(0)
    active = img[img > 0]
    top = float(np.percentile(active, 99)) if active.size else 1.0
    return np.clip(img / max(top, 1e-9), 0.0, 1.0)


def colour_binary(moving: np.ndarray, events: np.ndarray) -> np.ndarray:
    """moving/static/no-event -> (H, W, 3) RGB."""
    rgb = np.zeros((*events.shape, 3), dtype=np.float32)      # no events -> black
    rgb[events & ~moving] = _STATIC_GREY
    rgb[events & moving] = _INSTANCE_COLOURS[0]               # red
    return rgb


def colour_instances(instances: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Instance-id map -> (H, W, 3) RGB, one colour per object, black off-events."""
    rgb = np.zeros((*events.shape, 3), dtype=np.float32)
    rgb[events] = _STATIC_GREY
    for k, i in enumerate(sorted({int(v) for v in np.unique(instances) if v > 0})):
        rgb[(instances == i) & events] = _INSTANCE_COLOURS[k % len(_INSTANCE_COLOURS)]
    return rgb


def _count(lbl: np.ndarray) -> int:
    return len({int(v) for v in np.unique(lbl) if v > 0})


def save_event_colour_figure(
    surface,
    pred,
    event_mask,
    out_path: Path,
    *,
    gt_moving=None,
    objects=None,
    objects_gt_mask=None,
    gt_objects=None,
    gt_slow=None,
    min_instance: int = 50,
    pred_title: str | None = None,
    objects_title: str | None = None,
) -> np.ndarray:
    """Write the events / moving / objects / ground-truth figure; returns the object map.

    objects          object ids from the CNN's moving pixels (hdems.grouping)
    objects_gt_mask  the same grouping on the GROUND-TRUTH moving pixels -- the best
                     the grouping can do; the gap to ``objects`` is the CNN's share
    gt_objects       ground-truth independently moving objects (rigid parts merged)
    pred_title / objects_title  panel titles when the decision is not the CNN's
                     (e.g. the training-free VSA grouping)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = _as_numpy(event_mask).astype(bool)
    moving = (_as_numpy(pred) > 0) & events
    if objects is None:                       # no flow available: spatial pieces only
        instances = connected_components(moving, min_size=min_instance)
        obj_title = f"connected pieces ({_count(instances)})"
    else:
        instances = _as_numpy(objects).astype(np.int64)
        obj_title = f"{objects_title or 'OBJECTS: CNN + motion grouping'} ({_count(instances)})"

    panels = [
        (event_image(surface), "events", dict(cmap="gray")),
        (colour_binary(moving, events), pred_title or "CNN: moving (red)", {}),
        (colour_instances(instances, events), obj_title, {}),
    ]
    if objects_gt_mask is not None:
        og = _as_numpy(objects_gt_mask).astype(np.int64)
        panels.append((colour_instances(og, events),
                       f"grouping on GT moving px ({_count(og)}) [best case]", {}))
    if gt_objects is not None:
        gt = _as_numpy(gt_objects).astype(np.int64)
        rgb = colour_instances(gt, events)
        title = f"ground-truth objects ({_count(gt)})"
        if gt_slow is not None:
            slow = (_as_numpy(gt_slow) > 0) & events & (gt == 0)
            n_slow = _count(np.where(slow, _as_numpy(gt_slow), 0))
            if n_slow:
                # moving, but too slowly to call -> ignored in training and scoring
                rgb[slow] = _SLOW_YELLOW
                title += f" + {n_slow} too slow (yellow, ignored)"
        panels.append((rgb, title, {}))
    elif gt_moving is not None:
        gt = _as_numpy(gt_moving).astype(np.int64) // 1000
        panels.append((colour_instances(gt, events), "ground-truth moving", {}))

    fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    for a, (im, title, kw) in zip(np.atleast_1d(ax), panels):
        a.imshow(im, interpolation="nearest", **kw)
        a.set_title(title, fontsize=10)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return instances
