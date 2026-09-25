"""Colour the events that belong to moving objects.

Two renderings, both shown ONLY where there are events -- a fully coloured frame
looks impressive but most of it would be untrained guesswork:

  binary   : moving events red, static events grey, no-event pixels black
  instances: connected components of the predicted moving mask, one colour each,
             so per-object colours come out of a 2-class model (no 32-class head)
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


def event_image(surface) -> np.ndarray:
    """(T,2,H,W) or (2,H,W) time surface -> per-pixel event activity (H,W)."""
    s = _as_numpy(surface)
    return s.reshape(-1, s.shape[-2], s.shape[-1]).sum(0)


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


def save_event_colour_figure(
    surface,
    pred,
    event_mask,
    out_path: Path,
    *,
    gt_moving=None,
    min_instance: int = 50,
) -> np.ndarray:
    """Write the events / prediction / instances figure; returns the instance map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = _as_numpy(event_mask).astype(bool)
    moving = (_as_numpy(pred) > 0) & events
    instances = connected_components(moving, min_size=min_instance)
    n_inst = len({int(v) for v in np.unique(instances) if v > 0})

    panels = [
        (event_image(surface), "events", dict(cmap="gray")),
        (colour_binary(moving, events), "predicted moving (red)", {}),
        (colour_instances(instances, events), f"instances ({n_inst})", {}),
    ]
    if gt_moving is not None:
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
