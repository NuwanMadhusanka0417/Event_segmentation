"""Ground-truth optical flow for EVIMO2, from depth maps + object/camera poses.

Every labelled pixel lies on a tracked object whose pose is given IN THE CAMERA
FRAME at each GT frame. A 3-D point on object o therefore moves as

    X0 = Z · K⁻¹ p0                           (back-project with the depth map)
    X1 = R_o1 R_o0ᵀ (X0 − t_o0) + t_o1        (carried by the object's motion)
    p1 = π(X1)                                (project)

which covers static surfaces (only the camera moves) and independently moving
objects alike. Pixels without depth (unlabelled background) get NaN.

This is what the VSA-Flow paper evaluates with (EPE against GT flow); here it
lets us check the flow stage on its own before any classifier is trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hdems.data.evimo2_reader import gt_key
from hdems.data.motion_labels import _pose


def gt_flow_between(
    meta: dict[str, Any],
    depth_npz: Any,
    masks_npz: Any,
    j0: int,
    j1: int,
) -> np.ndarray | None:
    """Full-resolution displacement p1 - p0 from GT frame j0 to j1 -> (2, H, W)."""
    frames = meta["frames"]
    f0, f1 = frames[j0], frames[j1]
    dkey, mkey = gt_key("depth", j0), gt_key("mask", j0)    # keyed by POSITION, not id
    if dkey not in depth_npz.files or mkey not in masks_npz.files:
        return None
    mm = meta["meta"]
    fx, fy, cx, cy = float(mm["fx"]), float(mm["fy"]), float(mm["cx"]), float(mm["cy"])
    Z = depth_npz[dkey].astype(np.float64) / 1000.0              # mm -> m
    obj = masks_npz[mkey] // 1000
    H, W = Z.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    X0 = np.stack([(uu - cx) / fx * Z, (vv - cy) / fy * Z, Z], axis=-1)
    out = np.full((2, H, W), np.nan)
    for o in np.unique(obj):
        key = str(int(o))
        if o <= 0 or key not in f0 or key not in f1:
            continue
        R0, _, t0 = _pose(f0[key])
        R1, _, t1 = _pose(f1[key])
        sel = (obj == o) & (Z > 0)
        if not sel.any():
            continue
        X1 = (X0[sel] - t0) @ R0 @ R1.T + t1          # rows of R1 R0^T (X0 - t0) + t1
        ok = X1[:, 2] > 1e-3
        ys, xs = np.nonzero(sel)
        out[0, ys[ok], xs[ok]] = fx * X1[ok, 0] / X1[ok, 2] + cx - uu[sel][ok]
        out[1, ys[ok], xs[ok]] = fy * X1[ok, 1] / X1[ok, 2] + cy - vv[sel][ok]
    return out


def gt_flow_for_sample(
    seq_dir: Path,
    frame_index: int,
    window_s: float,
    time_fracs: list[float],
    meta: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """GT displacement over the interval the cost volume matches (F0 -> F1).

    The sample's reference surface ends at ts - window + f0*window and the first
    target at ts - window + f1*window (f = time_frames). GT frames come at ~60 Hz,
    so the flow is measured between the two GT frames bracketing the reference
    time and rescaled to the F0 -> F1 interval (flow ~ linear over ~17 ms).
    """
    if meta is None:
        meta = np.load(Path(seq_dir) / "dataset_info.npz", allow_pickle=True)["meta"].item()
    frames = meta["frames"]
    ts = float(frames[frame_index]["ts"])
    t_ref = ts - window_s + float(time_fracs[0]) * window_s
    interval = (float(time_fracs[1]) - float(time_fracs[0])) * window_s
    stamps = np.array([float(f["ts"]) if isinstance(f, dict) else np.nan for f in frames])
    j0 = int(np.nanargmin(np.abs(stamps - t_ref)))
    if j0 + 1 >= len(frames):
        return None
    dt = stamps[j0 + 1] - stamps[j0]
    if not np.isfinite(dt) or dt <= 0:
        return None
    depth = np.load(Path(seq_dir) / "dataset_depth.npz")
    masks = np.load(Path(seq_dir) / "dataset_mask.npz")
    g = gt_flow_between(meta, depth, masks, j0, j0 + 1)
    return None if g is None else g * (interval / dt)
