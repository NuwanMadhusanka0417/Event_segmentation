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


def gt_static_flow_between(
    meta: dict[str, Any],
    depth_npz: Any,
    j0: int,
    j1: int,
) -> np.ndarray | None:
    """Full-resolution displacement p1 - p0 a STATIC world point would make -> (2, H, W).

    The camera pose (camera -> world: X_w = R_c X_c + t_c) moves; the point does not:

        X1 = R_c1ᵀ (R_c0 X0 + t_c0 − t_c1)

    This is the ideal "background" the ego fit tries to recover -- exact parallax from
    the true depth, rotation and translation. On a moving object it is where the
    object WOULD be if it were static, so flow − this = the object's own motion.
    Defined wherever there is depth (EVIMO2: the tracked objects, table included);
    NaN on the unlabelled background.
    """
    frames = meta["frames"]
    f0, f1 = frames[j0], frames[j1]
    dkey = gt_key("depth", j0)
    if dkey not in depth_npz.files or not (isinstance(f0, dict) and isinstance(f1, dict)) \
            or "cam" not in f0 or "cam" not in f1:
        return None
    mm = meta["meta"]
    fx, fy, cx, cy = float(mm["fx"]), float(mm["fy"]), float(mm["cx"]), float(mm["cy"])
    Z = depth_npz[dkey].astype(np.float64) / 1000.0              # mm -> m
    H, W = Z.shape
    Rc0, _, tc0 = _pose(f0["cam"])
    Rc1, _, tc1 = _pose(f1["cam"])
    out = np.full((2, H, W), np.nan)
    sel = Z > 0
    if not sel.any():
        return out
    ys, xs = np.nonzero(sel)
    z = Z[sel]
    X0 = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], axis=-1)
    X1 = (X0 @ Rc0.T + tc0 - tc1) @ Rc1                # rows of R_c1ᵀ (R_c0 X0 + t_c0 − t_c1)
    ok = X1[:, 2] > 1e-3
    out[0, ys[ok], xs[ok]] = fx * X1[ok, 0] / X1[ok, 2] + cx - xs[ok]
    out[1, ys[ok], xs[ok]] = fy * X1[ok, 1] / X1[ok, 2] + cy - ys[ok]
    return out


def _bracket(meta: dict[str, Any], frame_index: int, window_s: float,
             time_fracs: list[float]) -> tuple[int, float] | None:
    """(j0, scale): GT frames j0 -> j0+1 bracket the reference time; scale turns their
    displacement into the F0 -> F1 interval (negative: the targets lie EARLIER)."""
    frames = meta["frames"]
    ts = float(frames[frame_index]["ts"])
    t_ref = ts - window_s + float(time_fracs[0]) * window_s
    interval = (float(time_fracs[1]) - float(time_fracs[0])) * window_s
    stamps = np.array([float(f["ts"]) if isinstance(f, dict) and "ts" in f else np.nan
                       for f in frames])
    if not np.isfinite(stamps).any():
        return None
    j0 = int(np.nanargmin(np.abs(stamps - t_ref)))
    if j0 + 1 >= len(frames):
        return None
    dt = stamps[j0 + 1] - stamps[j0]
    if not np.isfinite(dt) or dt <= 0:
        return None
    return j0, interval / dt


def gt_flow_for_sample(
    seq_dir: Path,
    frame_index: int,
    window_s: float,
    time_fracs: list[float],
    meta: dict[str, Any] | None = None,
    *,
    static: bool = False,
) -> np.ndarray | None:
    """GT displacement over the interval the cost volume matches (F0 -> F1).

    The sample's reference surface ends at ts - window + f0*window and the first
    target at ts - window + f1*window (f = time_frames). GT frames come at ~60 Hz,
    so the flow is measured between the two GT frames bracketing the reference
    time and rescaled to the F0 -> F1 interval (flow ~ linear over ~17 ms).

    ``static`` -> the static-world (camera-only) flow instead (gt_static_flow_between).
    """
    if meta is None:
        meta = np.load(Path(seq_dir) / "dataset_info.npz", allow_pickle=True)["meta"].item()
    br = _bracket(meta, frame_index, window_s, time_fracs)
    if br is None:
        return None
    j0, scale = br
    with np.load(Path(seq_dir) / "dataset_depth.npz") as depth:
        if static:
            g = gt_static_flow_between(meta, depth, j0, j0 + 1)
        else:
            with np.load(Path(seq_dir) / "dataset_mask.npz") as masks:
                g = gt_flow_between(meta, depth, masks, j0, j0 + 1)
    return None if g is None else g * scale


def to_working(g: np.ndarray, ratio: int, height: int, width: int) -> np.ndarray:
    """Full-resolution GT displacement (2, H, W) -> working grid (2, height, width),
    working px: NaN-aware mean over each ratio x ratio sensor block (the area
    downsampling the surfaces get), / ratio. A block with no finite value stays NaN."""
    r = int(ratio)
    g = g[:, : height * r, : width * r]
    if g.shape[1] < height * r or g.shape[2] < width * r:          # pad a short sensor
        pad = np.full((2, height * r, width * r), np.nan)
        pad[:, : g.shape[1], : g.shape[2]] = g
        g = pad
    if r == 1:
        return g.astype(np.float32)
    blocks = g.reshape(2, height, r, width, r)
    ok = np.isfinite(blocks)
    n = ok.sum(axis=(2, 4))
    s = np.where(ok, blocks, 0.0).sum(axis=(2, 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan) / r
    return out.astype(np.float32)
