"""Which EVIMO2 objects are MOVING at each frame, derived from the pose metadata.

Why this exists
---------------
EVIMO2 masks annotate every tracked surface: the table, static props and the
independently moving objects all get an id. So ``mask > 0`` means "on a tracked
object", NOT "moving". Measured on ``samsung_mono/imo``: ``mask > 0`` covers 61%
(train) / 72% (eval) of pixels, while objects that actually move cover 0.9% / 4.1%.
Training against ``mask > 0`` therefore asks the network to mark the static table
as moving, which no motion feature can explain -- it learns "there are events here".

``dataset_info.npz`` carries, for every frame, the camera pose and each object's
pose in the camera frame. Composing them gives each object's WORLD pose, and an
object that keeps its world pose is static no matter how the camera moves.

Threshold in PIXELS, not m/s
----------------------------
What matters is whether the motion is observable by the sensor, so the world speed
is converted to image-plane displacement over one window:

    px_per_window = fx * window_s * (v_translation + omega * rot_radius_m) / Z

The same 0.1 m/s object is 0.18 px/window in one sequence and 2.1 px in another,
so a fixed m/s threshold would label identical physical speeds inconsistently.
The rotation term is an approximation (a nominal object radius) so that an object
spinning in place still counts as moving -- its surface moves even if its centre
does not.

Objects between the two thresholds are AMBIGUOUS and become ignore-label pixels
rather than being forced into either class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

IGNORE_LABEL = 255


@dataclass(frozen=True)
class MotionParams:
    """Thresholds for "is this object moving?", in image pixels per window."""

    move_px: float = 1.0        # >= this -> moving
    static_px: float = 0.3      # <= this -> static; in between -> ambiguous (ignored)
    window_s: float = 0.05      # the time surface window the labels must agree with
    rot_radius_m: float = 0.05  # nominal object radius for the rotation term
    min_depth_m: float = 0.05   # guard against degenerate Z


@dataclass
class SequenceMotion:
    """Per-frame motion state for one sequence (frame index -> object ids)."""

    moving: dict[int, frozenset[int]]
    ambiguous: dict[int, frozenset[int]]
    speed_px: dict[int, dict[int, float]]

    def frames_with_movers(self) -> set[int]:
        return {fi for fi, ids in self.moving.items() if ids}


def _quat_to_matrix(q: dict[str, float]) -> np.ndarray:
    """(x, y, z, w) quaternion -> 3x3 rotation matrix."""
    x, y, z, w = float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n > 0:
        x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_vec(q: dict[str, float]) -> np.ndarray:
    v = np.array([float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])])
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def _pose(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pose entry -> (rotation matrix, quaternion, translation)."""
    p = entry["pos"]
    q = p["q"]
    t = p["t"]
    return (_quat_to_matrix(q), _quat_vec(q),
            np.array([float(t["x"]), float(t["y"]), float(t["z"])]))


def compute_sequence_motion(meta: dict[str, Any], params: MotionParams) -> SequenceMotion:
    """Per-frame moving / ambiguous object ids for one sequence's metadata."""
    frames = meta.get("frames", [])
    fx = float(meta.get("meta", {}).get("fx", 520.0))
    n = len(frames)

    ids: set[str] = set()
    for f in frames:
        if isinstance(f, dict):
            ids.update(k for k in f if k.isdigit())

    ts = np.full(n, np.nan)
    world_pos: dict[str, np.ndarray] = {i: np.full((n, 3), np.nan) for i in ids}
    world_quat: dict[str, np.ndarray] = {i: np.full((n, 4), np.nan) for i in ids}
    depth: dict[str, np.ndarray] = {i: np.full(n, np.nan) for i in ids}

    for k, f in enumerate(frames):
        if not isinstance(f, dict) or "cam" not in f:
            continue
        ts[k] = float(f.get("ts", np.nan))
        Rc, qc, tc = _pose(f["cam"])
        for i in ids:
            if i not in f:
                continue
            _Ro, qo, to = _pose(f[i])
            world_pos[i][k] = Rc @ to + tc          # object position in the WORLD
            world_quat[i][k] = _quat_mul(qc, qo)    # object orientation in the WORLD
            depth[i][k] = to[2]                     # depth in the CAMERA frame

    moving: dict[int, frozenset[int]] = {}
    ambiguous: dict[int, frozenset[int]] = {}
    speed_px: dict[int, dict[int, float]] = {}

    for k in range(n):
        if np.isnan(ts[k]):
            # No pose for this frame: every object is unknown, so ignore them all.
            moving[k] = frozenset()
            ambiguous[k] = frozenset(int(i) for i in ids)
            speed_px[k] = {}
            continue
        mov, amb, speeds = set(), set(), {}
        for i in ids:
            lo, hi = _neighbours(ts, world_pos[i], k)
            if lo is None or hi is None:
                amb.add(int(i))                     # object present but unmeasurable
                continue
            dt = ts[hi] - ts[lo]
            if dt <= 0:
                amb.add(int(i))
                continue
            v_trans = float(np.linalg.norm(world_pos[i][hi] - world_pos[i][lo]) / dt)
            omega = _angular_speed(world_quat[i][lo], world_quat[i][hi], dt)
            z = depth[i][k]
            if np.isnan(z):
                z = np.nanmedian(depth[i])
            z = max(float(z) if not np.isnan(z) else params.min_depth_m, params.min_depth_m)
            px = fx * params.window_s * (v_trans + omega * params.rot_radius_m) / z
            speeds[int(i)] = px
            if px >= params.move_px:
                mov.add(int(i))
            elif px > params.static_px:
                amb.add(int(i))
        moving[k] = frozenset(mov)
        ambiguous[k] = frozenset(amb)
        speed_px[k] = speeds

    return SequenceMotion(moving=moving, ambiguous=ambiguous, speed_px=speed_px)


def _neighbours(ts: np.ndarray, pos: np.ndarray, k: int) -> tuple[int | None, int | None]:
    """Nearest frames before/after k with a pose (central difference, one-sided at ends)."""
    if np.isnan(pos[k]).any():
        return None, None
    lo = hi = None
    for j in range(k - 1, -1, -1):
        if not np.isnan(pos[j]).any() and not np.isnan(ts[j]):
            lo = j
            break
    for j in range(k + 1, len(ts)):
        if not np.isnan(pos[j]).any() and not np.isnan(ts[j]):
            hi = j
            break
    if lo is None and hi is None:
        return None, None
    return (lo if lo is not None else k), (hi if hi is not None else k)


def _angular_speed(q_lo: np.ndarray, q_hi: np.ndarray, dt: float) -> float:
    """Rotation angle between two world orientations, per second."""
    if np.isnan(q_lo).any() or np.isnan(q_hi).any():
        return 0.0
    dot = float(np.clip(abs(np.dot(q_lo, q_hi)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot)) / dt


_CACHE: dict[tuple[str, MotionParams], SequenceMotion] = {}


def sequence_motion(seq_dir: Path, params: MotionParams,
                    meta: dict[str, Any] | None = None) -> SequenceMotion:
    """Cached per-sequence motion state (pose arithmetic only -- no pixels read)."""
    key = (str(seq_dir), params)
    hit = _CACHE.get(key)
    if hit is None:
        if meta is None:
            info = np.load(Path(seq_dir) / "dataset_info.npz", allow_pickle=True)
            meta = info["meta"].item()
        hit = compute_sequence_motion(meta, params)
        _CACHE[key] = hit
    return hit


def frame_motion(seq_dir: Path, frame_index: int, params: MotionParams,
                 meta: dict[str, Any] | None = None) -> tuple[frozenset[int], frozenset[int]]:
    """(moving ids, ambiguous ids) for one frame."""
    sm = sequence_motion(seq_dir, params, meta)
    return (sm.moving.get(frame_index, frozenset()),
            sm.ambiguous.get(frame_index, frozenset()))


def params_from_config(cfg: dict[str, Any]) -> MotionParams:
    """Read ``motion_label`` from the config; window_s follows dataset.window_ms."""
    ml = cfg.get("motion_label", {}) or {}
    ds = cfg.get("dataset", {}) or {}
    return MotionParams(
        move_px=float(ml.get("move_px", 1.0)),
        static_px=float(ml.get("static_px", 0.3)),
        window_s=float(ds.get("window_ms", 50.0)) / 1000.0,
        rot_radius_m=float(ml.get("rot_radius_m", 0.05)),
    )
