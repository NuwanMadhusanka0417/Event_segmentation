"""Camera ROTATION from the IMU (or Vicon), and the image motion it causes.

Why
---
The benchmark breakdown (2026-10-10, 491 test frames) put camera rotation first among
the error sources: event IoU 62.7% with a still camera, 28% at 30-60 deg/s and 18%
above 60 deg/s. The affine ego model cannot represent rotation (its flow is quadratic
in the image coordinates) and a large mover can capture the robust fit. Rotation is
exactly what a gyroscope measures, independent of depth, so it can be removed from
the measured flow before anything is fitted.

Rotational flow
---------------
For camera angular velocity w (camera frame, rad/s) a static point seen at pixel p at
time t was, Δ seconds EARLIER, at

    p' = π(K R K⁻¹ p̃),   R = exp([w]ₓ Δ)

(R_wc(t-Δ) = R_wc(t) exp(-[w]ₓΔ) for a body-frame angular velocity). The front end's
flow is exactly that displacement: the reference surface ends at the label time and
the targets lie EARLIER (dataset.time_frames [1, .75, .5, 0]), one interval = 12.5 ms.
No depth is needed. Translation (parallax) is NOT covered -- the ego fit handles what
is left.

Sources (``motion.rot_source``)
-------------------------------
  imu    gyroscope of the Prophesee camera on the same rig (dataset_info meta["imu"],
         ~1 kHz), rotated into the Samsung camera frame with the extrinsic rotation
         estimated ONCE on the training sequences (scripts/calibrate_imu.py ->
         configs/imu_calib.json). A real sensor: usable for test results.
  vicon  angular velocity from the motion-capture camera trajectory. Ground truth:
         an ORACLE / upper bound, never a test result.
  none   no rotation removed (events only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hdems.data.motion_labels import _quat_to_matrix

ROT_SOURCES = ("none", "imu", "vicon")
# For a sequence WITHOUT a gyroscope recording (rot_source imu): "vicon" = motion-capture
# rotation instead (training / validation only -- never the test split), "none" = remove
# nothing (the behaviour before 2026-10-13).
ROT_FALLBACKS = ("none", "vicon")
IMU_TOPICS = {"left": "/prophesee/left/imu", "right": "/prophesee/right/imu"}
_WARNED: set[str] = set()


# --------------------------------------------------------------------------- SO(3)
def hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def so3_exp(v: np.ndarray) -> np.ndarray:
    """Rotation matrix of the rotation vector v (Rodrigues)."""
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3) + hat(v)
    K = hat(v / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of R."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-7:
        return 0.5 * w
    return th / (2.0 * np.sin(th)) * w


# ------------------------------------------------------------------- measurements
def gyro(meta: dict[str, Any], topic: str = "left") -> tuple[np.ndarray, np.ndarray]:
    """(ts (N,), angular velocity (N, 3) rad/s in the IMU frame)."""
    samples = (meta.get("imu") or {}).get(IMU_TOPICS.get(topic, topic)) or []
    if not samples:
        return np.zeros(0), np.zeros((0, 3))
    ts = np.array([float(s["ts"]) for s in samples])
    w = np.array([[float(s["angular_velocity"][k]) for k in "xyz"] for s in samples])
    order = np.argsort(ts)
    return ts[order], w[order]


def vicon_rates(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Camera angular velocity in the CAMERA frame from the motion-capture trajectory.

    meta["full_trajectory"] (~200 Hz) holds the camera pose (camera -> world); the
    body-frame rate between consecutive poses is log(R_iᵀ R_j) / dt, at the midpoint.
    """
    traj = meta.get("full_trajectory") or meta.get("frames") or []
    ts, Rs = [], []
    for f in traj:
        if isinstance(f, dict) and "cam" in f:
            ts.append(float(f.get("ts", f["cam"].get("ts", np.nan))))
            Rs.append(_quat_to_matrix(f["cam"]["pos"]["q"]))
    if len(ts) < 2:
        return np.zeros(0), np.zeros((0, 3))
    ts = np.asarray(ts)
    order = np.argsort(ts)
    ts, Rs = ts[order], [Rs[i] for i in order]
    out_t, out_w = [], []
    for i in range(len(ts) - 1):
        dt = ts[i + 1] - ts[i]
        if dt <= 1e-5:
            continue
        out_t.append(0.5 * (ts[i] + ts[i + 1]))
        out_w.append(so3_log(Rs[i].T @ Rs[i + 1]) / dt)
    return np.asarray(out_t), np.asarray(out_w)


def window_mean(ts: np.ndarray, w: np.ndarray, t0: float, t1: float) -> np.ndarray | None:
    """Mean rate over [t0, t1]; the nearest sample if none falls inside."""
    if ts.size == 0:
        return None
    m = (ts >= t0) & (ts <= t1)
    if m.any():
        return w[m].mean(0)
    i = int(np.argmin(np.abs(ts - 0.5 * (t0 + t1))))
    return w[i] if abs(ts[i] - 0.5 * (t0 + t1)) < 0.05 else None


# -------------------------------------------------------------------- calibration
@dataclass(frozen=True)
class ImuCalib:
    R: np.ndarray            # IMU frame -> camera frame
    dt: float                # IMU timestamp + dt = camera time


def load_calib(path: str | Path, topic: str = "left") -> ImuCalib:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if topic not in doc:
        raise KeyError(f"IMU calibration {path} has no entry for topic {topic!r} "
                       "(run scripts/calibrate_imu.py)")
    c = doc[topic]
    return ImuCalib(np.asarray(c["R"], dtype=np.float64), float(c["dt"]))


def procrustes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising sum |b_i - R a_i|^2 (Kabsch)."""
    U, _, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


# --------------------------------------------------------------- rotational flow
def rotation_flow(
    omega: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    interval_s: float,
    height: int, width: int,
    ratio: int = 1,
) -> np.ndarray:
    """(2, height, width) displacement p(t - interval) - p(t) of a STATIC point under
    camera rotation ``omega`` (camera frame, rad/s), in WORKING pixels.

    The grid is the working grid (1/ratio of the sensor); each working pixel is the
    centre of its ratio x ratio sensor block, as the area downsampling makes it.
    """
    R = so3_exp(np.asarray(omega, dtype=np.float64) * interval_s)
    jj, ii = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    xs = (jj + 0.5) * ratio - 0.5                     # sensor coordinates
    ys = (ii + 0.5) * ratio - 0.5
    X = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)])   # (3, H, W)
    Xr = np.tensordot(R, X, axes=1)
    z = np.clip(Xr[2], 1e-6, None)
    u = fx * Xr[0] / z + cx - xs
    v = fy * Xr[1] / z + cy - ys
    return (np.stack([u, v]) / ratio).astype(np.float32)


class RotationSource:
    """Per-sample camera angular velocity (camera frame) from the IMU or Vicon.

    The rate is averaged over [ts - avg_ms + shift_ms, ts + shift_ms] and multiplied by
    ``gain``. shift / gain default to 0 / 1; scripts/check_imu_timing.py estimates them
    on TRAINING or VALIDATION frames (never the test set) -- e.g. the front end's flow
    can measure displacements on a slightly different scale than the true motion
    (measured flow = 1.28 x IMU rotation on 2026-10-11), and the rotation removed must
    be in the FLOW's units.
    """

    def __init__(self, source: str = "none", *, topic: str = "left",
                 calib_path: str | Path | None = None, avg_ms: float = 25.0,
                 shift_ms: float = 0.0, gain: float = 1.0, fallback: str = "none") -> None:
        if source not in ROT_SOURCES:
            raise ValueError(f"motion.rot_source must be one of {ROT_SOURCES}, got {source!r}")
        if fallback not in ROT_FALLBACKS:
            raise ValueError(f"motion.rot_fallback must be one of {ROT_FALLBACKS}, got {fallback!r}")
        self.source = source
        self.topic = topic
        self.avg_s = float(avg_ms) / 1000.0
        self.shift_s = float(shift_ms) / 1000.0
        self.gain = float(gain)
        self.fallback = fallback
        self.calib = load_calib(calib_path, topic) if source == "imu" else None
        self._rates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.no_imu: set[str] = set()          # sequences that had no gyroscope recording

    def _series(self, key: str, meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if key not in self._rates:
            if self.source == "imu":
                ts, w = gyro(meta, self.topic)
                if ts.size == 0:
                    # EVIMO2 has NO gyroscope in scene9 / 10 / 12 (the strict-split training
                    # scenes). Removing nothing there while the test removes the rotation
                    # trains and tests different inputs; "vicon" keeps training consistent.
                    self.no_imu.add(key)
                    if key not in _WARNED:
                        _WARNED.add(key)
                        print(f"[imu] {key}: no gyroscope recording -> "
                              + ("motion-capture rotation (fallback)" if self.fallback == "vicon"
                                 else "no rotation removed"))
                    self._rates[key] = vicon_rates(meta) if self.fallback == "vicon" else (ts, w)
                else:
                    self._rates[key] = (ts + self.calib.dt, w @ self.calib.R.T)
            else:
                self._rates[key] = vicon_rates(meta)
        return self._rates[key]

    def omega(self, key: str, meta: dict[str, Any], ts: float) -> np.ndarray | None:
        """Mean camera rate over the avg_ms before the label time ``ts`` (shifted by
        shift_ms), times gain (None = no data)."""
        if self.source == "none":
            return None
        t, w = self._series(key, meta)
        w_mean = window_mean(t, w, ts - self.avg_s + self.shift_s, ts + self.shift_s)
        return None if w_mean is None else self.gain * w_mean
