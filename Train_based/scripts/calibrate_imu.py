"""Estimate the IMU -> camera rotation (and time offset) for EVIMO2 samsung_mono.

The Samsung camera has no IMU of its own; the gyroscopes in dataset_info.npz belong
to the two Prophesee cameras on the SAME rig, so one fixed rotation maps their rates
into the Samsung camera frame. It is estimated here ONCE, on the TRAINING sequences
only, by aligning the gyro with the camera rate from the motion-capture trajectory
(orthogonal Procrustes, plus a grid search over the time offset) -- the usual
camera-IMU calibration, never touched by the test sequences.

Also checks that the motion-capture "cam" pose is the camera's OPTICAL frame: a static
object's pose in the camera frame must rotate with exactly the opposite rate.

    python scripts/calibrate_imu.py --config configs/evimo_seg.yaml --out configs/imu_calib.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdems.data.evimo2_reader import find_sequence_dirs, load_meta  # noqa: E402
from hdems.data.motion_labels import _quat_to_matrix  # noqa: E402
from hdems.eval import load_config  # noqa: E402
from hdems.imu import IMU_TOPICS, gyro, procrustes, so3_log, vicon_rates  # noqa: E402


def interval_means(ts: np.ndarray, w: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Exact mean of the piecewise-linear signal over each [t0, t1] (cumulative integral)."""
    I = np.concatenate([[0.0 * w[0]], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(ts)[:, None], 0)])
    Ia = np.stack([np.interp(t0, ts, I[:, k]) for k in range(3)], 1)
    Ib = np.stack([np.interp(t1, ts, I[:, k]) for k in range(3)], 1)
    return (Ib - Ia) / (t1 - t0)[:, None]


def pairs(meta: dict, topic: str, offset: float, min_rate: float):
    """(gyro mean, Vicon rate) over each motion-capture interval with real rotation."""
    tg, wg = gyro(meta, topic)
    tv, wv = vicon_rates(meta)
    if tg.size < 10 or tv.size < 10:
        return None
    half = 0.5 * np.median(np.diff(tv))
    t0, t1 = tv - half - offset, tv + half - offset       # camera time -> IMU time
    ok = (t0 > tg[0]) & (t1 < tg[-1]) & (np.linalg.norm(wv, axis=1) > min_rate)
    if ok.sum() < 5:
        return None
    return interval_means(tg, wg, t0[ok], t1[ok]), wv[ok]


def optical_frame_check(meta: dict) -> tuple[str, float] | None:
    """Best agreement between the 'cam' rate and minus a tracked object's rate in the
    camera frame (a static object gives exactly that). -> (object id, median angle deg)."""
    fr = [f for f in meta["frames"] if isinstance(f, dict) and "cam" in f]
    ids = sorted({k for f in fr for k in f if k.isdigit()})
    best = None
    for oid in ids:
        errs = []
        for a, b in zip(fr[:-1], fr[1:]):
            if oid not in a or oid not in b:
                continue
            dt = float(b["ts"]) - float(a["ts"])
            if dt <= 0:
                continue
            Rc = [_quat_to_matrix(f["cam"]["pos"]["q"]) for f in (a, b)]
            Ro = [_quat_to_matrix(f[oid]["pos"]["q"]) for f in (a, b)]
            w_cam = so3_log(Rc[0].T @ Rc[1]) / dt
            w_obj = -so3_log(Ro[1] @ Ro[0].T) / dt
            if np.linalg.norm(w_cam) > 0.3:
                errs.append(np.degrees(np.arccos(np.clip(
                    w_cam @ w_obj / (np.linalg.norm(w_cam) * np.linalg.norm(w_obj) + 1e-12), -1, 1))))
        if len(errs) >= 10:
            m = float(np.median(errs))
            if best is None or m < best[1]:
                best = (oid, m)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--split", default="train", help="sequences to calibrate on (never eval)")
    ap.add_argument("--out", default="configs/imu_calib.json")
    ap.add_argument("--min-rate", type=float, default=0.2, help="rad/s; slower intervals skipped")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["dataset"]["root"])
    seqs = find_sequence_dirs(root, args.split)
    if not seqs:
        raise SystemExit(f"no sequences under {root / args.split}")
    metas = {s.name: load_meta(s) for s in seqs}
    print(f"[calib] {len(seqs)} {args.split} sequences under {root}")

    checks = [c for c in (optical_frame_check(m) for m in metas.values()) if c]
    if checks:
        med = float(np.median([c[1] for c in checks]))
        print(f"[calib] optical-frame check: 'cam' rate vs best static object, median angle "
              f"{med:.1f} deg over {len(checks)} sequences (small = 'cam' is the optical frame)")

    out = {"note": f"scripts/calibrate_imu.py on the '{args.split}' split of {root}; "
                   "R maps IMU-frame rates into the Samsung camera frame, dt is added to "
                   "IMU timestamps"}
    for topic in IMU_TOPICS:
        best = None
        for off in np.arange(-0.03, 0.0301, 0.001):
            got = [pairs(m, topic, off, args.min_rate) for m in metas.values()]
            got = [g for g in got if g is not None]
            if not got:
                continue
            a = np.concatenate([g[0] for g in got])
            b = np.concatenate([g[1] for g in got])
            R = procrustes(a, b)
            rms = float(np.sqrt(np.mean(np.sum((b - a @ R.T) ** 2, 1))))
            if best is None or rms < best[0]:
                best = (rms, off, R, a, b)
        if best is None:
            print(f"[calib] {topic}: no usable data")
            continue
        rms, off, R, a, b = best
        pred = a @ R.T
        corr = [float(np.corrcoef(pred[:, k], b[:, k])[0, 1]) for k in range(3)]
        rel = rms / float(np.sqrt(np.mean(np.sum(b ** 2, 1))))
        print(f"[calib] {topic}: offset {off * 1000:+.0f} ms  residual {rms:.3f} rad/s "
              f"({rel:.1%} of the rate)  per-axis corr {corr[0]:.3f} {corr[1]:.3f} {corr[2]:.3f}  "
              f"n={len(a)}  rotation angle {np.degrees(np.linalg.norm(so3_log(R))):.1f} deg")
        out[topic] = {"R": R.tolist(), "dt": float(off), "rms_rad_s": rms,
                      "relative_rms": rel, "corr_xyz": corr, "n": int(len(a))}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[calib] wrote {args.out}")


if __name__ == "__main__":
    main()
