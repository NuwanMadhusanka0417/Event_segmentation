"""Tune / check the gyro timing for rotation removal -- training-free, NEVER on the test set.

Why: on test frames the measured flow on static pixels was 1.28 x the rotational flow
predicted from the gyro (2026-10-11), and precision fell as rotation grew. Either the
gyro window does not match the time span the flow measures (multi-scale: 12.5, 25 and
50 ms back), or the flow itself measures displacements on a different scale.

For every (averaging window, time shift) of the gyro this reports, on STATIC
recent-event pixels of frames with real rotation:
  slope   least-squares gain of measured flow on predicted rotational flow (1 = same scale)
  corr    their correlation (higher = better timing)
and, on all chosen frames, the AUC(moving vs static) of the residual after rotation
removal + the configured background fit -- the number the pipeline cares about. The
best setting is then re-scored with gain = its slope. With --gt the flow's own scale
is measured against ground-truth flow (depth + poses): slope ~1.28 there means the
FLOW over-measures, ~1.0 means the gyro under-predicts.

    python scripts/check_imu_timing.py --config configs/evimo_seg.yaml --split holdout \
        --resolution-ratio 2 --max-frames 60 --device cuda --gt

Apply the result with motion.imu_avg_ms / imu_shift_ms / imu_gain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdems.config import apply_resolution_ratio, frontend_option  # noqa: E402
from hdems.data.build import build_dataset  # noqa: E402
from hdems.data.evimo2_reader import load_meta  # noqa: E402
from hdems.data.gt_flow import gt_flow_for_sample  # noqa: E402
from hdems.eval import load_config  # noqa: E402
from hdems.imu import RotationSource, rotation_flow  # noqa: E402
from hdems.models.hdems import HDEMS  # noqa: E402
from hdems.protocols import GTFrames, _spread_cap  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--split", default="holdout", help="holdout (validation) or train -- never eval")
    ap.add_argument("--resolution-ratio", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--avg-ms", type=float, nargs="+", default=[12.5, 25.0, 50.0])
    ap.add_argument("--shift-ms", type=float, nargs="+", default=[-20.0, -10.0, 0.0, 10.0, 20.0])
    ap.add_argument("--min-rate", type=float, default=0.3, help="rad/s: frames for the slope")
    ap.add_argument("--gt", action="store_true", help="also measure the flow scale against GT flow")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.split == "eval":
        raise SystemExit("never tune on the test (eval) split")

    torch.set_grad_enabled(False)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = apply_resolution_ratio(load_config(args.config), args.resolution_ratio, verbose=True)
    cfg["segmentation"] = {**cfg.get("segmentation", {}), "head": "mfunet", "num_classes": 2}
    model = HDEMS(cfg).to(dev).eval()
    ds = build_dataset(cfg, args.split)
    gtf = GTFrames(ds, [12.5])
    keep = [k for k in range(len(ds)) if not isinstance(gtf(k), str)]
    ks = _spread_cap(keep, args.max_frames)
    r = model.res_ratio
    interval = ds.interval_s
    calib = frontend_option(cfg, "motion", "imu_calib")
    topic = str(frontend_option(cfg, "motion", "imu_topic"))
    print(f"[imu-check] split {args.split}: {len(keep)} frames with a mover, using {len(ks)}; "
          f"interval {interval * 1000:g} ms; ratio {r}", flush=True)

    frames = []
    for n, k in enumerate(ks):
        s = ds[k]
        surf = s["surface"].unsqueeze(0).to(dev)
        flow, _ = model.compute_motion(surf)
        seq_dir, fi = ds.index[k]
        meta = load_meta(seq_dir)
        lab = s["mask"]
        frames.append(dict(surf=surf, flow=flow, lab=lab, score=s["score_mask"] & (lab != 255),
                           seq=seq_dir, fi=fi, meta=meta, ts=float(meta["frames"][fi]["ts"])))
        if (n + 1) % 10 == 0:
            print(f"  flow {n + 1}/{len(ks)}", flush=True)

    def score(avg, shift, gain=1.0):
        src = RotationSource("imu", topic=topic, calib_path=calib, avg_ms=avg, shift_ms=shift, gain=gain)
        rot_v, flow_v, fast, ys, res_v = [], [], [], [], []
        for f in frames:
            mm = f["meta"]["meta"]
            w = src.omega(f["seq"].name, f["meta"], f["ts"])
            if w is None:
                continue
            H, W = f["flow"].shape[-2:]
            rot = torch.from_numpy(rotation_flow(w, float(mm["fx"]), float(mm["fy"]), float(mm["cx"]),
                                                 float(mm["cy"]), interval, H, W, r))[None].to(dev)
            st = (f["score"] & (f["lab"] == 0)).to(dev)
            rot_v.append(rot[0][:, st].reshape(-1).cpu().numpy())
            flow_v.append(f["flow"][0][:, st].reshape(-1).cpu().numpy())
            fast.append(np.linalg.norm(w / max(gain, 1e-9)) > args.min_rate)
            res, _ = model.background_fit(f["flow"], f["surf"], rot)
            sc = f["score"].to(dev)
            ys.append((f["lab"].to(dev)[sc] == 1).cpu().numpy())
            res_v.append(res[0].norm(dim=0)[sc].cpu().numpy())
        if not rot_v:
            raise SystemExit("no frame has a rotation rate (no gyroscope in these sequences?)")
        use = [i for i, q in enumerate(fast) if q] or list(range(len(rot_v)))   # all if none rotates
        rv = np.concatenate([rot_v[i] for i in use])
        fv = np.concatenate([flow_v[i] for i in use])
        slope = float((rv * fv).sum() / max((rv * rv).sum(), 1e-9))
        corr = float(np.corrcoef(rv, fv)[0, 1])
        auc = roc_auc_score(np.concatenate(ys), np.concatenate(res_v))
        return slope, corr, auc, sum(fast)

    print(f"\n{'avg ms':>7s} {'shift ms':>9s} {'slope':>7s} {'corr':>6s} {'AUC':>6s}  (frames with rotation)")
    results = {}
    for avg in args.avg_ms:
        for shift in args.shift_ms:
            sl, co, auc, n = score(avg, shift)
            results[(avg, shift)] = (sl, co, auc)
            print(f"{avg:7g} {shift:9g} {sl:7.3f} {co:6.3f} {auc:6.3f}  ({n})", flush=True)
    cur = (float(frontend_option(cfg, "motion", "imu_avg_ms")), float(frontend_option(cfg, "motion", "imu_shift_ms")))
    best = max(results, key=lambda k: results[k][2])
    sl, co, auc = results[best]
    print(f"\ncurrent setting {cur}: AUC {results.get(cur, (np.nan,) * 3)[2]:.3f}")
    print(f"best timing: avg {best[0]:g} ms, shift {best[1]:g} ms -> AUC {auc:.3f}, slope {sl:.3f}")
    _, _, auc_g, _ = score(best[0], best[1], gain=sl)
    print(f"  with gain {sl:.3f}: AUC {auc_g:.3f}  "
          f"({'use it' if auc_g > auc + 0.002 else 'no better: keep gain 1'})")

    if args.gt:
        fv, gv = [], []
        for f in frames:
            g = gt_flow_for_sample(f["seq"], f["fi"], ds.window_s, ds.time_frames, f["meta"])
            if g is None:
                continue
            H, W = f["flow"].shape[-2:]
            g = torch.from_numpy(g[:, r // 2::r, r // 2::r][:, :H, :W] / r).float()
            st = (f["score"] & (f["lab"] == 0)) & torch.isfinite(g).all(0)
            if st.sum() < 50:
                continue
            fv.append(f["flow"][0].cpu()[:, st].reshape(-1).numpy())
            gv.append(g[:, st].reshape(-1).numpy())
        if fv:
            fv, gv = np.concatenate(fv), np.concatenate(gv)
            print(f"\nflow vs ground-truth flow on static pixels: slope {float((fv * gv).sum() / (gv * gv).sum()):.3f}, "
                  f"corr {float(np.corrcoef(fv, gv)[0, 1]):.3f}  "
                  f"(slope > 1: the FLOW over-measures; the gyro gain should follow it)")
    gtf.close()


if __name__ == "__main__":
    main()
