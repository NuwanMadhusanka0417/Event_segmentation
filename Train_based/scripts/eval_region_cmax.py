"""Per-region contrast maximisation (hdems/region_cmax.py) on the validation takes: the
November go/no-go test of the CVPR / ICCV direction. Training-free.

Same frames and scoring as scripts/eval_graphcut.py and scripts/check_protocol_ceiling.py
(holdout takes, evenly spread frames with a mover, event IoU at 12.5 ms, mean over frames),
so the numbers compare directly with the VSA cue (28.9) and the VSA graph cut (29.1) on
the same 60 frames. Thresholds are chosen on these frames, as they were for those two.

Also, on the same frames, from the ground-truth own motion of each event (GT flow minus the
GT static-world flow, depth + Vicon poses):
  oracle event   the perfect per-event cue thresholded -- the ceiling of any motion cue
  oracle region  the mean GT own motion of each CMax region thresholded -- the ceiling of
                 THIS region split (if it is low, the regions mix objects and background)
  speed check    CMax region speed vs the region's GT own speed (slope and median error,
                 regions that truly move) -- are slow motions now measured at full size?

    python scripts/eval_region_cmax.py --max-frames 60 --window-ms 25 50 100 --n-regions 30 60 --rot imu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdems.config import apply_resolution_ratio  # noqa: E402
from hdems.data.build import HOLDOUT, build_dataset  # noqa: E402
from hdems.data.gt_flow import gt_flow_for_sample  # noqa: E402
from hdems.eval import load_config  # noqa: E402
from hdems.imu import vicon_rates, window_mean  # noqa: E402
from hdems.protocols import GTFrames, _spread_cap  # noqa: E402
from hdems.region_cmax import AssignParams, CMaxParams, derotate, segment_iterative, segment_window  # noqa: E402

THR = (0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
GAIN_THR = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
GATE = (0.05, 0.1, 0.2, 0.3)               # speed counts only where the region's gain exceeds this


def frame_iou(p: np.ndarray, g: np.ndarray) -> float:
    u = int((p | g).sum())
    return int((p & g).sum()) / u if u else float("nan")


def best(per_frame: dict, grid) -> tuple[float, float]:
    """{thr: [frame IoUs]} -> (best mean IoU, its threshold)."""
    m = {t: float(np.nanmean(per_frame[t])) for t in grid}
    t = max(m, key=lambda q: m[q] if m[q] == m[q] else -1)
    return m[t], t


def cue_grids():
    """(cue name, threshold grid) of every decision rule scored."""
    return ([("speed", THR), ("speed+own", THR), ("gain", GAIN_THR)]
            + [(f"speed|gain>{gg:g}", THR) for gg in GATE] + [("oracle region", THR)])


def frame_job(job: dict) -> dict:
    """One frame, all configurations -> per-threshold event IoUs, speed pairs, seconds.
    Top-level so it can run in worker processes (--workers)."""
    out = {"ious": {}, "spd": {}, "secs": {}, "oracle_ev": {}}
    gt, own_gt, ts = job["gt"], job["own_gt"], job["ts"]
    for c in job["configs"]:
        win_ms, n_reg = c
        in_w = job["t"] >= ts - win_ms / 1000.0
        t, x, y = job["t"][in_w], job["x"][in_w], job["y"][in_w]
        t0 = time.perf_counter()
        xr, yr = derotate(x, y, t, ts, job["omegas"][win_ms], job["mm"])
        cp = CMaxParams(window_ms=win_ms, n_regions=n_reg, bg=job["bg"])
        o = (segment_iterative(xr, yr, t, ts, cp, AssignParams(lam=job["lam"], iters=job["iters"]))
             if job["method"] == "iterative" else segment_window(xr, yr, t, ts, cp))
        out["secs"][c] = time.perf_counter() - t0
        si = o["score_idx"]
        lbl = gt[y[si], x[si]]
        r = {"speed": {th: frame_iou(o["speed"] > th, lbl) for th in THR},
             "speed+own": {th: frame_iou((o["speed"] > th) & (o["own"] > 0.5), lbl) for th in THR},
             "gain": {th: frame_iou(o["gain"] > th, lbl) for th in GAIN_THR}}
        for gg in GATE:
            r[f"speed|gain>{gg:g}"] = {th: frame_iou((o["speed"] > th) & (o["gain"] > gg), lbl) for th in THR}
        out["spd"][c] = []
        if own_gt is not None:
            ev = own_gt[:, y[si], x[si]]                          # (2, n), toward EARLIER
            reg = o["region"]
            L = int(reg.max()) + 1 if len(reg) else 0
            cnt = np.maximum(np.bincount(reg, minlength=L), 1)
            mvec = np.stack([np.bincount(reg, -ev[0], L), np.bincount(reg, -ev[1], L)], 1) / cnt[:, None]
            rspeed = np.linalg.norm(mvec, axis=1)
            r["oracle region"] = {th: frame_iou(rspeed[reg] > th, lbl) for th in THR}
            movers = rspeed > 0.25
            cm = np.linalg.norm(o["v"] - o["v_bg"], axis=1)
            out["spd"][c] = list(zip(cm[movers], rspeed[movers]))
            if c == job["configs"][0]:
                sp = np.linalg.norm(ev, axis=0)
                out["oracle_ev"] = {th: frame_iou(sp > th, lbl) for th in THR}
        out["ious"][c] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--val-sequences", default="scene13_dyn_test_03,scene14_dyn_test_02")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--window-ms", type=float, nargs="+", default=[50.0])
    ap.add_argument("--n-regions", type=int, nargs="+", default=[30])
    ap.add_argument("--bg", choices=["zero", "median"], default="zero")
    ap.add_argument("--method", choices=["window", "iterative"], default="window",
                    help="window: one motion per k-means region; iterative: + per-event reassignment "
                         "between the region motions (graph cut) and refits")
    ap.add_argument("--lam", type=float, default=AssignParams.lam, help="iterative: smoothness")
    ap.add_argument("--iters", type=int, default=AssignParams.iters, help="iterative: rounds")
    ap.add_argument("--rot", choices=["imu", "vicon", "none"], default="imu",
                    help="camera rotation removed per event: IMU gyro (real sensor), Vicon (motion "
                         "capture: best case), none")
    ap.add_argument("--workers", type=int, default=1, help="frames processed in parallel")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["dataset"] = {**cfg.get("dataset", {}), "val_split": HOLDOUT,
                      "val_sequences": [s for s in args.val_sequences.split(",") if s]}
    cfg = apply_resolution_ratio(cfg, 2)
    ds = build_dataset(cfg, HOLDOUT)
    rot = ds.rotation
    gtf = GTFrames(ds, [12.5])
    keep = [k for k in range(len(ds)) if not isinstance(gtf(k), str)]
    ks = _spread_cap(keep, args.max_frames)
    configs = [(w, n) for w in args.window_ms for n in args.n_regions]
    tmax = max(args.window_ms) / 1000.0
    print(f"[cmax] validation {args.val_sequences}: {len(ks)} frames; windows {args.window_ms} ms; "
          f"regions {args.n_regions}; bg {args.bg}; rotation "
          f"{args.rot}; method {args.method}" + (f" (lam {args.lam:g}, {args.iters} rounds)"
                                                 if args.method == "iterative" else ""), flush=True)

    def jobs():
        for k in ks:
            seq_dir, _fi = ds.index[k]
            seq_dir = Path(seq_dir)
            gt, _evs, fi, ts = gtf(k)
            H, W = gt.shape
            meta = gtf.metas[seq_dir]
            stream = gtf.streams[seq_dir]
            i1 = int(np.searchsorted(stream.t, ts, side="right"))
            i0 = int(np.searchsorted(stream.t, ts - tmax, side="left"))
            t_all = np.asarray(stream.t[i0:i1], dtype=np.float64)
            xy = np.asarray(stream.xy[i0:i1]).astype(np.int64).reshape(-1, 2)
            ok = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
            g = gt_flow_for_sample(seq_dir, fi, ds.window_s, ds.time_frames, meta)
            gs = gt_flow_for_sample(seq_dir, fi, ds.window_s, ds.time_frames, meta, static=True)
            omegas = {}
            for w_ms in args.window_ms:
                om = None
                if args.rot != "none":
                    rt, rw = rot._series(seq_dir.name, meta) if args.rot == "imu" else vicon_rates(meta)
                    om = window_mean(rt, rw, ts - w_ms / 1000.0, ts)
                    if om is not None and args.rot == "imu":
                        om = rot.gain * om
                omegas[w_ms] = om
            yield {"t": t_all[ok], "x": xy[ok, 0], "y": xy[ok, 1], "ts": ts, "mm": meta["meta"], "gt": gt,
                   "own_gt": None if g is None or gs is None else np.nan_to_num(g - gs, nan=0.0),
                   "omegas": omegas, "configs": configs, "method": args.method, "bg": args.bg,
                   "lam": args.lam, "iters": args.iters}

    cues = {c: {name: {t: [] for t in grid} for name, grid in cue_grids()} for c in configs}
    oracle_ev = {t: [] for t in THR}
    spd = {c: [] for c in configs}              # (cmax speed, GT speed) per region that moves
    secs = {c: 0.0 for c in configs}
    pool = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
    results = pool.map(frame_job, jobs()) if pool else map(frame_job, jobs())
    for n, r in enumerate(results):
        for c in configs:
            for name, d in r["ious"][c].items():
                for th, v in d.items():
                    cues[c][name][th].append(v)
            spd[c].extend(r["spd"][c])
            secs[c] += r["secs"][c]
        for th, v in r["oracle_ev"].items():
            oracle_ev[th].append(v)
        if n == 0 or (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(ks)}  " + ", ".join(f"{int(w)}ms/{rg}: {secs[(w, rg)] / (n + 1):.1f} s/frame"
                                                       for w, rg in configs), flush=True)
    if pool:
        pool.shutdown()
    gtf.close()

    print(f"\nVALIDATION, {len(ks)} frames (event IoU %, mean over frames; threshold chosen on these frames)")
    print("  reference, same frames: VSA cue threshold 28.9, VSA graph cut 29.1 (2026-10-21)")
    ob, ot = best(oracle_ev, THR)
    print(f"  oracle event (perfect per-event motion): {100 * ob:.1f}  (thr {ot:g} px)")
    report = {"frames": len(ks), "rot": args.rot, "bg": args.bg, "method": args.method, "lam": args.lam,
              "oracle_event": [ob, ot], "configs": {}}
    for c in configs:
        r = {}
        for name, grid in cue_grids():
            if cues[c][name][grid[0]]:
                r[name] = best(cues[c][name], grid)
        pairs = np.asarray(spd[c]) if spd[c] else np.zeros((0, 2))
        if len(pairs):
            slope = float((pairs[:, 0] * pairs[:, 1]).sum() / (pairs[:, 1] ** 2).sum())
            r["speed_check"] = {"regions": len(pairs), "slope": slope,
                                "median_abs_err": float(np.median(np.abs(pairs[:, 0] - pairs[:, 1]))),
                                "median_gt": float(np.median(pairs[:, 1]))}
        r["s_per_frame"] = secs[c] / max(len(ks), 1)
        report["configs"][f"{int(c[0])}ms_{c[1]}reg"] = r
        print(f"  window {int(c[0]):3d} ms, {c[1]:3d} regions:")
        for k, v in r.items():
            if isinstance(v, tuple):
                print(f"      {k:<18s} {100 * v[0]:5.1f}  (thr {v[1]:g})")
        if "speed_check" in r:
            s = r["speed_check"]
            print(f"      moving regions {s['regions']}: CMax speed = {s['slope']:.2f} x GT "
                  f"(median GT {s['median_gt']:.2f} px, median |error| {s['median_abs_err']:.2f} px); "
                  f"{r['s_per_frame']:.1f} s/frame")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
