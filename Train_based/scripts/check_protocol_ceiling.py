"""How high can a motion-based decision score under OUR benchmark protocol? Training-free.

Per event of the 12.5 ms window, the PERFECT motion cue is used: the ground-truth flow minus
the ground-truth static-world flow (depth + Vicon poses), i.e. each event's own motion
relative to the static scene. Events without depth (untracked background, which is static)
get 0. Thresholding this oracle cue is the ceiling of any "segment by motion" method --
a better flow, ego model or decision stage cannot beat it under this protocol.

If the ceiling is far below a published number, that number was measured differently
(event window, which objects count as moving, ...), not just with a better method.

    python scripts/check_protocol_ceiling.py --split holdout --max-frames 60
    python scripts/check_protocol_ceiling.py --split benchmark              # the 5 test sequences
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdems.config import apply_resolution_ratio  # noqa: E402
from hdems.data.build import HOLDOUT, build_dataset  # noqa: E402
from hdems.data.gt_flow import gt_flow_for_sample  # noqa: E402
from hdems.eval import load_config  # noqa: E402
from hdems.protocols import HUA2025, GTFrames, _spread_cap  # noqa: E402
from hdems.veckm import INTERVAL_S  # noqa: E402

THR = (0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--split", choices=["holdout", "benchmark"], default="holdout")
    ap.add_argument("--val-sequences", default="scene13_dyn_test_03,scene14_dyn_test_02")
    ap.add_argument("--max-frames", type=int, default=60, help="holdout: frames; benchmark: per sequence")
    ap.add_argument("--window-ms", type=float, default=12.5)
    ap.add_argument("--noise-px", type=float, nargs="+", default=[0.0],
                    help="add Gaussian flow error of this size (px / interval, per axis) to the "
                         "perfect motion: how accurate must a flow be?")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["dataset"] = {**cfg.get("dataset", {}), "val_split": HOLDOUT,
                      "val_sequences": [s for s in args.val_sequences.split(",") if s]}
    cfg = apply_resolution_ratio(cfg, 2)
    split = HOLDOUT if args.split == "holdout" else cfg["dataset"].get("eval_split", "eval")
    ds = build_dataset(cfg, split)
    gtf = GTFrames(ds, [args.window_ms])
    if args.split == "holdout":
        keep = [k for k in range(len(ds)) if not isinstance(gtf(k), str)]
        frames = [("val", k) for k in _spread_cap(keep, args.max_frames)]
    else:
        by_seq: dict[str, list[int]] = {}
        for k, (seq_dir, _fi) in enumerate(ds.index):
            lab = HUA2025.label_of(Path(seq_dir).name)
            if lab is not None and not isinstance(gtf(k), str):
                by_seq.setdefault(lab, []).append(k)
        frames = [(lab, k) for lab in HUA2025.reported for k in _spread_cap(by_seq.get(lab, []), args.max_frames)]
    print(f"[ceiling] {args.split}: {len(frames)} frames, window {args.window_ms:g} ms", flush=True)

    ious = {(sg, t): {} for sg in args.noise_px for t in THR}
    rng = np.random.default_rng(0)
    depth_cover, moving_share = [], []
    for n, (lab, k) in enumerate(frames):
        seq_dir, _fi = ds.index[k]
        gt, evs, fi, ts = gtf(k)
        x, y = evs[args.window_ms]
        meta = gtf.metas[Path(seq_dir)]
        g = gt_flow_for_sample(Path(seq_dir), fi, ds.window_s, ds.time_frames, meta)
        gs = gt_flow_for_sample(Path(seq_dir), fi, ds.window_s, ds.time_frames, meta, static=True)
        if g is None or gs is None:
            continue
        own = np.nan_to_num((g - gs)[:, y, x], nan=0.0)         # each event's own motion; no depth = static
        lbl = gt[y, x]
        depth_cover.append(float(np.isfinite(g[0, y, x]).mean()))
        moving_share.append(float(lbl.mean()))
        for sg in args.noise_px:
            cue = np.linalg.norm(own + (rng.normal(0.0, sg, own.shape) if sg else 0.0), axis=0)
            for t in THR:
                p = cue > t
                u = int((p | lbl).sum())
                ious[(sg, t)].setdefault(lab, []).append(int((p & lbl).sum()) / u if u else np.nan)
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(frames)}", flush=True)
    gtf.close()

    labs = sorted({lab for d in ious.values() for lab in d}, key=lambda s: list(HUA2025.reported).index(s)
                  if s in HUA2025.reported else -1)
    print(f"\nevents with depth: {100 * np.mean(depth_cover):.0f}%; moving events: {100 * np.mean(moving_share):.0f}%")
    for sg in args.noise_px:
        print(f"ORACLE cue |GT flow - GT static-world flow|{f' + flow error {sg:g} px' if sg else ''} "
              "thresholded (event IoU %, per-sequence mean):")
        print(f"{'threshold px':>13s}{'mean':>7s}" + "".join(f"{s:>8s}" for s in labs))
        best = (-1, None)
        for t in THR:
            ps = [100 * float(np.nanmean(ious[(sg, t)][s])) for s in labs]
            print(f"{t:13g}{np.mean(ps):7.1f}" + "".join(f"{v:8.1f}" for v in ps))
            best = max(best, (float(np.mean(ps)), t))
        print(f"  -> best {best[0]:.1f}% at {best[1]:g} px\n")
    if args.split == "benchmark":
        print(f"{'Hua et al.':>13s}{77.4:7.1f}" + "".join(f"{v:8.1f}" for v in HUA2025.reported.values()))


if __name__ == "__main__":
    main()
