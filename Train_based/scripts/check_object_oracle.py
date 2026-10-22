"""Which KIND of method can reach Hua et al.'s 77.4% under our benchmark? Training-free oracles
on the 5 test sequences (event IoU at 12.5 ms, frames with a mover, per-sequence mean).

scripts/check_protocol_ceiling.py thresholds the perfect motion of each EVENT (79.3% mean,
51.5% on 14-05). That bounds per-event methods only. The ground truth marks a whole OBJECT
as moving from its pose speed over ~167 ms (motion_label.speed_span_frames), so a method
that decides per object, and remembers recent motion, can score higher. Oracles here:

  event          per-event GT own motion (GT flow - GT static-world flow) > thr
  object         GT instances; an object moves if the median own motion of its events > thr
                 (perfect grouping, the motion decision still from the 12.5 ms window)
  object+memK    as object, but moving if it moved in ANY of the last K frames (temporal memory)

Run per sequence in parallel (all frames of the sequence, in time order):

    python scripts/check_object_oracle.py --workers 5
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdems.config import apply_resolution_ratio  # noqa: E402
from hdems.data.build import HOLDOUT, build_dataset  # noqa: E402
from hdems.data.gt_flow import gt_flow_for_sample  # noqa: E402
from hdems.eval import load_config  # noqa: E402
from hdems.protocols import HUA2025, GTFrames, gt_key  # noqa: E402

THR = (0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0)
MEMS = (3, 6, 12, 24)


def iou(p: np.ndarray, g: np.ndarray) -> float:
    u = int((p | g).sum())
    return int((p & g).sum()) / u if u else float("nan")


def build(config: str):
    cfg = load_config(config)
    cfg["dataset"] = {**cfg.get("dataset", {}), "val_split": HOLDOUT}
    cfg = apply_resolution_ratio(cfg, 2)
    return build_dataset(cfg, cfg["dataset"].get("eval_split", "eval"))


def run_sequence(job: tuple[str, str, list[int]]) -> tuple[str, dict]:
    """All frames of one sequence in time order -> {method: {thr: [frame IoUs]}}."""
    config, label, ks = job
    ds = build(config)
    gtf = GTFrames(ds, [12.5])
    res = {"event": {t: [] for t in THR}, "object": {t: [] for t in THR}}
    res |= {f"object+mem{m}": {t: [] for t in THR} for m in MEMS}
    hist: list[dict[int, float]] = []              # per frame: object id -> median own speed
    for k in ks:
        seq_dir, _fi = ds.index[k]
        seq_dir = Path(seq_dir)
        r = gtf(k)
        name, fi, ts = gtf.time_of(k)
        meta = gtf.metas[seq_dir]
        obj_map = gtf.masks[seq_dir][gt_key("mask", fi)] // 1000
        H, W = obj_map.shape
        x, y = gtf.streams[seq_dir].window(ts - 0.0125, ts, H, W)
        g = gt_flow_for_sample(seq_dir, fi, ds.window_s, ds.time_frames, meta)
        gs = gt_flow_for_sample(seq_dir, fi, ds.window_s, ds.time_frames, meta, static=True)
        speeds: dict[int, float] = {}
        own = None
        if g is not None and gs is not None:
            d = (g - gs)[:, y, x]
            own = np.linalg.norm(np.nan_to_num(d, nan=0.0), axis=0)
            ids = obj_map[y, x]
            fin = np.isfinite(d[0])
            for i in np.unique(ids[ids > 0]):
                sel = (ids == i) & fin
                if sel.sum() >= 5:
                    speeds[int(i)] = float(np.median(own[sel]))
        hist.append(speeds)
        if isinstance(r, str) or own is None:           # scored only on frames with a mover
            continue
        gt, _evs, _fi, _ts = r
        lbl = gt[y, x]
        ids = obj_map[y, x]
        for t in THR:
            res["event"][t].append(iou(own > t, lbl))
            mov = {i for i, s in speeds.items() if s > t}
            res["object"][t].append(iou(np.isin(ids, list(mov)) & (ids > 0), lbl))
            for m in MEMS:
                movm = {i for h in hist[-m:] for i, s in h.items() if s > t}
                res[f"object+mem{m}"][t].append(iou(np.isin(ids, list(movm)) & (ids > 0), lbl))
    gtf.close()
    return label, res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    ds = build(args.config)
    by_seq: dict[str, list[int]] = {}
    for k, (seq_dir, _fi) in enumerate(ds.index):
        lab = HUA2025.label_of(Path(seq_dir).name)
        if lab is not None:
            by_seq.setdefault(lab, []).append(k)
    jobs = [(args.config, lab, sorted(by_seq[lab], key=lambda k: ds.index[k][1])) for lab in HUA2025.reported
            if lab in by_seq]
    print(f"[object oracle] {', '.join(f'{lab}: {len(ks)} frames' for _c, lab, ks in jobs)}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        out = dict(pool.map(run_sequence, jobs))
    seqs = [lab for lab in HUA2025.reported if lab in out]
    print(f"\n{'oracle':<16s}{'thr':>6s}{'mean':>7s}" + "".join(f"{s:>8s}" for s in seqs))
    for meth in out[seqs[0]]:
        best = None
        for t in THR:
            ps = [100 * float(np.nanmean(out[s][meth][t])) for s in seqs]
            if best is None or np.mean(ps) > best[0]:
                best = (float(np.mean(ps)), t, ps)
        print(f"{meth:<16s}{best[1]:6g}{best[0]:7.1f}" + "".join(f"{v:8.1f}" for v in best[2]))
    print(f"{'Hua et al.':<16s}{'':6s}{77.4:7.1f}" + "".join(f"{v:8.1f}" for v in HUA2025.reported.values()))


if __name__ == "__main__":
    main()
