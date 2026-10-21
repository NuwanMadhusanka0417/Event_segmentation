"""Graph-cut motion segmentation (hdems/graphcut_seg.py) on the Hua et al. benchmark, with the
VSA flow and/or VecKM_flow as input. No training: the smoothness lam and the decision
threshold delta are chosen on the VALIDATION takes, then the 5 test sequences are scored
(event IoU at 12.5 ms, per-sequence mean, like hdems/protocols.py).

Also reported for each flow: the same flow's cue thresholded per event (no regions) -- so
the gain of the region-level decision is measured directly.

    python scripts/eval_graphcut.py --flow vsa --max-frames 60 --device cpu          # quick
    python scripts/eval_graphcut.py --flow vsa veckm --veckm-dir <repo> --benchmark --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdems.baselines import THRESHOLD_GRID_PX  # noqa: E402
from hdems.config import apply_resolution_ratio  # noqa: E402
from hdems.data.build import HOLDOUT, build_dataset  # noqa: E402
from hdems.eval import load_config  # noqa: E402
from hdems.graphcut_seg import GCParams, segment  # noqa: E402
from hdems.models.hdems import HDEMS  # noqa: E402
from hdems.protocols import HUA2025, GTFrames, _spread_cap  # noqa: E402
from hdems.seg_features import motion_kwargs  # noqa: E402
from hdems.veckm import INTERVAL_S, load_veckm, undistort_radtan, veckm_normal_flow, veckm_slice  # noqa: E402

DELTA_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)


def frame_inputs(k, ds, gtf, model, dev, flows, est, ensemble):
    """Events of the benchmark window of frame k -> per flow source the graph inputs.
    Returns (labels per event (n,), node of each event (n,), node x, node y, H, W,
    {flow: (row_node, D, b, cue per event)})."""
    r = model.res_ratio
    s = ds[k]
    seq_dir, _fi = ds.index[k]
    gt, _evs, fi, ts = gtf(k)
    H, W = gt.shape
    meta = gtf.metas[Path(seq_dir)]
    mm = meta["meta"]
    stream = gtf.streams[Path(seq_dir)]
    i1 = int(np.searchsorted(stream.t, ts, side="right"))
    i0 = int(np.searchsorted(stream.t, ts - INTERVAL_S, side="left"))
    xy = np.asarray(stream.xy[i0:i1]).astype(np.int64).reshape(-1, 2)
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
    xe, ye = xy[ok, 0], xy[ok, 1]
    labels = gt[ye, xe]
    pix = ye * W + xe
    upix, node = np.unique(pix, return_inverse=True)
    nx, ny = upix % W, upix // W

    surf = s["surface"].unsqueeze(0).to(dev)
    flow, _conf = model.compute_motion(surf)
    rot = model._rot(motion_kwargs(s, dev).get("rot_flow"), flow)

    def up(a):
        return F.interpolate(a.float(), size=(H, W), mode="nearest")[0].cpu().numpy() * r

    rot_f = up(rot) if rot is not None else np.zeros((2, H, W))
    out = {}
    if "vsa" in flows:
        u = (up(flow) - rot_f)[:, ny, nx].T                                  # (N, 2) derotated
        n_nodes = len(nx)
        out["vsa"] = (np.repeat(np.arange(n_nodes), 2), np.tile(np.eye(2), (n_nodes, 1)), u.reshape(-1),
                      np.linalg.norm(u, axis=1)[node])
    if "veckm" in flows:
        t_sl, xy_sl = veckm_slice(stream, ts, 2.0 * est.P.t_radius, H, W)
        xu, yu = undistort_radtan(xy_sl[:, 0], xy_sl[:, 1], mm)
        nf, _unc = veckm_normal_flow(est, t_sl, xu, yu, ensemble)
        sc = t_sl >= ts - INTERVAL_S
        f = 0.5 * (float(mm["fx"]) + float(mm["fy"]))
        nvec = -nf[sc] * f * INTERVAL_S                                      # px / interval, EARLIER
        x2, y2 = xy_sl[sc, 0].astype(np.int64), xy_sl[sc, 1].astype(np.int64)
        spd = np.linalg.norm(nvec, axis=1)
        n0 = nvec / np.maximum(spd[:, None], 1e-9)
        bvals = spd - (rot_f[:, y2, x2].T * n0).sum(1)                       # derotated normal speed
        # the VecKM slice holds the same events as the window: map them to the nodes
        node2 = np.searchsorted(upix, y2 * W + x2)
        node2 = np.clip(node2, 0, len(upix) - 1)
        okn = upix[node2] == y2 * W + x2
        cue = np.zeros(len(upix))
        np.maximum.at(cue, node2[okn], np.abs(bvals[okn]))
        out["veckm"] = (node2[okn], n0[okn], bvals[okn], cue[node])
    return labels, node, nx, ny, H, W, out


def frame_iou(p: np.ndarray, g: np.ndarray) -> float:
    u = int((p | g).sum())
    return int((p & g).sum()) / u if u else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--flow", nargs="+", choices=["vsa", "veckm"], default=["vsa"])
    ap.add_argument("--veckm-dir", default=None)
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--sigma-px", type=float, default=GCParams.sigma_px)
    ap.add_argument("--n-init", type=int, default=GCParams.n_init)
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--val-sequences", default="scene13_dyn_test_03,scene14_dyn_test_02")
    ap.add_argument("--resolution-ratio", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=300, help="validation frames with a mover")
    ap.add_argument("--benchmark", action="store_true", help="also score the 5 test sequences")
    ap.add_argument("--max-per-sequence", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if "veckm" in args.flow and not args.veckm_dir:
        raise SystemExit("--flow veckm needs --veckm-dir")

    torch.set_grad_enabled(False)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    est = load_veckm(Path(args.veckm_dir)) if "veckm" in args.flow else None
    cfg = load_config(args.config)
    cfg["dataset"] = {**cfg.get("dataset", {}), "val_split": HOLDOUT,
                      "val_sequences": [s for s in args.val_sequences.split(",") if s]}
    cfg = apply_resolution_ratio(cfg, args.resolution_ratio, verbose=True)
    cfg["segmentation"] = {**cfg.get("segmentation", {}), "head": "mfunet", "num_classes": 2}
    model = HDEMS(cfg).to(dev).eval()
    base = GCParams(sigma_px=args.sigma_px, n_init=args.n_init)

    def run(ds, gtf, ks, lams, tag):
        """-> {flow: {"cue": [per-frame event cue], ("gc", lam): [per-frame motion score]}}, labels."""
        res = {fl: {"cue": []} | {("gc", lam): [] for lam in lams} for fl in args.flow}
        labs, secs = [], {fl: 0.0 for fl in args.flow}
        for n, k in enumerate(ks):
            labels, node, nx, ny, H, W, inp = frame_inputs(k, ds, gtf, model, dev, args.flow, est, args.ensemble)
            labs.append(labels)
            for fl in args.flow:
                rn, D, b, cue = inp[fl]
                res[fl]["cue"].append(cue)
                for lam in lams:
                    t0 = time.perf_counter()
                    o = segment(nx, ny, rn, D, b, H, W, replace(base, lam=lam))
                    secs[fl] += time.perf_counter() - t0
                    res[fl][("gc", lam)].append(o["motion_score"][node])
            if (n + 1) % 20 == 0 or n == 0:
                print(f"  {tag} {n + 1}/{len(ks)}  graph cut " + ", ".join(
                    f"{fl} {secs[fl] / ((n + 1) * len(lams)):.2f} s/frame" for fl in args.flow), flush=True)
        return res, labs

    # ---- validation: choose lam and delta (graph cut) and the cue threshold
    ds = build_dataset(cfg, HOLDOUT)
    gtf = GTFrames(ds, [12.5])
    keep = [k for k in range(len(ds)) if not isinstance(gtf(k), str)]
    ks = _spread_cap(keep, args.max_frames)
    print(f"[graphcut] validation {args.val_sequences}: {len(ks)} frames; flows {args.flow}; lams {args.lams}", flush=True)
    res, labs = run(ds, gtf, ks, args.lams, "val")
    gtf.close()

    report = {"validation": {}, "choice": {}}
    print(f"\nVALIDATION (event IoU %, frames with a mover, mean over frames)")
    for fl in args.flow:
        cue_iou = {t: float(np.nanmean([frame_iou(c > t, g) for c, g in zip(res[fl]["cue"], labs)]))
                   for t in THRESHOLD_GRID_PX}
        tc = max(cue_iou, key=lambda q: cue_iou[q] if cue_iou[q] == cue_iou[q] else -1)
        best = (-1.0, None, None)
        for lam in args.lams:
            for dl in DELTA_GRID:
                v = float(np.nanmean([frame_iou(m > dl, g) for m, g in zip(res[fl][("gc", lam)], labs)]))
                report["validation"][f"{fl} gc lam={lam:g} delta={dl:g}"] = v
                if v > best[0]:
                    best = (v, lam, dl)
        report["choice"][fl] = {"cue_threshold": tc, "cue_val_iou": cue_iou[tc],
                                "lam": best[1], "delta": best[2], "gc_val_iou": best[0]}
        print(f"  {fl:6s} cue (threshold {tc:g} px): {100 * cue_iou[tc]:.1f}   graph cut (lam {best[1]:g}, "
              f"delta {best[2]:g} px): {100 * best[0]:.1f}")
        for lam in args.lams:
            row = [report["validation"][f"{fl} gc lam={lam:g} delta={dl:g}"] for dl in DELTA_GRID]
            print(f"     lam {lam:<4g} " + " ".join(f"{100 * v:5.1f}" for v in row) + "   (delta "
                  + " ".join(f"{d:g}" for d in DELTA_GRID) + ")")

    # ---- benchmark: the 5 test sequences with the validation choice
    if args.benchmark:
        dse = build_dataset(cfg, cfg.get("dataset", {}).get("eval_split", "eval"))
        gte = GTFrames(dse, [12.5])
        by_seq: dict[str, list[int]] = {}
        for k, (seq_dir, _fi) in enumerate(dse.index):
            lab = HUA2025.label_of(Path(seq_dir).name)
            if lab is not None and not isinstance(gte(k), str):
                by_seq.setdefault(lab, []).append(k)
        frames = [(lab, k) for lab in HUA2025.reported for k in _spread_cap(by_seq.get(lab, []), args.max_per_sequence)]
        print(f"\n[graphcut] BENCHMARK: {len(frames)} test frames with a mover", flush=True)
        per = {}
        for fl in args.flow:
            ch = report["choice"][fl]
            per[(fl, "cue")] = {}
            per[(fl, "gc")] = {}
        lam_by_flow = {fl: report["choice"][fl]["lam"] for fl in args.flow}
        for n, (lab, k) in enumerate(frames):
            labels, node, nx, ny, H, W, inp = frame_inputs(k, dse, gte, model, dev, args.flow, est, args.ensemble)
            for fl in args.flow:
                rn, D, b, cue = inp[fl]
                ch = report["choice"][fl]
                o = segment(nx, ny, rn, D, b, H, W, replace(base, lam=lam_by_flow[fl]))
                per[(fl, "gc")].setdefault(lab, []).append(frame_iou(o["motion_score"][node] > ch["delta"], labels))
                per[(fl, "cue")].setdefault(lab, []).append(frame_iou(cue > ch["cue_threshold"], labels))
            if (n + 1) % 50 == 0:
                print(f"  test {n + 1}/{len(frames)}", flush=True)
        gte.close()
        seqs = list(HUA2025.reported)
        print("\nBENCHMARK (event IoU % at 12.5 ms, per-sequence mean; settings from validation)")
        print(f"{'method':<26s}{'mean':>7s}" + "".join(f"{s:>8s}" for s in seqs))
        report["benchmark"] = {}
        for (fl, kind), d in per.items():
            ps = {s: 100 * float(np.nanmean(d[s])) for s in seqs if d.get(s)}
            mean = float(np.mean(list(ps.values()))) if ps else float("nan")
            name = f"{fl} {'graph cut' if kind == 'gc' else 'cue threshold'}"
            report["benchmark"][name] = {"mean": mean, "per_sequence": ps}
            print(f"{name:<26s}{mean:7.2f}" + "".join(f"{ps.get(s, float('nan')):8.1f}" for s in seqs))
        print(f"{'Hua et al. 2025':<26s}{77.40:7.2f}" + "".join(f"{v:8.1f}" for v in HUA2025.reported.values()))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
