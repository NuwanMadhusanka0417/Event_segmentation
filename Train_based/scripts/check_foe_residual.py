"""Pre-test: does the depth-free FOE residual separate movers better than the ego fit?
Training-free, on VALIDATION takes only -- never the test set.

Why: after the IMU removes the rotation, what is left of the background flow is
translation, which moves near static objects more than far ones (parallax). The ego
fit (quadratic) cannot represent depth, so it calls near static objects "moving".
The FOE test (hdems/foe.py) only asks whether a pixel moves ALONG its line through the
focus of expansion, which holds for a static point at ANY depth.

For each cue this reports, on the frames with a mover of the held-out takes:
  AUC     moving vs static at the scored pixels (working resolution), like the
          earlier pre-tests (residual 0.756 on scene9, 2026-10-13)
  IoU     the benchmark metric (event IoU, full resolution) of the cue thresholded at
          its best value on these frames = the training-free baseline's upper end
and the AUC split by camera TRANSLATION speed (Vicon, for the analysis only): the
FOE cue should help most where the camera translates.

Cues: residual (current pipeline: IMU + ego fit) | derotated |flow - rotation| | perp
| foe (perp + wrong-sign part) | min(residual, foe) | sqrt(residual * foe), and
res:<model> = the residual with another ego model on the SAME flow (--compare-ego,
e.g. planar8 = the 8-parameter plane flow of Hua et al. 2025).

    python scripts/check_foe_residual.py --config configs/evimo_seg.yaml \
        --val-sequences scene13_dyn_test_03,scene14_dyn_test_02 --resolution-ratio 2 \
        --max-frames 300 --device cuda --out results/foe_check.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdems.config import apply_resolution_ratio, restore_frontend  # noqa: E402
from hdems.data.build import HOLDOUT, build_dataset  # noqa: E402
from hdems.data.motion_labels import _pose  # noqa: E402
from hdems.eval import load_config  # noqa: E402
from hdems.baselines import THRESHOLD_GRID_PX  # noqa: E402
from hdems.foe import FoeParams, foe_cue_maps  # noqa: E402
from hdems.models.hdems import HDEMS  # noqa: E402
from hdems.models.motion import EGO_MODELS  # noqa: E402
from hdems.protocols import GTFrames, _spread_cap  # noqa: E402
from hdems.seg_features import motion_kwargs, score_pixel_mask  # noqa: E402

CUES = ("residual", "derotated", "perp", "foe", "min(res,foe)", "sqrt(res*foe)")
GRID = np.array(THRESHOLD_GRID_PX)      # the same thresholds as the --baseline benchmark runs
TRANS_BUCKETS = ((0.0, 0.1), (0.1, 0.3), (0.3, 10.0))      # camera translation, m/s


def camera_speed(meta: dict, fi: int) -> float:
    """|camera translation velocity| (m/s) at GT frame fi, central difference (Vicon)."""
    frames = meta["frames"]
    lo, hi = max(fi - 1, 0), min(fi + 1, len(frames) - 1)
    a, b = frames[lo], frames[hi]
    if not (isinstance(a, dict) and isinstance(b, dict) and "cam" in a and "cam" in b):
        return float("nan")
    dt = float(b["ts"]) - float(a["ts"])
    return float(np.linalg.norm(_pose(b["cam"])[2] - _pose(a["cam"])[2]) / dt) if dt > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--split", default=HOLDOUT, help="holdout (validation) or train -- never eval")
    ap.add_argument("--val-sequences", default="scene13_dyn_test_03,scene14_dyn_test_02",
                    help="held-out takes forming the holdout split ('' = dataset.val_* of the config)")
    ap.add_argument("--checkpoint", default=None, help="optional: rebuild ITS front end")
    ap.add_argument("--resolution-ratio", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--window-ms", type=float, default=12.5, help="event window of the IoU")
    ap.add_argument("--min-flow", type=float, default=FoeParams.min_flow_px,
                    help="sensor px / interval: flow vectors used to estimate the FOE")
    ap.add_argument("--inlier-px", type=float, default=FoeParams.inlier_px,
                    help="sensor px / interval: perpendicular residual of an FOE inlier")
    ap.add_argument("--min-moving-frac", type=float, default=FoeParams.min_moving_frac)
    ap.add_argument("--compare-ego", nargs="*", choices=list(EGO_MODELS), default=["planar8"],
                    help="also score the residual of these ego models (same flow, same rotation)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write the numbers to this .json")
    args = ap.parse_args()
    if args.split == "eval":
        raise SystemExit("never tune or select on the test (eval) split")

    torch.set_grad_enabled(False)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    if args.checkpoint:
        meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = restore_frontend(cfg, meta.get("frontend"))
    if args.val_sequences:
        cfg["dataset"] = {**cfg.get("dataset", {}), "val_split": HOLDOUT,
                          "val_sequences": [s for s in args.val_sequences.split(",") if s]}
    cfg = apply_resolution_ratio(cfg, args.resolution_ratio, verbose=True)
    cfg["segmentation"] = {**cfg.get("segmentation", {}), "head": "mfunet", "num_classes": 2}
    model = HDEMS(cfg).to(dev).eval()
    model.vsa_consistency = False                     # not a cue here: skip 3 encodes
    params = FoeParams(min_flow_px=args.min_flow, inlier_px=args.inlier_px,
                       min_moving_frac=args.min_moving_frac)
    r = model.res_ratio
    own_ego = model.ego_model
    other_ego = [e for e in dict.fromkeys(args.compare_ego) if e != own_ego]
    cues = list(CUES) + [f"res:{e}" for e in other_ego]

    ds = build_dataset(cfg, args.split)
    gtf = GTFrames(ds, [args.window_ms])
    keep = [k for k in range(len(ds)) if not isinstance(gtf(k), str)]
    ks = _spread_cap(keep, args.max_frames)
    print(f"[foe-check] split {args.split} ({args.val_sequences or 'config'}): {len(keep)} frames "
          f"with a mover, using {len(ks)}; ratio {r}; rotation {model.rot_source}; "
          f"ego model {model.ego_model}; FOE {params}", flush=True)

    px = {c: [] for c in cues}             # pixel-level, for the AUC
    px_y, px_frame = [], []
    ev = {c: [] for c in cues}             # per-frame event values, for the IoU
    ev_g = []
    frames = []
    for n, k in enumerate(ks):
        s = ds[k]
        surf = s["surface"].unsqueeze(0).to(dev)
        aux = motion_kwargs(s, dev)
        m = model.motion_features(surf, aux.get("rot_flow"))
        maps, foes = foe_cue_maps(m["flow"], m["rot"], model.ego_mask(surf), params, r)
        res = m["residual"].norm(dim=1, keepdim=True)
        cue = {"residual": res, "derotated": maps["derotated"], "perp": maps["perp"],
               "foe": maps["foe"], "min(res,foe)": torch.minimum(res, maps["foe"]),
               "sqrt(res*foe)": torch.sqrt(res * maps["foe"])}
        for e in other_ego:                        # the same flow + rotation, another ego model
            model.ego_model = e
            try:
                res_e, _ = model.background_fit(m["flow"], surf, m["rot"])
            finally:
                model.ego_model = own_ego
            cue[f"res:{e}"] = res_e.norm(dim=1, keepdim=True)
        cue = {c: (v * r)[0, 0] for c, v in cue.items()}                   # sensor px
        lab = s["mask"].to(dev)
        sc = score_pixel_mask(s, surf)[0] & (lab != 255)
        y = (lab[sc] == 1).cpu().numpy()
        for c in cues:
            px[c].append(cue[c][sc].cpu().numpy())
        px_y.append(y)
        px_frame.append(np.full(y.size, n))

        gt, evs, fi, _ts = gtf(k)
        Hf, Wf = gt.shape
        x_e, y_e = evs[args.window_ms]
        for c in cues:
            full = F.interpolate(cue[c][None, None], size=(Hf, Wf), mode="nearest")[0, 0]
            ev[c].append(full.cpu().numpy()[y_e, x_e])
        ev_g.append(gt[y_e, x_e])
        seq, _fi, _ = gtf.time_of(k)
        f = foes[0]
        rec = {"sequence": seq, "frame": int(fi), "cam_speed": camera_speed(gtf.metas[Path(ds.index[k][0])], fi),
               "foe_defined": f.e is not None, "foe_inliers": f.inliers, "moving_frac": f.moving_frac,
               "movers": int(y.sum()), "static": int((~y).sum())}
        if 0 < y.sum() < y.size:                         # per-frame AUC: where does a cue win?
            rec["auc"] = {c: float(roc_auc_score(y, px[c][-1])) for c in cues}
        frames.append(rec)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(ks)}", flush=True)
    gtf.close()

    y_all = np.concatenate(px_y)
    fr_all = np.concatenate(px_frame)
    speeds = np.array([f["cam_speed"] for f in frames])
    report = {"frames": len(ks), "params": dataclasses.asdict(params),
              "foe_defined": float(np.mean([f["foe_defined"] for f in frames])), "cues": {}}

    def event_iou(c: str, t: float) -> float:
        out = []
        for p, g in zip(ev[c], ev_g):
            pr = p > t
            u = int((pr | g).sum())
            out.append(int((pr & g).sum()) / u if u else np.nan)
        return float(np.nanmean(out))

    print(f"\nFOE defined (enough translation) on {report['foe_defined']:.0%} of the frames; "
          f"camera translation median {np.nanmedian(speeds):.2f} m/s")
    hdr = "".join(f"{f'AUC v{lo:g}-{hi:g}':>13s}" for lo, hi in TRANS_BUCKETS)
    print(f"\n{'cue':<15s}{'AUC':>7s}{hdr}{'best IoU':>10s}{'@ px':>7s}")
    for c in cues:
        x_all = np.concatenate(px[c])
        auc = roc_auc_score(y_all, x_all) if 0 < y_all.sum() < y_all.size else float("nan")
        buckets = []
        for lo, hi in TRANS_BUCKETS:
            fsel = np.flatnonzero((speeds >= lo) & (speeds < hi))
            sel = np.isin(fr_all, fsel)
            yb = y_all[sel]
            buckets.append(roc_auc_score(yb, x_all[sel]) if 0 < yb.sum() < yb.size else float("nan"))
        ious = {float(t): event_iou(c, t) for t in GRID}
        t_best = max(ious, key=lambda t: ious[t] if ious[t] == ious[t] else -1)
        report["cues"][c] = {"auc": auc, "auc_by_translation": dict(zip(map(str, TRANS_BUCKETS), buckets)),
                             "event_iou": ious, "best_threshold": t_best, "best_iou": ious[t_best]}
        print(f"{c:<15s}{auc:7.3f}" + "".join(f"{b:13.3f}" for b in buckets)
              + f"{100 * ious[t_best]:9.1f}%{t_best:7g}")
    # frames where the FOE was estimated: per-frame AUC (no pooling across frames)
    foe_fr = [f for f in frames if f["foe_defined"] and "auc" in f]
    if foe_fr:
        print(f"\nper-frame AUC (median) on the {len(foe_fr)} frames with LEFTOVER motion "
              f"(>= {params.min_moving_frac:.0%} of event px move >= {params.min_flow_px:g} px after "
              f"derotation -> FOE estimated):\n  "
              + "  ".join(f"{c} {np.median([f['auc'][c] for f in foe_fr]):.3f}" for c in cues))
    rest = [f for f in frames if not f["foe_defined"] and "auc" in f]
    if rest:
        print(f"per-frame AUC (median) on the {len(rest)} QUIET frames:\n  "
              + "  ".join(f"{c} {np.median([f['auc'][c] for f in rest]):.3f}" for c in cues))
    n_b = [int(np.sum((speeds >= lo) & (speeds < hi))) for lo, hi in TRANS_BUCKETS]
    print(f"(frames per translation bucket: {n_b}; IoU = event IoU {args.window_ms:g} ms at the "
          f"best threshold on THESE frames -- a validation number, not a test result)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        report["per_frame"] = frames
        Path(args.out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
        print(f"-> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
