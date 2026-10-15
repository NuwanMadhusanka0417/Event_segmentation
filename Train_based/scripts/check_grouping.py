"""How well does the object grouping (hdems/grouping.py) split moving pixels into objects?
VALIDATION takes only -- grouping settings are chosen here, never on the test set.

Which pixels are grouped (--fg):
  gt    the GROUND-TRUTH moving event pixels: a perfect classifier, so only the grouping
        is scored
  cue   the training-free motion cue |residual| > --cue-px (sensor px): a noisy stand-in
        for a classifier, with many false positives -- a robustness test
  head  the trained head's moving pixels (--checkpoint, at its tuned threshold): what
        eval / the figures actually group

Reported per method, over frames with a moving object:
  obj mIoU   matched instance IoU (hdems.instances.instance_metrics, unmatched = 0)
  pred/gt    groups per frame vs objects per frame
  pieces     groups covering >= 10% of a GT object, averaged over objects (1 = whole)
  split %    GT objects broken into >= 2 such pieces
  merged %   groups that cover >= 10% of two or more GT objects

    python scripts/check_grouping.py --config configs/evimo_seg.yaml --fg gt \\
        --val-sequences scene13_dyn_test_03,scene14_dyn_test_02 --max-frames 300 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdems.config import apply_resolution_ratio, restore_frontend  # noqa: E402
from hdems.data.build import HOLDOUT, build_dataset  # noqa: E402
from hdems.eval import load_config  # noqa: E402
from hdems.grouping import group_objects, params_from_config  # noqa: E402
from hdems.instances import instance_metrics  # noqa: E402
from hdems.models.hdems import HDEMS  # noqa: E402
from hdems.protocols import GTFrames, _spread_cap, forward_sample, moving_prob  # noqa: E402
from hdems.seg_features import motion_kwargs, score_pixel_mask  # noqa: E402


def fragmentation(pred: np.ndarray, gt: np.ndarray, frac: float = 0.1) -> tuple[list[int], list[int]]:
    """(pieces per GT object, GT objects per predicted group), counting overlaps >= frac."""
    gts = [(g, gt == g) for g in (int(i) for i in np.unique(gt) if i > 0)]
    pieces = []
    for _g, gm in gts:
        ids, cnt = np.unique(pred[gm], return_counts=True)
        pieces.append(int(((ids > 0) & (cnt >= frac * gm.sum())).sum()))
    merged = []
    for p in (int(i) for i in np.unique(pred) if i > 0):
        pm = pred == p
        merged.append(sum(int((pm & gm).sum() >= frac * gm.sum()) for _g, gm in gts))
    return pieces, merged


def method_table(base, r: int, args) -> dict:
    """name -> GroupingParams; split settings are in FULL-resolution px."""
    m = {f"CONFIG ({base.method}, link {base.adjacency_px * r})": base,
         "motion first (old)": replace(base, method="motion")}
    for link in args.links:
        for mm in args.min_motions:
            m[f"spatial, link {link}, filter {mm:g}"] = replace(
                base, method="spatial", split_depth=0, adjacency_px=max(1, link // r), min_motion=mm / r)
    for diff in args.split_diffs:
        for gain in args.split_gains:
            m[f"spatial+split d{diff:g} g{gain:g}"] = replace(
                base, method="spatial", split_min_diff=diff / r, split_gain=gain,
                adjacency_px=max(1, args.split_link // r), min_motion=0.0)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--val-sequences", default="scene13_dyn_test_03,scene14_dyn_test_02")
    ap.add_argument("--fg", choices=["gt", "cue", "head"], default="gt")
    ap.add_argument("--cue-px", type=float, default=1.25,
                    help="--fg cue: |residual| threshold, sensor px (the validation-tuned 1.25)")
    ap.add_argument("--checkpoint", default=None,
                    help="rebuild ITS front end; with --fg head also its trained head")
    ap.add_argument("--resolution-ratio", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--links", type=int, nargs="+", default=[2, 4, 6],
                    help="adjacency_px values for spatial grouping (FULL-resolution px)")
    ap.add_argument("--min-motions", type=float, nargs="+", default=[0.0, 0.25],
                    help="min_motion_px values: drop groups slower than this (FULL-resolution px)")
    ap.add_argument("--split-diffs", type=float, nargs="*", default=[2.0],
                    help="spatial+split: split_min_diff_px values (FULL-resolution px / interval)")
    ap.add_argument("--split-gains", type=float, nargs="+", default=[0.5])
    ap.add_argument("--split-link", type=int, default=4, help="adjacency_px of the split variants")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.fg == "head" and not args.checkpoint:
        raise SystemExit("--fg head needs --checkpoint")

    torch.set_grad_enabled(False)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    ck = None
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = restore_frontend(cfg, ck.get("frontend"))
        if args.fg == "head":                       # rebuild the trained head as eval does
            cfg["velocity"] = {**cfg.get("velocity", {}),
                               **{k: ck[k] for k in ("axis_combine", "event_combine") if ck.get(k)},
                               "event_feature": ck.get("event_feature", "phi")}
            cfg["segmentation"] = {**cfg.get("segmentation", {}), **(ck.get("head_config") or {}),
                                   "head": ck.get("head", "mfunet")}
    cfg["dataset"] = {**cfg.get("dataset", {}), "val_split": HOLDOUT,
                      "val_sequences": [s for s in args.val_sequences.split(",") if s]}
    cfg = apply_resolution_ratio(cfg, args.resolution_ratio if ck is None else None, verbose=True)
    cfg["segmentation"] = {**cfg.get("segmentation", {}), "num_classes": 2}
    if args.fg != "head":
        cfg["segmentation"]["head"] = "mfunet"
    model = HDEMS(cfg).to(dev).eval()
    thr = 0.5
    if args.fg == "head":
        model.load_state_dict(ck["model"], strict=False)
        thr = float(ck.get("decision_threshold") or 0.5)
        model.decision_threshold = thr
    r = model.res_ratio
    base = params_from_config(cfg)
    methods = method_table(base, r, args)

    ds = build_dataset(cfg, HOLDOUT)
    gtf = GTFrames(ds, [12.5])
    keep = [k for k in range(len(ds)) if not isinstance(gtf(k), str)]
    ks = _spread_cap(keep, args.max_frames)
    gtf.close()
    print(f"[grouping-check] validation {args.val_sequences}: {len(ks)} frames with a mover; ratio {r}; "
          f"pixels grouped: {args.fg}" + (f" (P > {thr:.2f})" if args.fg == "head" else "")
          + (f" (|residual| > {args.cue_px:g} px)" if args.fg == "cue" else ""), flush=True)

    acc = {n: {"miou": [], "n_pred": [], "n_gt": [], "pieces": [], "merged": []} for n in methods}
    fg_iou = []
    for i, k in enumerate(ks):
        s = ds[k]
        surf = s["surface"].unsqueeze(0).to(dev)
        ev = score_pixel_mask(s, surf)[0].cpu().numpy()
        gt = s["gt_instances"].numpy().astype(np.int64) * ev
        if args.fg == "head":
            out, _surf, _aux = forward_sample(model, s, dev)
            res = out["residual"][0].cpu().numpy()
            mask = (moving_prob(out) > thr).cpu().numpy() & ev
        else:
            flow, _conf = model.compute_motion(surf)
            res_t, _bg = model.background_fit(flow, surf, motion_kwargs(s, dev).get("rot_flow"))
            res = res_t[0].cpu().numpy()
            mask = (gt > 0) if args.fg == "gt" else (np.hypot(res[0], res[1]) * r > args.cue_px) & ev
        if not (gt > 0).any():
            continue
        u = (mask | (gt > 0)).sum()
        fg_iou.append((mask & (gt > 0)).sum() / u if u else np.nan)
        for n, p in methods.items():
            pred = group_objects(res, mask, p)
            m = instance_metrics(pred, gt, valid=ev)
            pieces, merged = fragmentation(pred, gt)
            a = acc[n]
            a["miou"].append(m["instance_miou"])
            a["n_pred"].append(m["n_pred"])
            a["n_gt"].append(m["n_gt"])
            a["pieces"] += pieces
            a["merged"] += merged
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(ks)}", flush=True)

    print(f"\nmoving / static pixel IoU of the grouped pixels: {np.nanmean(fg_iou):.3f} "
          f"(1 = ground truth)")
    print(f"{'method':<30s}{'obj mIoU':>9s}{'pred/gt':>12s}{'pieces':>8s}{'split %':>9s}{'merged %':>10s}")
    report = {"fg": args.fg, "frames": len(acc[next(iter(acc))]["miou"]), "methods": {}}
    for n, a in acc.items():
        pieces, merged = np.array(a["pieces"]), np.array(a["merged"])
        row = {"obj_miou": float(np.nanmean(a["miou"])), "pred_per_frame": float(np.mean(a["n_pred"])),
               "gt_per_frame": float(np.mean(a["n_gt"])),
               "pieces_per_object": float(pieces.mean()) if pieces.size else float("nan"),
               "split_pct": 100.0 * float((pieces >= 2).mean()) if pieces.size else float("nan"),
               "merged_pct": 100.0 * float((merged >= 2).mean()) if merged.size else float("nan")}
        report["methods"][n] = row
        print(f"{n:<30s}{row['obj_miou']:9.3f}{row['pred_per_frame']:6.2f}/{row['gt_per_frame']:<5.2f}"
              f"{row['pieces_per_object']:8.2f}{row['split_pct']:8.0f}%{row['merged_pct']:9.0f}%")
    print(f"({report['frames']} frames with a moving object; split settings in full-resolution px)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
