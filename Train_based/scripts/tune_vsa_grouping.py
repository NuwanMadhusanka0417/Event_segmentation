#!/usr/bin/env python3
"""Tune the training-free VSA grouping (and the no-CNN threshold baseline) on the
VALIDATION split -- never on eval.

    python scripts/tune_vsa_grouping.py --config configs/evimo_seg.yaml \
        --split holdout --resolution-ratio 2 --max-frames 150 --device cuda \
        --out configs/vsa_grouping_tuned.yaml

For every candidate setting the frames are processed in index order (so the per-
sequence background prior follows time), scored like hdems.eval (score_pixel_mask,
ignore label excluded), and ranked by object mIoU (tie-break: FG IoU). The winner is
written as a ``vsa_grouping:`` YAML fragment that ``hdems.eval --vsa-group`` /
``--threshold-baseline`` merges automatically (or pass it with --vsa-params). The full
table is written next to it (``.txt``).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdems.config import apply_resolution_ratio, resolution_ratio_of   # noqa: E402
from hdems.data.build import build_dataset                            # noqa: E402
from hdems.data.motion_labels import IGNORE_LABEL                     # noqa: E402
from hdems.eval import load_config                                    # noqa: E402
from hdems.grouping import group_objects                              # noqa: E402
from hdems.grouping import params_from_config as grouping_params      # noqa: E402
from hdems.instances import binary_iou, hull_iou, instance_metrics    # noqa: E402
from hdems.models.hdems import HDEMS                                  # noqa: E402
from hdems.seg_features import event_pixel_mask, score_pixel_mask     # noqa: E402
from hdems.vsa_grouping import BackgroundPrior, vsa_group_objects     # noqa: E402
from hdems.vsa_grouping import params_from_config as vsa_params       # noqa: E402

FULL_GRID = {"sigma_s_px": [20.0, 30.0, 50.0], "sigma_v_px": [0.5, 1.0],
             "merge_vel_px": [0.5, 1.0], "merge_gap_px": ["0", "half"],
             "flow_smooth_px": [None, 35]}
QUICK_GRID = {"sigma_s_px": [30.0], "sigma_v_px": [0.5, 1.0],
              "merge_vel_px": [0.5, 1.0], "merge_gap_px": ["0", "half"],
              "flow_smooth_px": [None]}
THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5]                 # full-res px per interval


def _score(frames, predict) -> dict[str, float]:
    """predict(frame) -> objects map; frame scoring as in hdems.eval."""
    fg, inst, hull = [], [], []
    for f in frames:
        objects = predict(f)
        v = f["valid"]
        fg.append(binary_iou(objects > 0, f["gt_fg"], valid=v))
        s = instance_metrics(objects, f["gt_inst"], valid=v)
        if s["n_gt"] > 0:
            inst.append(s["instance_miou"])
            hull.append(hull_iou(objects, f["gt_inst"]))
    fin = [x for x in fg if x == x]
    return {"object_miou": float(np.nanmean(inst)) if inst else float("nan"),
            "fg_iou": float(np.mean(fin)) if fin else float("nan"),
            "hull_iou": float(np.nanmean(hull)) if hull else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--split", default="holdout", help="validation split (never eval)")
    ap.add_argument("--resolution-ratio", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=150, help="evenly spread; 0 = all")
    ap.add_argument("--quick", action="store_true", help="small grid (8 settings)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="configs/vsa_grouping_tuned.yaml")
    args = ap.parse_args()
    ev_split = load_config(args.config).get("dataset", {}).get("eval_split", "eval")
    if args.split == ev_split:
        raise SystemExit(f"refusing to tune on the eval split ({ev_split!r}): use the "
                         "held-out validation split (holdout)")

    base = load_config(args.config)
    grid = QUICK_GRID if args.quick else FULL_GRID
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # one front end per flow-smoothing option (the flow cache keys on it)
    models, cfgs = {}, {}
    for sm in grid["flow_smooth_px"]:
        c = copy.deepcopy(base)
        if sm:
            c["matching"] = {**c.get("matching", {}), "smooth": int(sm)}
        c = apply_resolution_ratio(c, args.resolution_ratio, verbose=sm == grid["flow_smooth_px"][0])
        c["segmentation"] = {**c.get("segmentation", {}), "head": "cnn", "num_classes": 2}
        cfgs[sm], models[sm] = c, HDEMS(c).to(device).eval()
    cfg0 = cfgs[grid["flow_smooth_px"][0]]
    r = resolution_ratio_of(cfg0)
    smooth_full = {sm: int(sm or base.get("matching", {}).get("smooth", 1))
                   for sm in grid["flow_smooth_px"]}

    ds = build_dataset(cfg0, args.split)
    idx = (np.arange(len(ds)) if not args.max_frames or args.max_frames >= len(ds)
           else np.unique(np.linspace(0, len(ds) - 1, args.max_frames).round().astype(int)))
    print(f"[tune] split={args.split}: {len(idx)} of {len(ds)} frames, ratio {r}, "
          f"{'quick' if args.quick else 'full'} grid")

    # per frame: labels + flow / residual for every smoothing option (computed once)
    frames = []
    t0 = time.time()
    with torch.no_grad():
        for k, i in enumerate(idx):
            s = ds[int(i)]
            x = s["surface"].unsqueeze(0).to(device)
            lbl = s["mask"].numpy()
            sc = score_pixel_mask(s, x)[0].cpu().numpy()
            f = {"seq": s.get("seq_id"), "fi": s.get("frame_index"),
                 "valid": sc & (lbl != IGNORE_LABEL), "score": sc, "gt_fg": lbl == 1,
                 "gt_inst": s["gt_instances"].numpy().astype(np.int64),
                 "ref": event_pixel_mask(x[:, :1])[0].cpu(), "flow": {}, "conf": {}}
            for sm, m in models.items():
                flow, conf = m.compute_motion(x)
                f["flow"][sm], f["conf"][sm] = flow[0].cpu(), conf[0, 0].cpu()
                if sm == grid["flow_smooth_px"][0]:
                    f["res"] = m.residual_from_flow(flow, x)[0][0].cpu().numpy()
            frames.append(f)
            if (k + 1) % 25 == 0:
                print(f"  flow {k + 1}/{len(idx)}  ({time.time() - t0:.0f} s)", flush=True)

    # --- VSA grouping grid ---------------------------------------------------
    rows = []
    keys = list(grid)
    for combo in itertools.product(*grid.values()):
        setting = dict(zip(keys, combo))
        sm = setting["flow_smooth_px"]
        gap = 0 if setting["merge_gap_px"] == "0" else smooth_full[sm] // 2
        vg = {**(base.get("vsa_grouping") or {}), **setting, "merge_gap_px": gap}
        p = vsa_params({**cfgs[sm], "vsa_grouping": vg})
        prior = BackgroundPrior()
        model = models[sm]
        tic = time.time()
        res = _score(frames, lambda f: vsa_group_objects(
            f["flow"][sm], f["ref"], model, p, conf=f["conf"][sm], prior=prior,
            seq_id=f["seq"], frame_index=f["fi"])[0])
        res["ms"] = (time.time() - tic) / len(frames) * 1000
        rows.append(({**setting, "merge_gap_px": gap}, res))
        print(f"  {setting} -> object mIoU {res['object_miou']:.4f}  FG IoU {res['fg_iou']:.4f}  "
              f"hull {res['hull_iou']:.4f}  ({res['ms']:.0f} ms/frame)", flush=True)

    # --- threshold baseline --------------------------------------------------
    gp = grouping_params(cfg0)
    base_rows = []
    for t in THRESHOLDS:
        def pred(f, t=t):
            mov = (np.hypot(f["res"][0], f["res"][1]) > t / r) & f["ref"].numpy()
            return group_objects(f["res"], mov & f["score"], gp)
        res = _score(frames, pred)
        base_rows.append((t, res))
        print(f"  baseline threshold {t} px -> object mIoU {res['object_miou']:.4f}  "
              f"FG IoU {res['fg_iou']:.4f}  hull {res['hull_iou']:.4f}", flush=True)

    rank = lambda m: (np.nan_to_num(m["object_miou"], nan=-1), np.nan_to_num(m["fg_iou"], nan=-1))
    best_setting, best = max(rows, key=lambda row: rank(row[1]))
    best_t, best_b = max(base_rows, key=lambda row: rank(row[1]))
    tuned = {k: v for k, v in best_setting.items()}
    tuned["baseline_threshold_px"] = best_t

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# tuned by scripts/tune_vsa_grouping.py on split={args.split} "
              f"({len(frames)} frames, ratio {r})\n"
              f"# VSA grouping: object mIoU {best['object_miou']:.4f}  FG IoU {best['fg_iou']:.4f}  "
              f"hull IoU {best['hull_iou']:.4f}\n"
              f"# threshold baseline: object mIoU {best_b['object_miou']:.4f}  "
              f"FG IoU {best_b['fg_iou']:.4f}  hull IoU {best_b['hull_iou']:.4f}\n")
    out.write_text(header + yaml.safe_dump({"vsa_grouping": tuned}, sort_keys=False), encoding="utf-8")
    table = [f"{s} | object mIoU {m['object_miou']:.4f} | FG IoU {m['fg_iou']:.4f} | "
             f"hull {m['hull_iou']:.4f} | {m['ms']:.0f} ms" for s, m in
             sorted(rows, key=lambda row: rank(row[1]), reverse=True)]
    table += [f"baseline threshold {t} | object mIoU {m['object_miou']:.4f} | "
              f"FG IoU {m['fg_iou']:.4f} | hull {m['hull_iou']:.4f}" for t, m in base_rows]
    out.with_suffix(".txt").write_text(header + "\n".join(table) + "\n", encoding="utf-8")
    print(f"\n[tune] best VSA grouping: {best_setting}\n        -> {best}")
    print(f"[tune] best threshold: {best_t} px -> {best_b}")
    print(f"[tune] wrote {out} and {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
