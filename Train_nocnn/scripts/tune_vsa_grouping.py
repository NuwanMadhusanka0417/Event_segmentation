#!/usr/bin/env python3
"""Tune the training-free back ends on a VALIDATION split -- never on eval:

  --moving-test         IMU rotation + static-scene test (moving_test: model, threshold,
                        min_rot) -- hdems/static_test.py
  --vsa-group           VSA grouping (vsa_grouping: sigma_s/v, merge_vel, merge_gap)
  --threshold-baseline  |affine ego residual| > threshold (vsa_grouping.baseline_threshold_px)

    python scripts/tune_vsa_grouping.py --config configs/evimo_seg.yaml \
        --resolution-ratio 2 --max-frames 150 --device cuda \
        --out configs/vsa_grouping_tuned.yaml

Validation frames: the TRAIN-split sequences of moving_test.tune_scenes (the only ones
with IMU besides eval: scene9/10/12 were recorded without it), else dataset.val_scenes;
--val-scenes overrides. Frames are processed in index order (the VSA background prior
follows time), scored like hdems.eval (score_pixel_mask, ignore label excluded).

Ranking: FG IoU (moving vs static over the scored events -- the benchmark's quantity)
by default. Object mIoU (--rank object) averages over the GROUND-TRUTH objects only, so
extra predicted objects cost nothing: the old object ranking picked a setting that split
the static scene into ~10-30 "objects" (FG IoU 0.05). The winners are written as
``vsa_grouping:`` + ``moving_test:`` YAML that hdems.eval merges (--vsa-params), with the
full table next to it (``.txt``).
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
from hdems.static_test import MODELS, derotate, moving_score          # noqa: E402
from hdems.static_test import params_from_config as static_params     # noqa: E402
from hdems.vsa_grouping import BackgroundPrior, vsa_group_objects     # noqa: E402
from hdems.vsa_grouping import params_from_config as vsa_params       # noqa: E402

FULL_GRID = {"sigma_s_px": [20.0, 30.0, 50.0], "sigma_v_px": [0.5, 1.0],
             "merge_vel_px": [0.5, 1.0], "merge_gap_px": ["0", "half"],
             "flow_smooth_px": [None, 35]}
QUICK_GRID = {"sigma_s_px": [30.0], "sigma_v_px": [0.5, 1.0],
              "merge_vel_px": [0.5, 1.0], "merge_gap_px": ["0", "half"],
              "flow_smooth_px": [None]}
THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5]                 # full-res px per interval
# static-scene test: every model x rotation gate x switch-on level is scored once,
# thresholds are cheap. "none" (= |derotated flow|) must stay in: on the 50 eval figure
# frames it beat every background model (FG IoU 0.372 vs epipolar 0.273), which only
# helped when the camera turned at 15-45 deg/s. static_px is where a model switches on.
MOVING_FULL = {"model": list(MODELS), "min_rot_px": [0.0, 0.5, 1.0],
               "static_px": [0.5, 1.0, 2.0, 3.0]}
MOVING_QUICK = {"model": ["none", "epipolar", "quadratic"], "min_rot_px": [0.0, 0.5],
                "static_px": [1.0, 2.0]}
MOVING_THRESHOLDS = [0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]   # full-res px


def _score(frames, predict, with_objects: bool = True) -> dict[str, float]:
    """predict(frame) -> objects map; frame scoring as in hdems.eval."""
    fg, inst, hull, npred = [], [], [], []
    for f in frames:
        objects = predict(f)
        v = f["valid"]
        fg.append(binary_iou(objects > 0, f["gt_fg"], valid=v))
        if not with_objects:
            continue
        s = instance_metrics(objects, f["gt_inst"], valid=v)
        if s["n_gt"] > 0:
            inst.append(s["instance_miou"])
            hull.append(hull_iou(objects, f["gt_inst"]))
            npred.append(s["n_pred"])
    fin = [x for x in fg if x == x]
    return {"object_miou": float(np.nanmean(inst)) if inst else float("nan"),
            "fg_iou": float(np.mean(fin)) if fin else float("nan"),
            "hull_iou": float(np.nanmean(hull)) if hull else float("nan"),
            "objects": float(np.mean(npred)) if npred else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--split", default="holdout", help="validation split (never eval)")
    ap.add_argument("--val-scenes", default=None,
                    help="comma list of TRAIN-split scenes to tune on (default: "
                         "moving_test.tune_scenes, else dataset.val_scenes)")
    ap.add_argument("--resolution-ratio", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=150, help="evenly spread; 0 = all")
    ap.add_argument("--quick", action="store_true", help="small grids")
    ap.add_argument("--rank", choices=["fg", "object"], default="fg",
                    help="fg = FG IoU (default) | object = object mIoU (ignores false objects)")
    ap.add_argument("--skip-vsa", action="store_true", help="tune only the static-scene test")
    ap.add_argument("--min-gain", type=float, default=0.02,
                    help="a background model (epipolar/quadratic/affine) is chosen only if it "
                         "beats model 'none' by this much FG IoU on validation")
    ap.add_argument("--rot-source", choices=["imu", "vicon", "none"], default=None,
                    help="camera rotation (overrides motion.rot_source; vicon = ORACLE)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="configs/vsa_grouping_tuned.yaml")
    args = ap.parse_args()

    base = load_config(args.config)
    if args.rot_source:
        base["motion"] = {**(base.get("motion") or {}), "rot_source": args.rot_source}
    ev_split = base.get("dataset", {}).get("eval_split", "eval")
    if args.split == ev_split:
        raise SystemExit(f"refusing to tune on the eval split ({ev_split!r}): use the "
                         "held-out validation split (holdout)")
    scenes = (args.val_scenes.split(",") if args.val_scenes
              else (base.get("moving_test") or {}).get("tune_scenes")
              or base.get("dataset", {}).get("val_scenes"))
    base["dataset"] = {**base.get("dataset", {}), "val_scenes": list(scenes)}
    grid = QUICK_GRID if args.quick else FULL_GRID
    mgrid = MOVING_QUICK if args.quick else MOVING_FULL
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rank = ((lambda m: (np.nan_to_num(m["fg_iou"], nan=-1), np.nan_to_num(m["object_miou"], nan=-1)))
            if args.rank == "fg" else
            (lambda m: (np.nan_to_num(m["object_miou"], nan=-1), np.nan_to_num(m["fg_iou"], nan=-1))))

    # one front end per flow-smoothing option (the flow cache keys on it)
    smooths = [None] if args.skip_vsa else grid["flow_smooth_px"]
    models, cfgs = {}, {}
    for sm in smooths:
        c = copy.deepcopy(base)
        if sm:
            c["matching"] = {**c.get("matching", {}), "smooth": int(sm)}
        c = apply_resolution_ratio(c, args.resolution_ratio, verbose=sm == smooths[0])
        c["segmentation"] = {**c.get("segmentation", {}), "head": "cnn", "num_classes": 2}
        cfgs[sm], models[sm] = c, HDEMS(c).to(device).eval()
    cfg0 = cfgs[smooths[0]]
    r = resolution_ratio_of(cfg0)
    smooth_full = {sm: int(sm or base.get("matching", {}).get("smooth", 1)) for sm in smooths}
    rot_src = str((base.get("motion") or {}).get("rot_source", "none"))
    derot_vsa = bool((base.get("vsa_grouping") or {}).get("derotate", False)) and rot_src != "none"
    sp0 = static_params(cfg0)

    ds = build_dataset(cfg0, args.split)
    if len(ds) == 0:
        raise SystemExit(f"[tune] split {args.split!r} has no frames: check dataset.root / "
                         f"the validation scenes {scenes} (python scripts/check_env.py --config ...)")
    idx = (np.arange(len(ds)) if not args.max_frames or args.max_frames >= len(ds)
           else np.unique(np.linspace(0, len(ds) - 1, args.max_frames).round().astype(int)))
    print(f"[tune] split={args.split} scenes={list(scenes)}: {len(idx)} of {len(ds)} frames, "
          f"ratio {r}, {'quick' if args.quick else 'full'} grids, rank by {args.rank}, "
          f"rotation {rot_src}")

    # per frame: labels + flow (+ IMU rotation) for every smoothing option (computed once)
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
                 "ref": event_pixel_mask(x[:, :1])[0].cpu(), "flow": {}, "conf": {},
                 "rot": s["rot_flow"].numpy() if "rot_flow" in s else None,
                 "rot_ok": bool(s["rot_ok"]) if "rot_ok" in s else False,
                 "K": s["intrinsics"].numpy() if "intrinsics" in s else None}
            for sm, m in models.items():
                flow, conf = m.compute_motion(x)
                f["flow"][sm], f["conf"][sm] = flow[0].cpu(), conf[0, 0].cpu()
                if sm == smooths[0]:
                    f["res"] = m.residual_from_flow(flow, x)[0][0].cpu().numpy()
            frames.append(f)
            if (k + 1) % 25 == 0:
                print(f"  flow {k + 1}/{len(idx)}  ({time.time() - t0:.0f} s)", flush=True)
    n_rot = sum(f["rot_ok"] for f in frames)
    print(f"[tune] {n_rot} of {len(frames)} frames have an IMU rate")
    if rot_src != "none" and n_rot < len(frames) // 2:
        print("[tune] WARNING: most validation frames have NO IMU rate -- the IMU methods are "
              "tuned without rotation compensation (scene9/10/12 have no IMU)")

    # --- static-scene test grid (IMU rotation + background model) --------------
    mrows = []
    gp = grouping_params(cfg0)
    for model_name, min_rot, static_px in itertools.product(
            mgrid["model"], mgrid["min_rot_px"], mgrid["static_px"]):
        if model_name == "none" and static_px != mgrid["static_px"][0]:
            continue                         # no background model: static_px plays no role
        key = (model_name, min_rot, static_px)
        sp = static_params({**cfg0, "moving_test": {**(cfg0.get("moving_test") or {}),
                                                    "model": model_name, "min_rot_px": min_rot,
                                                    "static_px": static_px}})
        tic = time.time()
        for f in frames:                     # score map per frame (float32); the derotated
            sc_map, d, _ = moving_score(f["flow"][smooths[0]], f["ref"], sp,   # flow per gate
                                        rot_flow=f["rot"], intrinsics=f["K"], ratio=r)
            f.setdefault("mscore", {})[key] = sc_map.astype(np.float32)
            f.setdefault("derot", {})[min_rot] = d.astype(np.float32)
        ms = (time.time() - tic) / len(frames) * 1000
        for t in MOVING_THRESHOLDS:
            res = _score(frames, lambda f, t=t: (
                (f["mscore"][key] > t / r) & f["ref"].numpy()).astype(np.int64), with_objects=False)
            res["ms"] = ms
            mrows.append(({"model": model_name, "min_rot_px": min_rot, "static_px": static_px,
                           "threshold_px": t}, res))
        best_t = max((row for row in mrows if (row[0]["model"], row[0]["min_rot_px"],
                                               row[0]["static_px"]) == key),
                     key=lambda row: rank(row[1]))
        print(f"  moving_test {model_name:9s} min_rot {min_rot:.1f} static {static_px:.1f} -> best threshold "
              f"{best_t[0]['threshold_px']} px: FG IoU {best_t[1]['fg_iou']:.4f}  ({ms:.0f} ms/frame)",
              flush=True)
    # objects (RANSAC grouping on the derotated flow) for the 5 best by FG IoU
    for setting, res in sorted(mrows, key=lambda row: rank(row[1]), reverse=True)[:5]:
        key = (setting["model"], setting["min_rot_px"], setting["static_px"])
        t = setting["threshold_px"]
        obj = _score(frames, lambda f: group_objects(
            f["derot"][key[1]], (f["mscore"][key] > t / r) & f["ref"].numpy() & f["score"], gp))
        obj.pop("fg_iou")                    # FG IoU stays that of the moving mask, as in eval
        res.update(obj)
    best_m_setting, best_m = max(mrows, key=lambda row: rank(row[1]))
    # Prefer the simplest model unless a background model clearly wins. Measured (ratio 2,
    # 30 validation / 50 eval figure frames): epipolar beat 'none' by 0.013 FG IoU on the
    # validation takes (camera turning 20-45 deg/s) but LOST 0.046 on eval (mostly a slow
    # camera); with no margin the tuner would pick the model that generalises worse.
    nones = [row for row in mrows if row[0]["model"] == "none"]
    if nones and best_m_setting["model"] != "none":
        n_set, n_res = max(nones, key=lambda row: rank(row[1]))
        gain = best_m["fg_iou"] - n_res["fg_iou"]
        if gain < args.min_gain:
            print(f"[tune] {best_m_setting['model']} beats 'none' by only {gain:+.4f} FG IoU "
                  f"(< --min-gain {args.min_gain}): keeping model none")
            best_m_setting, best_m = n_set, n_res

    # --- VSA grouping grid ------------------------------------------------------
    rows = []
    if not args.skip_vsa:
        keys = list(grid)
        for combo in itertools.product(*grid.values()):
            setting = dict(zip(keys, combo))
            sm = setting["flow_smooth_px"]
            gap = 0 if setting["merge_gap_px"] == "0" else smooth_full[sm] // 2
            vg = {**(base.get("vsa_grouping") or {}), **setting, "merge_gap_px": gap}
            p = vsa_params({**cfgs[sm], "vsa_grouping": vg})
            prior = BackgroundPrior()
            model = models[sm]

            def flow_of(f, sm=sm):
                if derot_vsa and f["rot"] is not None:
                    return torch.from_numpy(derotate(f["flow"][sm].numpy(), f["rot"],
                                                     sp0.min_rot)[0].astype(np.float32))
                return f["flow"][sm]
            tic = time.time()
            res = _score(frames, lambda f: vsa_group_objects(
                flow_of(f), f["ref"], model, p, conf=f["conf"][sm], prior=prior,
                seq_id=f["seq"], frame_index=f["fi"])[0])
            res["ms"] = (time.time() - tic) / len(frames) * 1000
            rows.append(({**setting, "merge_gap_px": gap}, res))
            print(f"  vsa {setting} -> FG IoU {res['fg_iou']:.4f}  object mIoU "
                  f"{res['object_miou']:.4f}  objects {res['objects']:.1f}  ({res['ms']:.0f} ms/frame)",
                  flush=True)

    # --- threshold baseline (affine ego residual, no IMU) ---------------------
    base_rows = []
    for t in THRESHOLDS:
        def pred(f, t=t):
            mov = (np.hypot(f["res"][0], f["res"][1]) > t / r) & f["ref"].numpy()
            return group_objects(f["res"], mov & f["score"], gp)
        res = _score(frames, pred)
        base_rows.append((t, res))
        print(f"  baseline threshold {t} px -> FG IoU {res['fg_iou']:.4f}  object mIoU "
              f"{res['object_miou']:.4f}  objects {res['objects']:.1f}", flush=True)
    best_t, best_b = max(base_rows, key=lambda row: rank(row[1]))

    tuned = {"vsa_grouping": {"baseline_threshold_px": best_t},
             "moving_test": dict(best_m_setting)}
    lines = [f"# tuned by scripts/tune_vsa_grouping.py on split={args.split} scenes={list(scenes)} "
             f"({len(frames)} frames, {n_rot} with IMU, ratio {r}, ranked by {args.rank})",
             f"# moving_test: FG IoU {best_m['fg_iou']:.4f}  object mIoU "
             f"{best_m.get('object_miou', float('nan')):.4f}  objects/frame "
             f"{best_m.get('objects', float('nan')):.1f}",
             f"# threshold baseline: FG IoU {best_b['fg_iou']:.4f}  object mIoU {best_b['object_miou']:.4f}"]
    if rows:
        best_setting, best = max(rows, key=lambda row: rank(row[1]))
        tuned["vsa_grouping"].update(best_setting)
        lines.insert(2, f"# VSA grouping: FG IoU {best['fg_iou']:.4f}  object mIoU "
                        f"{best['object_miou']:.4f}  objects/frame {best['objects']:.1f}")
    header = "\n".join(lines) + "\n"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")
    fmt = lambda m: (f"FG IoU {m['fg_iou']:.4f} | object mIoU {m.get('object_miou', float('nan')):.4f} | "
                     f"objects {m.get('objects', float('nan')):.1f}")
    table = [f"moving_test {s} | {fmt(m)} | {m['ms']:.0f} ms"
             for s, m in sorted(mrows, key=lambda row: rank(row[1]), reverse=True)]
    table += [f"vsa {s} | {fmt(m)} | hull {m['hull_iou']:.4f} | {m['ms']:.0f} ms"
              for s, m in sorted(rows, key=lambda row: rank(row[1]), reverse=True)]
    table += [f"baseline threshold {t} | {fmt(m)}" for t, m in base_rows]
    out.with_suffix(".txt").write_text(header + "\n".join(table) + "\n", encoding="utf-8")
    print("\n" + header.rstrip())
    print(f"[tune] wrote {out} and {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
