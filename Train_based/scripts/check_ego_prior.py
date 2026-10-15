"""Sweep the ego-fit anchor (motion.ego_prior_px) and a capture fallback -- training-free,
on VALIDATION takes only, never the test set.

Why (2026-10-15, 60 validation frames): on frames where a moving object covers > 50% of
the scored pixels, the robust ego fit takes the mover for the background and the
residual INVERTS (AUC 0.33; derotated flow without a fit 0.55). Elsewhere the fit helps
(0.92 vs 0.87). With the IMU rotation removed and a camera that hardly translates, the
background should barely move, so:

  prior P    the fit starts at "nothing moves" with tolerance 4 x P (sensor px / 12.5 ms);
             0 = plain least-squares start. A tighter anchor keeps a slow mover out.
  gate G     if the fitted background (rotation excluded) moves more than G px at the
             median event pixel, the fit has locked onto something that is not the
             static scene: use |flow - rotation| for that frame instead.

Reports pixel AUC (all frames, and split by the movers' share of the frame), and the best
event IoU (12.5 ms) over the threshold grid -- the benchmark metric -- for every setting.
The flow comes from the cache, so a sweep takes minutes.

    python scripts/check_ego_prior.py --config configs/evimo_seg.yaml \\
        --val-sequences scene13_dyn_test_03,scene14_dyn_test_02 --max-frames 300 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdems.baselines import THRESHOLD_GRID_PX  # noqa: E402
from hdems.config import apply_resolution_ratio, restore_frontend  # noqa: E402
from hdems.data.build import HOLDOUT, build_dataset  # noqa: E402
from hdems.eval import load_config  # noqa: E402
from hdems.models.hdems import HDEMS  # noqa: E402
from hdems.protocols import GTFrames, _spread_cap  # noqa: E402
from hdems.seg_features import motion_kwargs, score_pixel_mask  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--val-sequences", default="scene13_dyn_test_03,scene14_dyn_test_02")
    ap.add_argument("--checkpoint", default=None, help="optional: rebuild ITS front end")
    ap.add_argument("--resolution-ratio", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--priors", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0],
                    help="motion.ego_prior_px values (sensor px / 12.5 ms)")
    ap.add_argument("--gates", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 2.0],
                    help="capture fallback thresholds (sensor px)")
    ap.add_argument("--window-ms", type=float, default=12.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    if args.checkpoint:
        cfg = restore_frontend(cfg, torch.load(args.checkpoint, map_location="cpu",
                                               weights_only=False).get("frontend"))
    cfg["dataset"] = {**cfg.get("dataset", {}), "val_split": HOLDOUT,
                      "val_sequences": [s for s in args.val_sequences.split(",") if s]}
    cfg = apply_resolution_ratio(cfg, args.resolution_ratio, verbose=True)
    cfg["segmentation"] = {**cfg.get("segmentation", {}), "head": "mfunet", "num_classes": 2}
    model = HDEMS(cfg).to(dev).eval()
    r = model.res_ratio
    own = model.ego_prior_px
    model.ego_gate_px = 0.0              # the gates are applied HERE, per setting (not in the model)
    if model.rot_source == "none":
        raise SystemExit("the anchor only acts with rotation removed (motion.rot_source imu|vicon)")

    ds = build_dataset(cfg, HOLDOUT)
    gtf = GTFrames(ds, [args.window_ms])
    keep = [k for k in range(len(ds)) if not isinstance(gtf(k), str)]
    ks = _spread_cap(keep, args.max_frames)
    print(f"[ego-prior] validation {args.val_sequences}: {len(keep)} frames with a mover, using "
          f"{len(ks)}; ego model {model.ego_model}; current prior {own} px", flush=True)

    names = ["derotated"] + [f"prior {p:g}" for p in args.priors] + \
            [f"prior {p:g} + gate {g:g}" for p in args.priors for g in args.gates]
    px = {n: [] for n in names}
    ev = {n: [] for n in names}
    ys, shares, frame_auc, ev_g = [], [], [], []
    for i, k in enumerate(ks):
        s = ds[k]
        surf = s["surface"].unsqueeze(0).to(dev)
        flow, _conf = model.compute_motion(surf)
        rot = model._rot(motion_kwargs(s, dev).get("rot_flow"), flow)
        valid = model.ego_mask(surf)[0]
        d = flow if rot is None else flow - rot
        cue = {"derotated": d.norm(dim=1)[0] * r}
        for p in args.priors:
            model.ego_prior_px = p
            res, bg = model.background_fit(flow, surf, rot)
            fitted = (bg - (rot if rot is not None else 0.0)).norm(dim=1)[0] * r   # the ego model alone
            capture = float(fitted[valid].median()) if valid.any() else 0.0
            cue[f"prior {p:g}"] = res.norm(dim=1)[0] * r
            for g in args.gates:
                cue[f"prior {p:g} + gate {g:g}"] = cue["derotated"] if capture > g else cue[f"prior {p:g}"]
        model.ego_prior_px = own
        lab = s["mask"].to(dev)
        sc = score_pixel_mask(s, surf)[0] & (lab != 255)
        y = (lab[sc] == 1).cpu().numpy()
        ys.append(y)
        shares.append(float(y.mean()) if y.size else 0.0)
        gt, evs, _fi, _ts = gtf(k)
        x_e, y_e = evs[args.window_ms]
        ev_g.append(gt[y_e, x_e])
        fa = {}
        for n in names:
            px[n].append(cue[n][sc].cpu().numpy())
            full = F.interpolate(cue[n][None, None], size=gt.shape, mode="nearest")[0, 0]
            ev[n].append(full.cpu().numpy()[y_e, x_e])
            if 0 < y.sum() < y.size:
                fa[n] = float(roc_auc_score(y, px[n][-1]))
        frame_auc.append(fa)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ks)}", flush=True)
    gtf.close()

    y_all = np.concatenate(ys)
    shares = np.array(shares)
    big = shares > 0.5

    def best_iou(n):
        out = {}
        for t in THRESHOLD_GRID_PX:
            v = []
            for p, g in zip(ev[n], ev_g):
                pr = p > t
                u = int((pr | g).sum())
                v.append(int((pr & g).sum()) / u if u else np.nan)
            out[t] = float(np.nanmean(v))
        t = max(out, key=lambda q: out[q] if out[q] == out[q] else -1)
        return out[t], t

    print(f"\n{len(ks)} frames, {int(big.sum())} where movers are > 50% of the scored pixels")
    print(f"{'setting':<24s}{'AUC':>7s}{'AUC mover>50%':>15s}{'AUC other':>11s}{'best IoU':>10s}{'@ px':>6s}")
    report = {}
    for n in names:
        auc = roc_auc_score(y_all, np.concatenate(px[n]))
        fb = np.median([f[n] for f, b in zip(frame_auc, big) if b and n in f]) if big.any() else np.nan
        fo = np.median([f[n] for f, b in zip(frame_auc, big) if not b and n in f])
        iou, t = best_iou(n)
        report[n] = {"auc": auc, "frame_auc_mover_dominated": fb, "frame_auc_other": fo,
                     "best_iou": iou, "threshold": t}
        mark = "  <- current" if n == f"prior {own:g}" else ""
        print(f"{n:<24s}{auc:7.3f}{fb:15.3f}{fo:11.3f}{100 * iou:9.1f}%{t:6g}{mark}")
    print("(per-frame AUC columns are medians; IoU = event IoU at the best threshold on THESE "
          "validation frames)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
        print(f"-> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
