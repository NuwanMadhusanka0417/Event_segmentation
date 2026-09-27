"""Is the VSA flow actually optical flow?  EPE against ground truth, no training.

Measures the flow stage ON ITS OWN, the way the VSA-Flow paper does: end-point
error against GT flow built from EVIMO2 depth + poses (hdems/data/gt_flow.py).

Reported per split, over event pixels that have GT:
  EPE          median |flow - GT|           (px per F0->F1 interval, working res)
  EPE movers   same, on moving-object pixels only
  zero EPE     EPE of predicting zero flow  -> the flow must beat this clearly
  corr u / v   correlation of each component with GT (1 = perfect, 0 = noise)

Why it matters: on the moving-camera training sequences, image motion is 5-9 px
per interval, and a search window of M=7 (+-3 px) cannot represent it -- the
flow there was measured at correlation ~0, i.e. noise. Every accuracy change
(M, alpha, sc, tau, kernel, resolution_ratio) should move these numbers first.

Usage
-----
    python scripts/flow_epe.py --config configs/evimo_seg.yaml --frames 12 --device cuda
    python scripts/flow_epe.py ... --resolution-ratio 2
    python scripts/flow_epe.py ... --M 7 --alpha 0.3 --smooth 3      # old settings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hdems.config import apply_resolution_ratio, resolution_ratio_of   # noqa: E402
from hdems.data.build import build_dataset                            # noqa: E402
from hdems.data.evimo2_reader import load_meta                        # noqa: E402
from hdems.data.gt_flow import gt_flow_for_sample                     # noqa: E402
from hdems.data.motion_labels import IGNORE_LABEL                     # noqa: E402
from hdems.eval import load_config                                    # noqa: E402
from hdems.models.hdems import HDEMS                                  # noqa: E402
from hdems.seg_features import event_pixel_mask                       # noqa: E402


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def evaluate_split(model: HDEMS, cfg: dict, split: str, n_frames: int, device) -> dict | None:
    ds = build_dataset(cfg, split)
    if len(ds) == 0 or not hasattr(ds, "index") or not ds.index:
        print(f"[{split}] no raw-sequence samples (cached shards carry no GT index)")
        return None
    r = resolution_ratio_of(cfg)
    ds_cfg = cfg["dataset"]
    window_s = float(ds_cfg.get("window_ms", 50.0)) / 1000.0
    fracs = ds_cfg.get("time_frames") or [0.0, 0.25]
    picks = np.linspace(0, len(ds) - 1, num=min(n_frames * 3, len(ds))).astype(int)

    rows = []
    # Pixels pooled across frames. A per-frame correlation is meaningless on a
    # frame whose true flow is ~0 everywhere (static camera, no mover in view --
    # most eval frames): there is no variance to correlate with. Pooling lets
    # frames with real motion carry the correlation, as they should.
    pool_f, pool_g, pool_err_mov, pool_zero_mov = [], [], [], []
    rng = np.random.default_rng(0)
    print(f"\n===== {split}: {len(ds)} frames, measuring up to {n_frames} =====")
    print(f"  {'frame':>6} {'seq':26s} {'GT|flow|':>9} {'EPE':>6} {'EPE mov':>8} "
          f"{'zero EPE':>9} {'corr u':>7} {'corr v':>7}")
    for idx in picks:
        if len(rows) >= n_frames:
            break
        seq_dir, fi = ds.index[int(idx)]
        meta = load_meta(seq_dir)
        g_full = gt_flow_for_sample(seq_dir, fi, window_s, fracs, meta)
        if g_full is None:
            continue
        sample = ds[int(idx)]
        s = sample["surface"].unsqueeze(0).to(device)
        H, W = s.shape[-2:]
        g = g_full[:, ::r, ::r][:, :H, :W] / r                    # to working resolution
        with torch.no_grad():
            flow = model._flow_uncached(s)[0].cpu().numpy()
        ev = event_pixel_mask(s)[0].cpu().numpy()
        have = ev & np.isfinite(g[0])
        if have.sum() < 200:
            continue
        lbl = sample["mask"].numpy()
        mov = have & (lbl == 1) & (lbl != IGNORE_LABEL)
        err = np.linalg.norm(flow - np.nan_to_num(g), axis=0)
        gmag = np.linalg.norm(np.nan_to_num(g), axis=0)
        row = {
            "gt": float(np.median(gmag[have])),
            "epe": float(np.median(err[have])),
            "epe_mov": float(np.median(err[mov])) if mov.sum() >= 50 else float("nan"),
            "zero": float(np.median(gmag[have])),
            "cu": _corr(flow[0][have], g[0][have]),
            "cv": _corr(flow[1][have], g[1][have]),
        }
        rows.append(row)
        sel = np.flatnonzero(have.ravel())
        if sel.size > 20000:                                   # cap per frame
            sel = rng.choice(sel, 20000, replace=False)
        pool_f.append(flow.reshape(2, -1)[:, sel])
        pool_g.append(np.nan_to_num(g).reshape(2, -1)[:, sel])
        if mov.sum():
            pool_err_mov.append(err[mov])
            pool_zero_mov.append(gmag[mov])
        print(f"  {int(idx):6d} {seq_dir.name[:26]:26s} {row['gt']:9.2f} {row['epe']:6.2f} "
              f"{row['epe_mov']:8.2f} {row['zero']:9.2f} {row['cu']:7.3f} {row['cv']:7.3f}")
    if not rows:
        print("  no frame had GT flow on enough event pixels")
        return None
    mean = {k: float(np.nanmean([rw[k] for rw in rows])) for k in rows[0]}
    fu, gu = np.concatenate([p[0] for p in pool_f]), np.concatenate([p[0] for p in pool_g])
    fv, gv = np.concatenate([p[1] for p in pool_f]), np.concatenate([p[1] for p in pool_g])
    corr = (_corr(fu, gu) + _corr(fv, gv)) / 2
    if pool_err_mov:
        epe_mov = float(np.median(np.concatenate(pool_err_mov)))
        zero_mov = float(np.median(np.concatenate(pool_zero_mov)))
    else:
        epe_mov = zero_mov = float("nan")
    verdict = ("PASS" if corr >= 0.7 and mean["epe"] < 0.6 * mean["zero"]
               else "WEAK" if corr >= 0.4 else "FAIL")
    n_mov = sum(1 for rw in rows if rw["epe_mov"] == rw["epe_mov"])
    print(f"  POOLED over {len(rows)} frames ({n_mov} with a visible mover):")
    print(f"    all event px : EPE {mean['epe']:.2f} px vs zero-flow {mean['zero']:.2f} | "
          f"correlation {corr:.3f}  -> {verdict}")
    print(f"    moving px    : EPE {epe_mov:.2f} px vs zero-flow {zero_mov:.2f}")
    return {**mean, "corr": corr, "epe_mov": epe_mov, "zero_mov": zero_mov,
            "verdict": verdict, "n": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--split", default="both", choices=["train", "eval", "both"])
    ap.add_argument("--frames", type=int, default=12, help="frames per split")
    ap.add_argument("--resolution-ratio", type=int, default=None)
    ap.add_argument("--d", type=int, default=None, help="override hypervector dim")
    ap.add_argument("--M", type=int, default=None, help="override matching.M (full-res value)")
    ap.add_argument("--alpha", type=float, default=None, help="override matching.alpha")
    ap.add_argument("--smooth", type=int, default=None,
                    help="override matching.smooth / paper sc (full-res value)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.d:
        cfg["d"] = args.d
    match = cfg.setdefault("matching", {})
    if args.M:
        match["M"] = args.M
    if args.alpha is not None:
        match["alpha"] = args.alpha
    if args.smooth:
        match["smooth"] = args.smooth
    cfg = apply_resolution_ratio(cfg, args.resolution_ratio, verbose=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = HDEMS(cfg).to(device).eval()
    model.flow_cache = None             # measuring the flow itself; don't fill the cache
    enc, match = cfg["encoder"], cfg["matching"]        # AFTER the ratio: the values in use
    print(f"[flow-epe] d={cfg.get('d')} kernel={enc.get('kernel', 'conv')} "
          f"polarity_binding={enc.get('polarity_binding', True)} scales={enc.get('scales', 2)} "
          f"N={enc.get('patch_size')} M={match['M']} alpha={match.get('alpha')} "
          f"smooth={match.get('smooth')} tau_ms={cfg.get('time_surface', {}).get('tau_ms')} "
          f"device={device}")

    splits = ["train", "eval"] if args.split == "both" else [args.split]
    ds_cfg = cfg["dataset"]
    results = {}
    for sp in splits:
        name = ds_cfg.get("split", "train") if sp == "train" else ds_cfg.get("eval_split", "eval")
        results[sp] = evaluate_split(model, cfg, name, args.frames, device)

    print("\n=========== FLOW CHECK ===========")
    for sp, res in results.items():
        if res:
            print(f"  {sp:5s}: pooled corr {res['corr']:.3f}  EPE {res['epe']:.2f} vs zero-flow "
                  f"{res['zero']:.2f} | movers EPE {res['epe_mov']:.2f} vs zero "
                  f"{res['zero_mov']:.2f}  -> {res['verdict']}")
    print("  PASS = pooled correlation >= 0.7 AND EPE < 0.6 x zero-flow EPE. Train the\n"
          "  classifier only once both splits pass -- otherwise it learns from noise.")


if __name__ == "__main__":
    main()
