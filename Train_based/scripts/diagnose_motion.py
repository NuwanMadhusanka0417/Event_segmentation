"""GATES: does the motion pipeline carry a usable signal? No training needed.

Gate 1 -- four-panel figure per frame: events | |flow| | |residual| | motion label.
         PASS when the residual is bright on the moving object and dark on the
         table. FAIL means the flow/ego stage is broken and no retrain is worth
         running.

Gate 2 -- separability numbers: AUC of |residual| between moving and static EVENT
         pixels (0.5 = chance), and the events-only baseline, i.e. the foreground
         IoU you get by calling every event pixel "moving". After the label fix
         that baseline must be poor -- if it still scores well, the labels are
         still not motion.

Usage
-----
    python scripts/diagnose_motion.py --config configs/evimo_seg.yaml \
        --split eval --frames 6 --out diagnostics/ [--d 256] [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hdems.data.build import build_dataset                      # noqa: E402
from hdems.data.labels import resolve_label_mode                # noqa: E402
from hdems.data.motion_labels import IGNORE_LABEL               # noqa: E402
from hdems.eval import load_config                              # noqa: E402
from hdems.instances import binary_iou                          # noqa: E402
from hdems.models.hdems import HDEMS                            # noqa: E402
from hdems.models.motion import ego_residual                    # noqa: E402
from hdems.models.paper_flow import flow_from_cost, multiscale_cost_volume  # noqa: E402
from hdems.seg_features import event_pixel_mask                 # noqa: E402


def auc_above(a: np.ndarray, b: np.ndarray, sample: int = 600) -> float:
    """P(score of a random moving pixel > score of a random static pixel)."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    step = max(1, b.size // sample)
    return float((a[:, None] > b[None, ::step]).mean())


def four_panel(surface, flow_mag, res_mag, label, events, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = surface.reshape(-1, *surface.shape[-2:]).sum(0)
    show_lbl = np.where(label == IGNORE_LABEL, 0.5, label.astype(float))  # ignore = grey
    hi_f = np.percentile(flow_mag[events], 99) if events.any() else 1.0
    hi_r = np.percentile(res_mag[events], 99) if events.any() else 1.0
    panels = [
        (ev, "events", dict(cmap="gray")),
        (flow_mag * events, "|flow|", dict(cmap="magma", vmin=0, vmax=max(hi_f, 1e-6))),
        (res_mag * events, "|residual| (ego-compensated)", dict(cmap="magma", vmin=0, vmax=max(hi_r, 1e-6))),
        (show_lbl, "motion label (grey = ignore)", dict(cmap="gray", vmin=0, vmax=1)),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for a, (im, t, kw) in zip(ax, panels):
        a.imshow(im, interpolation="nearest", **kw)
        a.set_title(t, fontsize=10)
        a.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--split", default=None, help="dataset split (default: eval split)")
    ap.add_argument("--frames", type=int, default=6, help="frames to inspect")
    ap.add_argument("--stride", type=int, default=0, help="frame stride (0 = spread evenly)")
    ap.add_argument("--out", default="diagnostics", help="output directory for figures")
    ap.add_argument("--d", type=int, default=None, help="override hypervector dim (CPU runs)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.d:
        cfg["d"] = args.d
    label_mode = resolve_label_mode(cfg)
    if label_mode != "motion":
        print(f"[warn] label_mode={label_mode!r}: these gates are meant for 'motion'")

    split = args.split or cfg.get("dataset", {}).get("eval_split", "eval")
    ds = build_dataset(cfg, split)
    if len(ds) == 0:
        raise SystemExit(f"no samples in split {split!r}")
    device = torch.device(args.device)
    model = HDEMS(cfg).to(device).eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = args.stride or max(1, len(ds) // max(args.frames, 1))
    picks = list(range(0, len(ds), stride))[: args.frames]
    print(f"[gate] split={split} frames={len(ds)} inspecting {picks} d={cfg.get('d')}")

    aucs, base_ious, fg_fracs = [], [], []
    for k, idx in enumerate(picks):
        sample = ds[idx]
        surface = sample["surface"].unsqueeze(0).to(device)
        label = sample["mask"].numpy()
        with torch.no_grad():
            fields = model.encode_times(surface)
            cost = multiscale_cost_volume(fields, model.matcher.M, model.match_scales)
            flow = flow_from_cost(cost, model.matcher.M, alpha=model.flow_alpha,
                                  vel_scale=model.vel_scale, smooth=model.flow_smooth)
            events_t = event_pixel_mask(surface)
            residual, _ = ego_residual(flow, iters=model.ego_iters, valid=events_t)

        events = events_t[0].cpu().numpy()
        flow_mag = flow.norm(dim=1)[0].cpu().numpy()
        res_mag = residual.norm(dim=1)[0].cpu().numpy()

        scored = events & (label != IGNORE_LABEL)
        moving = scored & (label == 1)
        static = scored & (label == 0)
        auc = auc_above(res_mag[moving], res_mag[static])
        auc_flow = auc_above(flow_mag[moving], flow_mag[static])
        # events-only control: predict "moving" everywhere there are events
        base_iou = binary_iou(events & scored, moving, valid=scored)
        fg_frac = float(moving.sum()) / max(int(scored.sum()), 1)
        if moving.sum() >= 50:
            aucs.append(auc)
            base_ious.append(base_iou)
            fg_fracs.append(fg_frac)

        print(f"  frame {idx:5d}: event px {int(events.sum()):6d}  moving {int(moving.sum()):6d} "
              f"({fg_frac:5.1%} of scored)  ignore {int((label == IGNORE_LABEL).sum()):6d} | "
              f"AUC flow {auc_flow:.3f}  AUC residual {auc:.3f} | events-only IoU {base_iou:.3f}")
        four_panel(sample["surface"].numpy(), flow_mag, res_mag, label, events,
                   out_dir / f"gate_{k:02d}_frame{idx:05d}.png",
                   f"{split} frame {idx} — AUC(residual) {auc:.3f}, moving {fg_frac:.1%} of scored px")

    print(f"\nFigures -> {out_dir.resolve()}")
    if not aucs:
        print("GATE 1: no frame had >=50 moving event pixels — check label thresholds.")
        return
    mean_auc = float(np.mean(aucs))
    mean_base = float(np.mean(base_ious))
    print(f"GATE 1 (signal):        mean AUC(|residual|) = {mean_auc:.3f}   "
          f"({'PASS' if mean_auc >= 0.6 else 'WEAK' if mean_auc >= 0.55 else 'FAIL'}; 0.5 = chance)")
    print(f"GATE 2 (events-only):   mean FG IoU = {mean_base:.3f}   "
          f"({'PASS' if mean_base <= 0.35 else 'FAIL — labels may still not be motion'})")
    print(f"        moving pixels are {float(np.mean(fg_fracs)):.1%} of scored pixels on average")


if __name__ == "__main__":
    main()
