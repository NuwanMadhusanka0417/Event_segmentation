"""Test frozen GVFA + SegHead: per-window FG/BG images (streaming order)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import iter_windows_streaming, list_event_files, summarize_dir
from seg_model import GraphCfg, SegModel


def default_out_dir(name: str = "seg_v1_test") -> str:
    for key in ("PBS_JOBFS", "TMPDIR"):
        base = os.environ.get(key)
        if base:
            return str(Path(base) / "runs" / name)
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return str(Path(scratch) / "runs" / name)
    return str(Path("runs") / name)


def load_checkpoint(ckpt_path: str, device: torch.device) -> tuple[SegModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("cfg", {})
    cfg = GraphCfg(
        width=int(cfg_dict.get("width", 346)),
        height=int(cfg_dict.get("height", 260)),
        adapter_dim=int(cfg_dict.get("adapter_dim", 128)),
        num_layers=int(cfg_dict.get("num_layers", 3)),
        hv_dim=int(cfg_dict.get("hv_dim", 4000)),
        device=str(device),
    )
    model = SegModel(cfg).to(device)
    model.adapter.load_state_dict(ckpt["adapter"])
    model.head.load_state_dict(ckpt["head"])
    model.adapter.eval()
    model.head.eval()
    model.encoder.eval()
    return model, ckpt


def fg_metrics_np(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {"iou": iou, "precision": prec, "recall": rec, "f1": f1}


def render_pred_image(
    x, y, pred, width, height, out_path: Path, title: str = "",
):
    """Grey = background, red = foreground."""
    img = np.zeros((height, width, 3), dtype=np.float32)
    xi = np.clip(x.astype(int), 0, width - 1)
    yi = np.clip(y.astype(int), 0, height - 1)
    bg = ~pred.astype(bool)
    fg = pred.astype(bool)
    img[yi[bg], xi[bg]] = (0.55, 0.55, 0.55)
    img[yi[fg], xi[fg]] = (0.95, 0.15, 0.15)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(img, origin="upper")
    ax.set_title(title or out_path.name)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_diagnostic_grid(
    x, y, gt, pred, width, height, out_path: Path, meta: dict,
):
    """7-panel: raw / GT fg / pred fg / correct / FP / FN / overlay."""
    xi = np.clip(x.astype(int), 0, width - 1)
    yi = np.clip(y.astype(int), 0, height - 1)
    gt_b = gt.astype(bool)
    pr_b = pred.astype(bool)

    def scatter_panel(ax, mask, color, title, bg_mask=None):
        ax.set_xlim(0, width); ax.set_ylim(height, 0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        if bg_mask is not None and bg_mask.any():
            ax.scatter(xi[bg_mask], yi[bg_mask], c="0.75", s=2, linewidths=0)
        if mask is not None and mask.any():
            ax.scatter(xi[mask], yi[mask], c=color, s=3, linewidths=0)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    scatter_panel(axes[0], np.ones(len(x), dtype=bool), "k", "raw events")
    scatter_panel(axes[1], gt_b, "tab:red", "ground-truth fg", ~gt_b)
    scatter_panel(axes[2], pr_b, "tab:red", "predicted fg", ~pr_b)
    scatter_panel(axes[3], gt_b & pr_b, "tab:green", "correct (TP)", ~(gt_b & pr_b))
    scatter_panel(axes[4], pr_b & ~gt_b, "tab:orange", "false positive", ~(pr_b & ~gt_b))
    scatter_panel(axes[5], ~pr_b & gt_b, "tab:blue", "false negative", ~(~pr_b & gt_b))

    # Overlay: GT green circles, pred red points
    ax = axes[6]
    ax.set_xlim(0, width); ax.set_ylim(height, 0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("overlay (GT=lime, pred=red)", fontsize=9)
    ax.scatter(xi, yi, c="0.8", s=2, linewidths=0)
    if gt_b.any():
        ax.scatter(xi[gt_b], yi[gt_b], facecolors="none", edgecolors="lime",
                   s=12, linewidths=0.6)
    if pr_b.any():
        ax.scatter(xi[pr_b], yi[pr_b], c="red", s=4, linewidths=0)

    axes[7].axis("off")
    m = meta
    text = (
        f"clip={m['clip']}\n"
        f"window={m['window_index']:04d}  t0={m['t_start_ms']:.1f} ms\n"
        f"window_ms={m['window_ms']:.0f}  sensor={width}x{height}\n"
        f"events={len(x)}  fg_gt={int(gt_b.sum())}  fg_pred={int(pr_b.sum())}\n"
        f"IoU={m['iou']:.4f}\n"
        f"precision={m['precision']:.4f}\n"
        f"recall={m['recall']:.4f}\n"
        f"F1={m['f1']:.4f}"
    )
    axes[7].text(0.05, 0.95, text, va="top", ha="left", family="monospace", fontsize=10)
    fig.suptitle(f"{m['clip']}  window {m['window_index']:04d}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--window_ms", type=float, default=50.0)
    ap.add_argument("--stride_ms", type=float, default=None)
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save_grid", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or default_out_dir("seg_v1/test_images"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("[startup] Streaming FG/BG test (frozen GVFA + Adapter/SegHead)")
    print("[startup] src/ encoder is imported only — never modified")
    print(f"[startup] ckpt={args.ckpt}")
    print(f"[startup] out_dir={out_dir}")
    summarize_dir(args.test_dir, args.window_ms)

    model, ckpt = load_checkpoint(args.ckpt, device)
    # CLI sensor size overrides ckpt when provided explicitly
    model.cfg.width = args.width
    model.cfg.height = args.height
    n_train = model.assert_encoder_frozen()
    print(f"[startup] trainable params loaded (adapter+head) = {n_train:,}")
    print(f"[startup] encoder requires_grad=False  (asserted)")

    files = list_event_files(args.test_dir)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    any_labels = False

    for path in files:
        print(f"[test] {path.name} ...")
        for item in iter_windows_streaming(
            path, model.cfg, window_ms=args.window_ms, stride_ms=args.stride_ms,
        ):
            logits = model(item["graph"])
            pred = (torch.sigmoid(logits) >= args.thr).cpu().numpy().astype(bool)
            g = item["graph"]
            x, y = g["x"], g["y"]
            t_ms = int(round(item["t_start_ms"]))
            stem = f"{item['clip']}_{item['window_index']:04d}_{t_ms}ms"
            pred_path = out_dir / f"{stem}.png"
            render_pred_image(
                x, y, pred, args.width, args.height, pred_path,
                title=stem,
            )

            if item.get("has_labels"):
                any_labels = True
                gt = item["label"].astype(bool)
                m = fg_metrics_np(pred, gt)
                totals["tp"] += int((pred & gt).sum())
                totals["fp"] += int((pred & ~gt).sum())
                totals["fn"] += int((~pred & gt).sum())
                print(
                    f"  window {item['window_index']:04d}  "
                    f"IoU={m['iou']:.4f} P={m['precision']:.4f} "
                    f"R={m['recall']:.4f}  -> {pred_path.name}"
                )
                if args.save_grid:
                    grid_path = out_dir / f"{stem}_grid.png"
                    meta = {
                        "clip": item["clip"],
                        "window_index": item["window_index"],
                        "t_start_ms": item["t_start_ms"],
                        "window_ms": args.window_ms,
                        **m,
                    }
                    render_diagnostic_grid(
                        x, y, gt, pred, args.width, args.height, grid_path, meta,
                    )
            else:
                print(f"  window {item['window_index']:04d}  (no labels) -> {pred_path.name}")

    if any_labels:
        tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        iou = tp / max(tp + fp + fn, 1)
        print(
            f"[done] overall foreground  IoU={iou:.4f}  "
            f"P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}"
        )
    else:
        print("[done] no labels in test set — inference images only")


if __name__ == "__main__":
    main()
