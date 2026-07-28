"""Single entry point: supervised per-event FG/BG segmentation.

  python run.py --mode train --train_dir Data/Yolo/train --out_dir /scratch/.../seg_v1
  python run.py --mode test  --test_dir  Data/Yolo/test  --ckpt .../best.pt --out_dir ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    EventWindowDataset,
    fg_bg_ratio,
    iter_windows_streaming,
    list_clips,
    load_event_file,
    process_rss_gb,
    summarize_dir,
    _norm_path,
)
from seg_model import GraphCfg, SegModel, spatial_majority_smooth


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def default_out_dir(name: str = "seg_v1") -> str:
    """Prefer PBS jobfs / scratch — never $HOME on Gadi."""
    for key in ("PBS_JOBFS", "TMPDIR"):
        base = os.environ.get(key)
        if base:
            return str(Path(base) / "runs" / name)
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return str(Path(scratch) / "runs" / name)
    return str(Path("runs") / name)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss (α≈0.25, γ≈2.0) for class-imbalanced FG/BG."""
    targets = targets.float()
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t) ** gamma * ce).mean()


def fg_metrics(logits: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> dict:
    """Foreground IoU / precision / recall / F1 — headline metric is fg-IoU."""
    pred = (torch.sigmoid(logits) >= thr).long()
    gt = labels.long()
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {"iou": iou, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


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


def collate_identity(batch):
    assert len(batch) == 1, "batch_size must be 1 (variable event graphs)"
    return batch[0]


def clip_train_val_split(files: list[Path], val_frac: float = 0.2, seed: int = 0):
    """Hold out clips for validation (never split within a clip)."""
    files = list(files)
    if len(files) == 1:
        print("[data] WARNING: single clip only — using same clip for train/val "
              "(dev-only; not a generalization result)")
        return files, files
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    n_val = max(1, int(round(len(files) * val_frac)))
    n_val = min(n_val, len(files) - 1)
    val_idx = set(order[:n_val].tolist())
    train_files = [files[i] for i in range(len(files)) if i not in val_idx]
    val_files = [files[i] for i in range(len(files)) if i in val_idx]
    print(f"[data] clip split: train={len(train_files)}  val={len(val_files)}")
    for p in val_files:
        print(f"  val clip: {p.name}")
    return train_files, val_files


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: SegModel, loader: DataLoader, device: torch.device) -> dict:
    model.adapter.eval()
    model.head.eval()
    all_logits, all_labels = [], []
    for item in loader:
        logits = model(item["graph"]).to(device)
        labels = item["label"].to(device)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    if not all_logits:
        return {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "loss": 0.0}
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss = float(focal_loss(logits, labels).item())
    m = fg_metrics(logits, labels)
    m["loss"] = loss
    n_fg, n_bg, ratio = fg_bg_ratio(labels.numpy().astype(np.int64))
    m["fg"] = n_fg
    m["bg"] = n_bg
    m["fg_bg_ratio"] = ratio
    return m


def run_train(args):
    out_dir = Path(args.out_dir or default_out_dir("seg_v1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print("[startup] Supervised FG/BG training (frozen GVFA + Adapter/SegHead)")
    print("[startup] src/ encoder is imported only — never modified")
    print(f"[startup] out_dir={out_dir}  (scratch/jobfs preferred)")
    print(f"[startup] window_ms={args.window_ms}  sensor={args.width}x{args.height}")
    print(f"[startup] max_events_per_window={args.max_events_per_window}  "
          f"index_cache_dir={args.index_cache_dir}")
    print(f"[startup] RSS={process_rss_gb():.2f} GiB")

    train_root = _norm_path(args.train_dir)
    fmt = summarize_dir(
        train_root, args.window_ms,
        index_cache_dir=args.index_cache_dir,
        max_events_per_window=args.max_events_per_window,
    )
    all_files = list_clips(train_root, fmt)

    # Prefer DAT header sensor (Prophesee Gen1 = 304x240) when present
    width, height = args.width, args.height
    sample = load_event_file(all_files[0], require_labels=False)  # header only
    if "sensor" in sample:
        sw, sh = sample["sensor"]
        if (args.width, args.height) == (346, 260) and (sw, sh) != (346, 260):
            print(f"[startup] overriding sensor {args.width}x{args.height} "
                  f"-> {sw}x{sh} from DAT header")
            width, height = sw, sh

    cfg = GraphCfg(
        width=width, height=height,
        adapter_dim=args.adapter_dim, num_layers=args.num_layers,
        device=str(device),
    )
    train_files, val_files = clip_train_val_split(
        all_files, val_frac=args.val_frac, seed=args.seed)

    print(f"[startup] RSS before dataset init: {process_rss_gb():.2f} GiB")
    train_ds = EventWindowDataset(
        train_root, cfg, window_ms=args.window_ms,
        stride_ms=args.stride_ms, require_labels=True,
        clip_files=train_files,
        index_cache_dir=args.index_cache_dir,
        max_events_per_window=args.max_events_per_window,
        subsample_seed=args.seed,
        deterministic_subsample=False,
    )
    val_ds = EventWindowDataset(
        train_root, cfg, window_ms=args.window_ms,
        stride_ms=args.stride_ms, require_labels=True,
        clip_files=val_files,
        index_cache_dir=args.index_cache_dir,
        max_events_per_window=args.max_events_per_window,
        subsample_seed=args.seed,
        deterministic_subsample=True,
    )
    print(f"[data] train windows={len(train_ds)}  val windows={len(val_ds)}")
    print(f"[startup] RSS after dataset init: {process_rss_gb():.2f} GiB")

    nw = int(args.num_workers)
    pin = bool(args.pin_memory) and device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=collate_identity, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=collate_identity, num_workers=nw, pin_memory=pin)

    model = SegModel(cfg).to(device)
    n_train = model.assert_encoder_frozen()
    print(f"[startup] trainable params (adapter+head only) = {n_train:,}")
    print("[startup] encoder requires_grad=False  (asserted)")
    print(f"[loss] focal α={args.focal_alpha}  γ={args.focal_gamma}")
    print(f"[startup] RSS after model: {process_rss_gb():.2f} GiB")

    opt = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    best_iou, best_path = -1.0, out_dir / "best.pt"
    history = []
    rss_every = max(1, int(args.rss_every))

    for epoch in range(1, args.epochs + 1):
        model.adapter.train()
        model.head.train()
        model.encoder.eval()
        running = 0.0
        all_logits, all_labels = [], []

        for step, item in enumerate(train_loader, start=1):
            opt.zero_grad(set_to_none=True)
            logits = model(item["graph"])
            labels = item["label"].to(logits.device)
            loss = focal_loss(
                logits, labels, alpha=args.focal_alpha, gamma=args.focal_gamma)
            loss.backward()
            opt.step()
            running += float(loss.item())
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            if step == 1 or step % rss_every == 0:
                print(
                    f"[rss] epoch={epoch} step={step}  "
                    f"RSS={process_rss_gb():.2f} GiB  "
                    f"n_events={item.get('n_events')} "
                    f"(pre-cap={item.get('n_events_pre_cap')})"
                )

        train_logits = torch.cat(all_logits)
        train_labels = torch.cat(all_labels)
        train_m = fg_metrics(train_logits, train_labels)
        train_m["loss"] = running / max(len(train_loader), 1)
        n_fg, n_bg, ratio = fg_bg_ratio(train_labels.numpy().astype(np.int64))
        train_m["fg"] = n_fg
        train_m["bg"] = n_bg
        train_m["fg_bg_ratio"] = ratio
        val_m = evaluate(model, val_loader, device)

        row = {"epoch": epoch, "train": train_m, "val": val_m,
               "rss_gb": process_rss_gb()}
        history.append(row)
        print(
            f"[epoch {epoch:03d}] "
            f"fg/bg={ratio:.4f}  "
            f"train loss={train_m['loss']:.4f}  "
            f"IoU={train_m['iou']:.4f} P={train_m['precision']:.4f} "
            f"R={train_m['recall']:.4f} F1={train_m['f1']:.4f} | "
            f"val IoU={val_m['iou']:.4f} P={val_m['precision']:.4f} "
            f"R={val_m['recall']:.4f} F1={val_m['f1']:.4f}  "
            f"RSS={row['rss_gb']:.2f} GiB"
        )

        ckpt = {
            "epoch": epoch,
            "adapter": model.adapter.state_dict(),
            "head": model.head.state_dict(),
            "cfg": vars(cfg),
            "args": vars(args),
            "metrics": val_m,
            "data_format": fmt,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            torch.save(ckpt, best_path)
            print(f"  -> saved best fg-IoU={best_iou:.4f} to {best_path}")

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[done] best foreground IoU={best_iou:.4f}  ckpt={best_path}")
    print(f"[done] final RSS={process_rss_gb():.2f} GiB")

# ---------------------------------------------------------------------------
# Test / visualization
# ---------------------------------------------------------------------------
def load_checkpoint(ckpt_path: str, device: torch.device, args) -> tuple[SegModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("cfg", {})
    cfg = GraphCfg(
        width=int(args.width if args.width else cfg_dict.get("width", 346)),
        height=int(args.height if args.height else cfg_dict.get("height", 260)),
        adapter_dim=int(cfg_dict.get("adapter_dim", 128)),
        num_layers=int(cfg_dict.get("num_layers", args.num_layers)),
        hv_dim=int(cfg_dict.get("hv_dim", 4000)),
        device=str(device),
    )
    # CLI sensor always wins when provided (argparse always sets defaults)
    cfg.width = int(args.width)
    cfg.height = int(args.height)
    model = SegModel(cfg).to(device)
    model.adapter.load_state_dict(ckpt["adapter"])
    model.head.load_state_dict(ckpt["head"])
    model.adapter.eval()
    model.head.eval()
    model.encoder.eval()
    return model, ckpt


def render_pred_image(x, y, pred, width, height, out_path: Path, title: str = ""):
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


def render_diagnostic_grid(x, y, gt, pred, width, height, out_path: Path, meta: dict):
    """7-panel: raw / GT fg / pred fg / correct / FP / FN / overlay (+ metrics)."""
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
def run_test(args):
    out_dir = Path(args.out_dir or default_out_dir("seg_v1/test_images"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("[startup] Supervised FG/BG test (frozen GVFA + Adapter/SegHead)")
    print("[startup] src/ encoder is imported only — never modified")
    print(f"[startup] ckpt={args.ckpt}")
    print(f"[startup] out_dir={out_dir}")
    print(f"[startup] RSS={process_rss_gb():.2f} GiB")
    summarize_dir(
        args.test_dir, args.window_ms,
        index_cache_dir=args.index_cache_dir,
        max_events_per_window=args.max_events_per_window,
    )

    model, _ckpt = load_checkpoint(args.ckpt, device, args)
    # Prefer DAT header sensor when CLI still has DAVIS defaults
    test_files = list_clips(_norm_path(args.test_dir))
    sample = load_event_file(test_files[0], require_labels=False)
    if "sensor" in sample and (args.width, args.height) == (346, 260):
        sw, sh = sample["sensor"]
        if (sw, sh) != (346, 260):
            print(f"[startup] overriding sensor -> {sw}x{sh} from DAT header")
            args.width, args.height = sw, sh
            model.cfg.width, model.cfg.height = sw, sh
    n_train = model.assert_encoder_frozen()
    print(f"[startup] trainable params loaded (adapter+head) = {n_train:,}")
    print("[startup] encoder requires_grad=False  (asserted)")

    files = test_files
    totals = {"tp": 0, "fp": 0, "fn": 0}
    any_labels = False
    rows = []
    rss_every = max(1, int(args.rss_every))

    for path in files:
        print(f"[test] {path.name} ...")
        for item in iter_windows_streaming(
            path, model.cfg, window_ms=args.window_ms, stride_ms=args.stride_ms,
            index_cache_dir=args.index_cache_dir,
            max_events_per_window=args.max_events_per_window,
            subsample_seed=args.seed,
        ):
            logits = model(item["graph"])
            pred = (torch.sigmoid(logits) >= args.thr).cpu().numpy().astype(bool)
            # Test-time spatial neighbour-majority smoothing
            pred = spatial_majority_smooth(
                pred, item["graph"]["edge_spatial"], majority=args.smooth_majority,
            )
            g = item["graph"]
            x, y = g["x"], g["y"]
            t_ms = int(round(item["t_start_ms"]))
            stem = f"{item['clip']}_{item['window_index']:04d}_{t_ms}ms"
            pred_path = out_dir / f"{stem}.png"
            render_pred_image(
                x, y, pred, args.width, args.height, pred_path, title=stem,
            )

            if item["window_index"] == 0 or (item["window_index"] + 1) % rss_every == 0:
                print(
                    f"[rss] test window={item['window_index']}  "
                    f"RSS={process_rss_gb():.2f} GiB  "
                    f"n_events={item.get('n_events')} "
                    f"(pre-cap={item.get('n_events_pre_cap')})"
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
                rows.append({
                    "clip": item["clip"],
                    "window": item["window_index"],
                    "t_start_ms": item["t_start_ms"],
                    "n_events": len(x),
                    **m,
                })
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
        csv_path = out_dir / "metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["clip", "window", "t_start_ms", "n_events",
                            "iou", "precision", "recall", "f1"],
            )
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({
                "clip": "_overall_", "window": -1, "t_start_ms": "",
                "n_events": "", "iou": iou, "precision": prec,
                "recall": rec, "f1": f1,
            })
        print(f"[done] wrote {csv_path}")
    else:
        print("[done] no labels in test set — inference images only")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("train", "test"), required=True)
    ap.add_argument("--train_dir", default="Data/Yolo/train")
    ap.add_argument("--test_dir", default="Data/Yolo/test")
    ap.add_argument("--ckpt", default=None, help="checkpoint for --mode test")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window_ms", type=float, default=50.0)
    ap.add_argument("--stride_ms", type=float, default=None)
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--adapter_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--smooth_majority", type=float, default=0.6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--index_cache_dir", default="output/index_cache",
        help="directory for per-clip window-index .npz sidecars",
    )
    ap.add_argument(
        "--max_events_per_window", type=int, default=20000,
        help="uniform subsample cap per window (0 = no cap)",
    )
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true", default=False)
    ap.add_argument(
        "--rss_every", type=int, default=50,
        help="print process RSS every N windows/steps",
    )
    return ap

def main():
    args = build_parser().parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        if not args.ckpt:
            raise SystemExit("--mode test requires --ckpt")
        run_test(args)


if __name__ == "__main__":
    main()
