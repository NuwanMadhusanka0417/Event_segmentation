"""Train Adapter + SegHead on frozen GVFA per-event FG/BG labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import EventWindowDataset, summarize_dir
from seg_model import GraphCfg, SegModel


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
    gamma: float = 2.0,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Binary focal loss; pos_weight upweights the rare foreground class."""
    targets = targets.float()
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device),
    )
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    return ((1.0 - p_t) ** gamma * ce).mean()


def fg_metrics(logits: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> dict:
    """Foreground IoU / precision / recall / F1 — never raw accuracy."""
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


def collate_identity(batch):
    assert len(batch) == 1, "batch_size must be 1 (variable event graphs)"
    return batch[0]


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
    return m


def estimate_pos_weight(loader: DataLoader) -> float:
    n_fg = n_bg = 0
    for item in loader:
        y = item["label"]
        n_fg += int((y == 1).sum())
        n_bg += int((y == 0).sum())
    ratio = n_fg / max(n_bg, 1)
    print(f"[data] train fg={n_fg} bg={n_bg} fg/bg={ratio:.6f}")
    return max(n_bg / max(n_fg, 1), 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window_ms", type=float, default=50.0)
    ap.add_argument("--stride_ms", type=float, default=None)
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--adapter_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or default_out_dir("seg_v1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print("[startup] Streaming FG/BG training (frozen GVFA + Adapter/SegHead)")
    print(f"[startup] src/ encoder is imported only — never modified")
    print(f"[startup] out_dir={out_dir}  (scratch/jobfs preferred)")
    print(f"[startup] window_ms={args.window_ms}  sensor={args.width}x{args.height}")
    summarize_dir(args.train_dir, args.window_ms)
    summarize_dir(args.val_dir, args.window_ms)

    cfg = GraphCfg(
        width=args.width, height=args.height,
        adapter_dim=args.adapter_dim, num_layers=args.num_layers,
        device=str(device),
    )
    train_ds = EventWindowDataset(
        args.train_dir, cfg, window_ms=args.window_ms,
        stride_ms=args.stride_ms, require_labels=True)
    val_ds = EventWindowDataset(
        args.val_dir, cfg, window_ms=args.window_ms,
        stride_ms=args.stride_ms, require_labels=True)
    print(f"[data] train windows={len(train_ds)}  val windows={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_identity, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_identity, num_workers=0)

    model = SegModel(cfg).to(device)
    n_train = model.assert_encoder_frozen()
    print(f"[startup] trainable params (adapter+head only) = {n_train:,}")
    print(f"[startup] encoder requires_grad=False  (asserted)")

    pos_weight = estimate_pos_weight(train_loader)
    print(f"[loss] focal gamma=2.0  pos_weight={pos_weight:.3f}")

    opt = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    best_iou, best_path = -1.0, out_dir / "best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.adapter.train()
        model.head.train()
        model.encoder.eval()
        running = 0.0
        all_logits, all_labels = [], []

        for item in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(item["graph"])
            labels = item["label"].to(logits.device)
            loss = focal_loss(logits, labels, pos_weight=pos_weight)
            loss.backward()
            opt.step()
            running += float(loss.item())
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        train_m = fg_metrics(torch.cat(all_logits), torch.cat(all_labels))
        train_m["loss"] = running / max(len(train_loader), 1)
        val_m = evaluate(model, val_loader, device)

        row = {"epoch": epoch, "train": train_m, "val": val_m}
        history.append(row)
        print(
            f"[epoch {epoch:03d}] "
            f"train loss={train_m['loss']:.4f}  "
            f"IoU={train_m['iou']:.4f} P={train_m['precision']:.4f} "
            f"R={train_m['recall']:.4f} F1={train_m['f1']:.4f} | "
            f"val IoU={val_m['iou']:.4f} P={val_m['precision']:.4f} "
            f"R={val_m['recall']:.4f} F1={val_m['f1']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "adapter": model.adapter.state_dict(),
            "head": model.head.state_dict(),
            "cfg": vars(cfg),
            "args": vars(args),
            "metrics": val_m,
            "pos_weight": pos_weight,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            torch.save(ckpt, best_path)
            print(f"  -> saved best IoU={best_iou:.4f} to {best_path}")

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[done] best foreground IoU={best_iou:.4f}  ckpt={best_path}")


if __name__ == "__main__":
    main()
