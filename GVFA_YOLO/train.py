"""
Train / eval / smoke-test entry point for two-graph GVFA + point-wise YOLO.

Usage
-----
  # Forward-pass smoke test on the UNLABELED file (default, always runnable):
  python train.py smoke --events events_filtered.txt --window_ms 50

  # Train (needs a labeled adapter wired in dataset.py):
  python train.py train --events path/to/events.txt --labels path/to/boxes.npy

  # Eval COCO-style mAP on a labeled set:
  python train.py eval  --events path/to/events.txt --labels path/to/boxes.npy

Only the adapter (step 5) and head (step 6) train; GraphCNN is frozen.
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.ops import box_iou

from model import (GVFAEncoder, PointwiseYOLO, detection_loss, postprocess,
                   decode_boxes, D, NUM_LAYERS, N_CLASSES, LR, WINDOW_MS)
from dataset import Gen1ETraMDataset, load_unlabeled


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
#                             COCO-style mAP                                   #
# --------------------------------------------------------------------------- #
def _ap_from_pr(rec, prec):
    """101-point interpolated AP (COCO style)."""
    rec = np.concatenate(([0.0], rec, [1.0]))
    prec = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(prec) - 1, 0, -1):
        prec[i - 1] = max(prec[i - 1], prec[i])
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t]
        ap += (p.max() if p.size else 0.0)
    return ap / 101.0


def coco_map(all_preds, all_gts, n_classes, iou_thrs=None):
    """all_preds/all_gts: list per image.
       preds[i] = (boxes_xyxy[K,4], scores[K], labels[K])
       gts[i]   = (boxes_xyxy[M,4], labels[M])
    Returns mAP (mean over thrs & classes) and per-class AP@0.5.
    """
    if iou_thrs is None:
        iou_thrs = np.arange(0.5, 1.0, 0.05)          # COCO 0.5:0.95
    ap_table = np.zeros((len(iou_thrs), n_classes))
    for ti, thr in enumerate(iou_thrs):
        for c in range(n_classes):
            scores, tps, n_gt = [], [], 0
            for (pb, ps, pl), (gb, gl) in zip(all_preds, all_gts):
                cm_g = gl == c
                gboxes = gb[cm_g]
                n_gt += gboxes.shape[0]
                cm_p = pl == c
                pboxes, pscores = pb[cm_p], ps[cm_p]
                if pboxes.shape[0] == 0:
                    continue
                order = pscores.argsort(descending=True)
                pboxes, pscores = pboxes[order], pscores[order]
                matched = torch.zeros(gboxes.shape[0], dtype=torch.bool)
                if gboxes.shape[0]:
                    ious = box_iou(pboxes, gboxes)
                for k in range(pboxes.shape[0]):
                    scores.append(float(pscores[k]))
                    if gboxes.shape[0] == 0:
                        tps.append(0); continue
                    iou_k, j = ious[k].max(0)
                    if float(iou_k) >= thr and not matched[j]:
                        matched[j] = True; tps.append(1)
                    else:
                        tps.append(0)
            if n_gt == 0:
                ap_table[ti, c] = np.nan
                continue
            if not scores:
                ap_table[ti, c] = 0.0
                continue
            order = np.argsort(-np.array(scores))
            tp = np.array(tps)[order]
            fp = 1 - tp
            tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
            rec = tp_cum / max(n_gt, 1)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap_table[ti, c] = _ap_from_pr(rec, prec)
    mAP = np.nanmean(ap_table)
    ap50 = ap_table[0]                       # @0.5
    return float(mAP), ap50


# --------------------------------------------------------------------------- #
#                                  train                                       #
# --------------------------------------------------------------------------- #
def build_models(device, n_classes=N_CLASSES):
    encoder = GVFAEncoder(dim=D, num_layers=NUM_LAYERS, device=device).to(device)
    encoder.eval()                                    # frozen
    head = PointwiseYOLO(dim=D, n_classes=n_classes).to(device)
    return encoder, head


def train(args):
    device = get_device()
    print(f"[train] device={device}")
    ds = Gen1ETraMDataset(args.events, label_path=args.labels,
                          window_ms=args.window_ms)
    encoder, head = build_models(device, ds_classes(args))
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(args.epochs):
        running = {}
        for i in range(len(ds)):
            sample = ds[i]
            ev, boxes = sample["events"], sample["boxes"]
            with torch.no_grad():
                H, pos = encoder(ev)                  # GVFA frozen, no grad
            gt_boxes = torch.tensor(boxes[:, 1:], dtype=torch.float32, device=device) \
                if boxes.shape[0] else torch.zeros((0, 4), device=device)
            gt_cls = torch.tensor(boxes[:, 0], dtype=torch.long, device=device) \
                if boxes.shape[0] else torch.zeros((0,), dtype=torch.long, device=device)

            opt.zero_grad()
            with torch.autocast(device_type=device, enabled=use_amp):
                preds = head(H)
                loss, parts = detection_loss(preds, pos, gt_boxes, gt_cls, head.n_classes)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + (v if k != "n_pos" else 0)
        msg = " ".join(f"{k}={running[k] / max(len(ds),1):.3f}" for k in ("cls", "obj", "iou", "l1"))
        print(f"[epoch {epoch}] {msg}")
    torch.save(head.state_dict(), args.ckpt)
    print(f"[train] saved head -> {args.ckpt}")


def ds_classes(args):
    return args.n_classes


@torch.no_grad()
def evaluate(args):
    device = get_device()
    ds = Gen1ETraMDataset(args.events, label_path=args.labels, window_ms=args.window_ms)
    encoder, head = build_models(device, args.n_classes)
    if args.ckpt:
        head.load_state_dict(torch.load(args.ckpt, map_location=device))
    head.eval()
    all_preds, all_gts = [], []
    for i in range(len(ds)):
        sample = ds[i]
        ev, boxes = sample["events"], sample["boxes"]
        H, pos = encoder(ev)
        preds = head(H)
        pb, ps, pl = postprocess(preds, pos, score_thr=args.score_thr)
        all_preds.append((pb.cpu(), ps.cpu(), pl.cpu()))
        if boxes.shape[0]:
            gx, gy, gw, gh = (boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4])
            gxyxy = torch.tensor(np.stack([gx - gw / 2, gy - gh / 2,
                                           gx + gw / 2, gy + gh / 2], 1), dtype=torch.float32)
            gl = torch.tensor(boxes[:, 0], dtype=torch.long)
        else:
            gxyxy, gl = torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
        all_gts.append((gxyxy, gl))
    mAP, ap50 = coco_map(all_preds, all_gts, args.n_classes)
    print(f"[eval] mAP@[.5:.95] = {mAP:.4f}")
    for c in range(args.n_classes):
        print(f"       AP@0.5 class {c}: {ap50[c]:.4f}")
    return mAP


# --------------------------------------------------------------------------- #
#                       unlabeled forward-only smoke test                       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def smoke(args):
    device = get_device()
    print(f"[smoke] device={device}  events={args.events}  window_ms={args.window_ms}")
    ev = load_unlabeled(args.events, window_ms=args.window_ms, max_events=args.max_events)
    print(f"[smoke] window events: {ev.shape[0]}")
    encoder, head = build_models(device, args.n_classes)
    head.eval()

    H, pos = encoder(ev)
    print(f"[smoke] H (per-node, concat 2D): {tuple(H.shape)}  expected [N, {2*D}]")
    preds = head(H)
    print(f"[smoke] cls {tuple(preds['cls'].shape)}  reg {tuple(preds['reg'].shape)}  "
          f"obj {tuple(preds['obj'].shape)}")
    boxes, scores, labels = postprocess(preds, pos, score_thr=args.score_thr)
    print(f"[smoke] boxes after NMS (untrained head): {boxes.shape[0]}")

    # scatter: events + predicted boxes overlaid
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ev[:, 1], ev[:, 2], s=1, c=ev[:, 0], cmap="viridis", alpha=0.4)
    for b, s in zip(boxes.cpu().numpy(), scores.cpu().numpy()):
        x1, y1, x2, y2 = b
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="red", lw=1.0))
    ax.set_title(f"GVFA-YOLO smoke test (untrained) — {boxes.shape[0]} boxes / {ev.shape[0]} events")
    ax.set_xlim(0, 346); ax.set_ylim(260, 0)
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"[smoke] saved -> {args.out}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "train", "eval"], nargs="?", default="smoke")
    ap.add_argument("--events", default="events_filtered.txt")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--window_ms", type=float, default=WINDOW_MS)
    ap.add_argument("--max_events", type=int, default=40000)
    ap.add_argument("--n_classes", type=int, default=N_CLASSES)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--ckpt", default="head.pt")
    ap.add_argument("--out", default="boxes_on_events.png")
    args = ap.parse_args()

    if args.mode == "smoke":
        smoke(args)
    elif args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
