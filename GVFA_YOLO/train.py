"""
Train / eval / test entry point for edge-conditioned GVFA + point-wise YOLO.

Prophesee Gen1 Automotive Detection
------------------------------------
  # 1) Extract archives (once):
  #    7z x ../Data/Prophesee/train_a.7z -o../Data/Prophesee/train_a
  #    7z x ../Data/Prophesee/val_a.7z   -o../Data/Prophesee/val_a

  # 2) Train:
  python train.py train --data_dir ../Data/Prophesee/train_a --val_dir ../Data/Prophesee/val_a

  # 3) Eval mAP:
  python train.py eval --data_dir ../Data/Prophesee/val_a --ckpt head.pt

  # 4) Test + save event frames with detections:
  python train.py test --data_dir ../Data/Prophesee/val_a --ckpt head.pt --out_dir runs/test_frames

Legacy txt smoke test:
  python train.py smoke --events events_filtered.txt
"""

import argparse
import os
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.ops import box_iou

from gvfa_encoder import GVFAEncoder
from model import (PointwiseYOLO, detection_loss, postprocess,
                   D, NUM_LAYERS, N_CLASSES, LR, WINDOW_MS, set_sensor)
from dataset import make_dataset, load_unlabeled, PropheseeDetectionDataset
from prophesee_io import save_detection_frame, PROPHESEE_CLASS_NAMES


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_sensor(ds):
    w, h = ds.sensor
    set_sensor(w, h)
    return (w, h)


def build_models(device, n_classes=N_CLASSES, sensor=None):
    sensor_tuple = sensor or (304, 240)
    encoder = GVFAEncoder(
        dim=D, num_layers=NUM_LAYERS, device=device, sensor=sensor_tuple,
    ).to(device)
    encoder.eval()
    head = PointwiseYOLO(dim=D, n_classes=n_classes).to(device)
    return encoder, head


# --------------------------------------------------------------------------- #
#                             COCO-style mAP                                   #
# --------------------------------------------------------------------------- #
def _ap_from_pr(rec, prec):
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
    if iou_thrs is None:
        iou_thrs = np.arange(0.5, 1.0, 0.05)
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
                        tps.append(0)
                        continue
                    iou_k, j = ious[k].max(0)
                    if float(iou_k) >= thr and not matched[j]:
                        matched[j] = True
                        tps.append(1)
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
    ap50 = ap_table[0]
    return float(mAP), ap50


def _sample_to_tensors(sample, device):
    ev = sample["events"]
    boxes = sample["boxes"]
    gt_boxes = torch.tensor(boxes[:, 1:], dtype=torch.float32, device=device) \
        if boxes.shape[0] else torch.zeros((0, 4), device=device)
    gt_cls = torch.tensor(boxes[:, 0], dtype=torch.long, device=device) \
        if boxes.shape[0] else torch.zeros((0,), dtype=torch.long, device=device)
    return ev, gt_boxes, gt_cls


def _run_epoch(encoder, head, ds, device, optimizer=None, scaler=None):
    train = optimizer is not None
    if train:
        head.train()
    else:
        head.eval()

    running = {}
    all_preds, all_gts = [], []
    use_amp = scaler is not None and device == "cuda"

    indices = list(range(len(ds)))
    if train:
        random.shuffle(indices)

    for i in indices:
        sample = ds[i]
        ev, gt_boxes, gt_cls = _sample_to_tensors(sample, device)
        with torch.no_grad():
            H, pos = encoder(ev)
        if train:
            optimizer.zero_grad()
            with torch.autocast(device_type=device, enabled=use_amp):
                preds = head(H)
                loss, parts = detection_loss(preds, pos, gt_boxes, gt_cls, head.n_classes)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for k, v in parts.items():
                if k != "n_pos":
                    running[k] = running.get(k, 0.0) + v
        else:
            preds = head(H)
            pb, ps, pl = postprocess(preds, pos)
            all_preds.append((pb.cpu(), ps.cpu(), pl.cpu()))
            if gt_boxes.shape[0]:
                gx, gy, gw, gh = gt_boxes.unbind(1)
                gxyxy = torch.stack([gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2], 1)
                gl = gt_cls
            else:
                gxyxy, gl = torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
            all_gts.append((gxyxy.cpu(), gl.cpu()))

    if train:
        n = max(len(ds), 1)
        msg = " ".join(f"{k}={running[k] / n:.3f}" for k in ("cls", "obj", "iou", "l1"))
        return msg, None
    return None, (all_preds, all_gts)


# --------------------------------------------------------------------------- #
#                                  train                                       #
# --------------------------------------------------------------------------- #
def train(args):
    device = get_device()
    print(f"[train] device={device}")
    ds = make_dataset(args)
    sensor = configure_sensor(ds)
    print(f"[train] sensor={sensor[0]}x{sensor[1]}  windows={len(ds)}  "
          f"recordings={len(getattr(ds, 'pairs', []))}")

    encoder, head = build_models(device, args.n_classes, sensor=sensor)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    val_ds = None
    if args.val_dir:
        val_ds = PropheseeDetectionDataset(
            args.val_dir, window_ms=args.window_ms, max_events=args.max_events,
            max_recordings=args.max_recordings, max_windows=args.max_windows,
        )
        configure_sensor(val_ds)

    for epoch in range(args.epochs):
        msg, _ = _run_epoch(encoder, head, ds, device, opt, scaler)
        print(f"[epoch {epoch}] train {msg}")
        if val_ds is not None:
            _, (vp, vg) = _run_epoch(encoder, head, val_ds, device)
            mAP, ap50 = coco_map(vp, vg, args.n_classes)
            print(f"[epoch {epoch}] val   mAP@[.5:.95]={mAP:.4f}  "
                  + "  ".join(f"AP50_{PROPHESEE_CLASS_NAMES[c]}={ap50[c]:.3f}"
                             for c in range(args.n_classes)))

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save(head.state_dict(), args.ckpt)
    print(f"[train] saved head -> {args.ckpt}")


@torch.no_grad()
def evaluate(args):
    device = get_device()
    ds = make_dataset(args)
    sensor = configure_sensor(ds)
    print(f"[eval] sensor={sensor}  windows={len(ds)}")
    encoder, head = build_models(device, args.n_classes, sensor=sensor)
    if args.ckpt:
        head.load_state_dict(torch.load(args.ckpt, map_location=device))
    _, (all_preds, all_gts) = _run_epoch(encoder, head, ds, device)
    mAP, ap50 = coco_map(all_preds, all_gts, args.n_classes)
    print(f"[eval] mAP@[.5:.95] = {mAP:.4f}")
    for c in range(args.n_classes):
        name = PROPHESEE_CLASS_NAMES[c] if c < len(PROPHESEE_CLASS_NAMES) else str(c)
        print(f"       AP@0.5 {name}: {ap50[c]:.4f}")
    return mAP


@torch.no_grad()
def test_and_save_frames(args):
    """Run inference and save event-stream frames with predicted (and GT) boxes."""
    device = get_device()
    ds = make_dataset(args)
    sensor = configure_sensor(ds)
    sw, sh = sensor
    print(f"[test] sensor={sensor}  windows={len(ds)}  out_dir={args.out_dir}")

    encoder, head = build_models(device, args.n_classes, sensor=sensor)
    if args.ckpt:
        head.load_state_dict(torch.load(args.ckpt, map_location=device))
    head.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    saved = 0
    for i in range(len(ds)):
        sample = ds[i]
        ev, gt_boxes, gt_cls = _sample_to_tensors(sample, device)
        H, pos = encoder(ev)
        preds = head(H)
        pb, ps, pl = postprocess(preds, pos, score_thr=args.score_thr)

        gt_xyxy = None
        gt_labels = None
        if gt_boxes.shape[0]:
            gx, gy, gw, gh = gt_boxes.unbind(1)
            gt_xyxy = torch.stack([gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2], 1).cpu().numpy()
            gt_labels = gt_cls.cpu().numpy()

        rec_name = sample.get("name", f"sample_{i:05d}")
        win_idx = sample.get("window_idx", i)
        out_path = os.path.join(args.out_dir, rec_name, f"win_{win_idx:05d}.png")
        save_detection_frame(
            out_path, ev, sw, sh, pb, pl, ps,
            gt_boxes=gt_xyxy, gt_labels=gt_labels,
        )
        saved += 1
        if saved % 20 == 0:
            print(f"[test] saved {saved}/{len(ds)} frames ...")

    print(f"[test] done — {saved} frames in {args.out_dir}")


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
    print(f"[smoke] H (per-node, bundled): {tuple(H.shape)}  expected [N, {D}]")
    preds = head(H)
    boxes, scores, labels = postprocess(preds, pos, score_thr=args.score_thr)
    print(f"[smoke] boxes after NMS: {boxes.shape[0]}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ev[:, 1], ev[:, 2], s=1, c=ev[:, 0], cmap="viridis", alpha=0.4)
    for b in boxes.cpu().numpy():
        x1, y1, x2, y2 = b
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="red", lw=1.0))
    ax.set_title(f"GVFA-YOLO smoke — {boxes.shape[0]} boxes / {ev.shape[0]} events")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"[smoke] saved -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["smoke", "train", "eval", "test"], nargs="?", default="smoke")
    ap.add_argument("--dataset", choices=["prophesee", "txt"], default="prophesee")
    ap.add_argument("--data_dir", default=None,
                    help="Folder with Prophesee *_td.dat + *_bbox.npy pairs")
    ap.add_argument("--val_dir", default=None,
                    help="Validation folder (Prophesee), used during training")
    ap.add_argument("--events", default="events_filtered.txt",
                    help="Legacy txt events file (dataset=txt or smoke)")
    ap.add_argument("--labels", default=None, help="Legacy bbox .npy (dataset=txt)")
    ap.add_argument("--window_ms", type=float, default=WINDOW_MS)
    ap.add_argument("--max_events", type=int, default=40000)
    ap.add_argument("--max_recordings", type=int, default=None,
                    help="Limit number of Prophesee recordings (debug)")
    ap.add_argument("--max_windows", type=int, default=None,
                    help="Limit total windows across dataset (debug)")
    ap.add_argument("--n_classes", type=int, default=N_CLASSES)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--ckpt", default="checkpoints/head.pt")
    ap.add_argument("--out", default="boxes_on_events.png")
    ap.add_argument("--out_dir", default="runs/test_frames",
                    help="Output folder for test mode frame PNGs")
    args = ap.parse_args()

    if args.mode == "smoke":
        smoke(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        test_and_save_frames(args)


if __name__ == "__main__":
    main()
