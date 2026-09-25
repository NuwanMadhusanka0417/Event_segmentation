#!/usr/bin/env python3
"""Fit / train a segmentation head on the frozen HD-EMS front-end (EVIMO2).

Single entry point for every head:
  ridge, prototype -> closed-form fit (no backprop)
  cnn, motion      -> backprop training; the best epoch by val mIoU is kept
All heads save to checkpoints/[head]_[axis]_[event]_[N].pt (unless --out).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdems.data.labels import LABEL_MODES, num_classes_for, resolve_label_mode
from hdems.eval import build_dataset, evaluate_segmentation, load_config
from hdems.train import train_one_epoch
from hdems.feature_extract import (
    accumulate_feature_mean,
    accumulate_paper_feature_mean,
    extract_flat_batch,
    extract_paper_flat_batch,
)
from hdems.metrics import mean_iou
from hdems.models.hdems import HDEMS
from hdems.ridge_fit import (
    RidgeFitResult,
    fit_ridge_sklearn,
    fit_ridge_streaming,
    save_ridge_weights,
)
from hdems.ridge_head import RidgeHead
from hdems.feature_extract import extract_phi
from hdems.models.prototype_head import PrototypeHead, fit_prototypes, save_prototypes
from hdems.vsa.velocity import EVENT_COMBINES


def _cap_dataset(ds, max_samples: int | None):
    if max_samples is None or max_samples <= 0 or max_samples >= len(ds):
        return ds
    return Subset(ds, list(range(max_samples)))


@torch.no_grad()
def collect_batches(
    model: HDEMS,
    loader: DataLoader,
    device: torch.device,
    *,
    feature_mean: torch.Tensor,
    mean_center: bool,
    motion_features: bool,
    paper: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch in loader:
        if paper:
            x, y = extract_paper_flat_batch(
                model,
                batch["surface"].to(device),
                batch["mask"].to(device),
                feature_mean=feature_mean,
                mean_center=mean_center,
            )
        else:
            x, y = extract_flat_batch(
                model,
                batch["surface"].to(device),
                batch["mask"].to(device),
                feature_mean=feature_mean,
                mean_center=mean_center,
                motion_features=motion_features,
            )
        if x.numel():
            batches.append((x.cpu(), y.cpu()))
    return batches


@torch.no_grad()
def eval_ridge_miou(
    model: HDEMS,
    head: RidgeHead,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    paper: bool = False,
) -> float:
    head.eval()
    model.eval()
    scores: list[float] = []
    for batch in loader:
        surface = batch["surface"].to(device)
        mask = batch["mask"].to(device).long()
        if paper:
            feats, _ = model.paper_features(surface)
            logits = head.logits_from_features(feats)
        else:
            phi = extract_phi(model, surface)
            logits = head(phi, surface=surface)
        pred = logits.argmax(dim=1)
        scores.append(mean_iou(pred[0], mask[0], num_classes))
    return sum(scores) / max(len(scores), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit or train a segmentation head")
    ap.add_argument("--config", type=str, default="configs/evimo_seg.yaml")
    ap.add_argument("--out", type=str, default=None,
                    help="Output .pt. Default: "
                         "checkpoints/[head]_[feature]_[axis]_[event]_[label_mode]_[N].pt")
    ap.add_argument("--head", type=str, choices=["ridge", "prototype", "cnn", "motion"],
                    default=None,
                    help="ridge/prototype: closed-form fit; cnn/motion: backprop training "
                         "(overrides segmentation.head).")
    ap.add_argument("--axis-combine", type=str, choices=["bind", "bundle"], default=None,
                    help="Vx,Vy combine (overrides velocity.axis_combine).")
    ap.add_argument("--event-combine", type=str, choices=list(EVENT_COMBINES), default=None,
                    help="Event HV + velocity combine: bind | bundle | bindbundle ((X o Mv) + Mv) "
                         "| concat (overrides velocity.event_combine).")
    ap.add_argument("--event-feature", type=str, choices=["phi", "f"], default=None,
                    help="Event HV fused with velocity: phi (bundled neighbourhood field) | "
                         "f (VFA descriptor F0) (overrides velocity.event_feature).")
    ap.add_argument("--label-mode", type=str, choices=list(LABEL_MODES), default=None,
                    help="motion = pose-derived moving (Option A); objects = per-object id "
                         "(Option B); tracked = legacy mask>0 baseline.")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["streaming", "sklearn"], default=None)
    ap.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Use only the first N train frames (smoke tests). Overrides ridge.max_train_samples.",
    )
    ap.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Use only the first N val frames for lambda selection. Overrides ridge.max_val_samples.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.label_mode:
        cfg["dataset"] = {**cfg.get("dataset", {}), "label_mode": args.label_mode}
    if args.axis_combine or args.event_combine or args.event_feature:   # CLI overrides config
        vel = dict(cfg.get("velocity", {}))
        if args.axis_combine:
            vel["axis_combine"] = args.axis_combine
        if args.event_combine:
            vel["event_combine"] = args.event_combine
        if args.event_feature:
            vel["event_feature"] = args.event_feature
        cfg["velocity"] = vel
    axis = cfg.get("velocity", {}).get("axis_combine", "bind")
    event = cfg.get("velocity", {}).get("event_combine", "bind")
    feature = cfg.get("velocity", {}).get("event_feature", "phi")
    print(f"[fit] event_feature={feature}  axis_combine={axis}  event_combine={event}")
    ridge_cfg = cfg.get("ridge", {})
    # Label mode fixes the class count so head size and labels always agree.
    label_mode = resolve_label_mode(cfg)
    num_classes = num_classes_for(label_mode, cfg.get("segmentation", {}).get("num_classes", 32))
    cfg["segmentation"] = {**cfg.get("segmentation", {}), "num_classes": num_classes}
    seg_cfg = cfg.get("segmentation", {})
    print(f"[fit] label_mode={label_mode}  num_classes={num_classes}")
    head_type = (args.head or seg_cfg.get("head", "ridge")).lower()
    if head_type not in ("ridge", "prototype", "cnn", "motion"):
        raise SystemExit(f"unknown head {head_type!r} (ridge|prototype|cnn|motion)")
    trainable = head_type in ("cnn", "motion")
    mean_center = bool(seg_cfg.get("ridge_mean_center", True))
    motion_features = bool(seg_cfg.get("ridge_motion_features", True))
    # Paper mode: multi-time surfaces -> fit on the two-time cost-volume features
    # (Phi combined with ego-residual velocity) instead of single-frame Phi.
    paper = bool(cfg.get("dataset", {}).get("time_frames"))
    if head_type == "prototype":
        if not paper:
            raise SystemExit("prototype head requires dataset.time_frames (paper mode)")
        mean_center = False   # cosine-centroid: normalization handles scale
    if paper:
        print(f"[fit] PAPER mode, head={head_type}: features = Phi (X) velocity code")
    imbalance = ridge_cfg.get("imbalance", "balanced")
    alphas = ridge_cfg.get("alphas", [1e-3, 1e-1, 1.0, 10.0, 100.0])
    backend = args.backend or ridge_cfg.get("backend", "streaming")
    bg_ratio = float(ridge_cfg.get("bg_subsample_ratio", 0.2))

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if not trainable:                        # frozen feature extractor for ridge/prototype
        model = HDEMS({**cfg, "segmentation": {**seg_cfg, "head": "cnn"}}).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    train_ds = build_dataset(cfg, split=cfg.get("dataset", {}).get("split", "train"))
    val_split = ridge_cfg.get("val_split", "eval")
    val_ds = build_dataset(cfg, split=val_split)

    max_train = args.max_train_samples
    if max_train is None:
        max_train = ridge_cfg.get("max_train_samples")
    max_val = args.max_val_samples
    if max_val is None:
        max_val = ridge_cfg.get("max_val_samples")

    n_train_full = len(train_ds)
    n_val_full = len(val_ds)
    train_ds = _cap_dataset(train_ds, max_train)
    val_ds = _cap_dataset(val_ds, max_val)
    if len(train_ds) == 0:
        raise SystemExit("Train dataset empty — check dataset.root")
    if max_train or max_val:
        print(
            f"[ridge] sample cap: train {len(train_ds)}/{n_train_full}  "
            f"val {len(val_ds)}/{n_val_full}"
        )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # ---- trainable heads (cnn / motion): backprop, keep best epoch by val mIoU --
    if trainable:
        out_path = (Path(args.out) if args.out else
                    Path("checkpoints")
                    / f"{head_type}_{feature}_{axis}_{event}_{label_mode}_{len(train_ds)}.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        train_cfg = cfg.get("train", {})
        net = HDEMS({**cfg, "segmentation": {**seg_cfg, "head": head_type}}).to(device)
        print(f"[train] head={head_type}  trainable params: {net.num_trainable_params:,}")
        tr_loader = DataLoader(train_ds, batch_size=train_cfg.get("batch_size", 1),
                               shuffle=True, num_workers=train_cfg.get("num_workers", 4))
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                     lr=train_cfg.get("lr", 1e-4))
        use_dice = bool(train_cfg.get("use_dice", False))
        epochs = int(train_cfg.get("epochs", 30))

        best_score, best_state, best_epoch = float("-inf"), None, 0
        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(net, tr_loader, optimizer, device, "segmentation",
                                   use_dice=use_dice, num_classes=num_classes)
            if len(val_ds):
                score = evaluate_segmentation(net, val_loader, device, num_classes)["miou"]
                print(f"[train] epoch {epoch}/{epochs}  loss={loss:.4f}  val_mIoU={score:.4f}")
            else:                                   # no val split: lowest train loss wins
                score = -loss
                print(f"[train] epoch {epoch}/{epochs}  loss={loss:.4f}")
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        torch.save({"model": best_state, "task": "segmentation", "head": head_type,
                    "epoch": best_epoch, "val_miou": best_score if len(val_ds) else None,
                    "num_samples": len(train_ds), "num_classes": num_classes,
                    "axis_combine": axis, "event_combine": event,
                    "event_feature": feature, "label_mode": label_mode}, out_path)
        print(f"[train] saved {out_path}  best epoch={best_epoch}  score={best_score:.4f}")
        return

    if mean_center:
        print(f"[fit] computing feature mean on {len(train_ds)} train samples ...")
        if paper:
            feature_mean = accumulate_paper_feature_mean(model, train_loader, device)
        else:
            feature_mean = accumulate_feature_mean(
                model, train_loader, device, motion_features=motion_features,
            )
        print(f"[fit] feature_dim={feature_mean.numel()}  mean_center={mean_center}")
    else:
        feature_mean = None
        print(f"[fit] no mean-centering (head={head_type})")

    print("[ridge] collecting train pixels ...")
    train_batches = collect_batches(
        model, train_loader, device,
        feature_mean=feature_mean,
        mean_center=mean_center,
        motion_features=motion_features,
        paper=paper,
    )
    n_pix = sum(b[0].shape[0] for b in train_batches)
    print(f"[fit] train pixels: {n_pix}")

    # Output path: explicit --out, else [head]_[feature]_[axis]_[event]_[label_mode]_[N].pt
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (Path("checkpoints")
                    / f"{head_type}_{feature}_{axis}_{event}_{label_mode}_{len(train_ds)}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- prototype head: class-mean centroids, no alpha sweep -----------------
    if head_type == "prototype":
        protos = fit_prototypes(train_batches, num_classes)
        head = PrototypeHead(num_classes)
        head.set_prototypes(protos)
        miou = (eval_ridge_miou(model, head, val_loader, device, num_classes, paper=paper)
                if len(val_ds) else 0.0)
        save_prototypes(str(out_path), protos, num_classes,
                        extra={"val_miou": miou, "seed": args.seed,
                               "num_samples": len(train_ds),
                               "axis_combine": axis, "event_combine": event,
                               "event_feature": feature, "label_mode": label_mode})
        print(f"[proto] saved {out_path}  val_mIoU={miou:.4f}  "
              f"prototypes={tuple(protos.shape)}")
        return

    best_alpha = alphas[0]
    best_result: RidgeFitResult | None = None
    best_miou = -1.0

    for alpha in alphas:
        if backend == "sklearn":
            xs = torch.cat([b[0] for b in train_batches], dim=0).numpy()
            ys = torch.cat([b[1] for b in train_batches], dim=0).numpy()
            cw = "balanced" if imbalance == "balanced" else None
            result = fit_ridge_sklearn(
                xs, ys, num_classes=num_classes, alpha=float(alpha),
                class_weight=cw, feature_mean=feature_mean,
            )
            result.mean_center = mean_center
            result.motion_features = motion_features
        else:
            result = fit_ridge_streaming(
                train_batches,
                num_classes=num_classes,
                alpha=float(alpha),
                imbalance=imbalance,
                bg_subsample_ratio=bg_ratio,
                seed=args.seed,
                feature_mean=feature_mean,
                mean_center=mean_center,
                motion_features=motion_features,
            )

        head = RidgeHead(num_classes, mean_center=mean_center,
                         motion_features=(False if paper else motion_features))
        head.set_from_result(result)

        miou = (eval_ridge_miou(model, head, val_loader, device, num_classes, paper=paper)
                if len(val_ds) else 0.0)
        print(f"[ridge] alpha={alpha:g}  val_mIoU={miou:.4f}  backend={backend}")
        if miou > best_miou:
            best_miou = miou
            best_alpha = alpha
            best_result = result

    assert best_result is not None
    save_ridge_weights(
        str(out_path),
        best_result,
        extra={"val_miou": best_miou, "seed": args.seed, "backend": backend,
               "num_samples": len(train_ds),
               "axis_combine": axis, "event_combine": event,
               "event_feature": feature, "label_mode": label_mode},
    )
    print(f"[ridge] saved {out_path}  alpha={best_alpha:g}  val_mIoU={best_miou:.4f}  "
          f"W shape={tuple(best_result.weight.shape)}  imbalance={imbalance}")


if __name__ == "__main__":
    main()
