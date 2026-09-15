#!/usr/bin/env python3
"""Fit closed-form ridge readout on frozen HD-EMS features (EVIMO2 train split)."""

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

from hdems.eval import build_dataset, load_config
from hdems.feature_extract import accumulate_feature_mean, extract_flat_batch
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
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch in loader:
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
) -> float:
    head.eval()
    model.eval()
    scores: list[float] = []
    for batch in loader:
        surface = batch["surface"].to(device)
        mask = batch["mask"].to(device).long()
        phi = extract_phi(model, surface)
        logits = head(phi, surface=surface)
        pred = logits.argmax(dim=1)
        scores.append(mean_iou(pred[0], mask[0], num_classes))
    return sum(scores) / max(len(scores), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit ridge segmentation head")
    ap.add_argument("--config", type=str, default="configs/evimo_seg.yaml")
    ap.add_argument("--out", type=str, default="checkpoints/ridge_head.pt")
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
    ridge_cfg = cfg.get("ridge", {})
    seg_cfg = cfg.get("segmentation", {})
    num_classes = seg_cfg.get("num_classes", 32)
    mean_center = bool(seg_cfg.get("ridge_mean_center", True))
    motion_features = bool(seg_cfg.get("ridge_motion_features", True))
    imbalance = ridge_cfg.get("imbalance", "balanced")
    alphas = ridge_cfg.get("alphas", [1e-3, 1e-1, 1.0, 10.0, 100.0])
    backend = args.backend or ridge_cfg.get("backend", "streaming")
    bg_ratio = float(ridge_cfg.get("bg_subsample_ratio", 0.2))

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

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

    print(f"[ridge] computing feature mean on {len(train_ds)} train samples ...")
    feature_mean = accumulate_feature_mean(
        model, train_loader, device, motion_features=motion_features,
    )
    print(f"[ridge] feature_dim={feature_mean.numel()}  mean_center={mean_center}  "
          f"motion={motion_features}")

    print("[ridge] collecting train pixels ...")
    train_batches = collect_batches(
        model, train_loader, device,
        feature_mean=feature_mean,
        mean_center=mean_center,
        motion_features=motion_features,
    )
    n_pix = sum(b[0].shape[0] for b in train_batches)
    print(f"[ridge] train pixels: {n_pix}")

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

        head = RidgeHead(num_classes, mean_center=mean_center, motion_features=motion_features)
        head.set_from_result(result)

        miou = eval_ridge_miou(model, head, val_loader, device, num_classes) if len(val_ds) else 0.0
        print(f"[ridge] alpha={alpha:g}  val_mIoU={miou:.4f}  backend={backend}")
        if miou > best_miou:
            best_miou = miou
            best_alpha = alpha
            best_result = result

    assert best_result is not None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_ridge_weights(
        str(out_path),
        best_result,
        extra={"val_miou": best_miou, "seed": args.seed, "backend": backend},
    )
    print(f"[ridge] saved {out_path}  alpha={best_alpha:g}  val_mIoU={best_miou:.4f}  "
          f"W shape={tuple(best_result.weight.shape)}  imbalance={imbalance}")


if __name__ == "__main__":
    main()
