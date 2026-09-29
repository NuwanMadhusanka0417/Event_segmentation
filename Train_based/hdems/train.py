"""Training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from hdems.config import (
    apply_resolution_ratio,
    frontend_settings,
    head_settings,
    resolution_ratio_of,
)
from hdems.data.build import build_dataset as _build_dataset
from hdems.data.build import val_split_of
from hdems.data.labels import LABEL_MODES, num_classes_for, resolve_label_mode
from hdems.data.motion_labels import IGNORE_LABEL
from hdems.losses.flow import epe_loss
from hdems.losses.seg import dice_loss, seg_loss
from hdems.models.hdems import HDEMS
from hdems.seg_features import score_pixel_mask
from hdems.vsa.velocity import EVENT_COMBINES


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:     # config comments are UTF-8
        return yaml.safe_load(f)


build_dataset = _build_dataset      # re-exported: scripts import it from here


def train_one_epoch(
    model: HDEMS,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    *,
    use_dice: bool = False,
    num_classes: int = 16,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        surface = batch["surface"].to(device)
        optimizer.zero_grad()
        out = model(surface, task=task)

        if task == "flow":
            loss = epe_loss(out["flow"], batch["flow"].to(device))
        else:
            logits = out["seg_logits"]
            target = batch["mask"].to(device).long()
            # Train on EVENT pixels only: pixels with no events carry no evidence
            # and would otherwise flood the loss with trivial background. Only events
            # near the label time (score_pixel_mask): older ones are the trail a
            # mover leaves, which the label at ts calls "static".
            # Ignore label 255 = ambiguous object speed or mask boundary band.
            valid = score_pixel_mask(batch, surface) & (target != IGNORE_LABEL)
            if not valid.any():
                continue
            loss = seg_loss(logits, target.masked_fill(~valid, IGNORE_LABEL))
            if use_dice:
                # Dice counters heavy background/foreground imbalance. Pass a target
                # with ignore pixels zeroed AND masked out, so clamp() inside dice
                # cannot turn a 255 into a foreground pixel.
                loss = loss + dice_loss(
                    logits, target.masked_fill(~valid, 0), num_classes, mask=valid,
                )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HD-EMS")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Train on only the first N dataset frames (smoke test).",
    )
    parser.add_argument("--axis-combine", type=str, choices=["bind", "bundle"], default=None,
                        help="Override velocity.axis_combine (cnn head).")
    parser.add_argument("--event-combine", type=str, choices=list(EVENT_COMBINES), default=None,
                        help="Override velocity.event_combine (cnn head).")
    parser.add_argument("--event-feature", type=str, choices=["phi", "f"], default=None,
                        help="Override velocity.event_feature: phi | f (cnn head).")
    parser.add_argument("--label-mode", type=str, choices=list(LABEL_MODES), default=None,
                        help="motion = pose-derived moving (Option A); objects = per-object id "
                             "(Option B); tracked = legacy mask>0 baseline.")
    parser.add_argument("--resolution-ratio", type=int, default=None,
                        help="Process at 1/R resolution: 1 full, 2 half, 4 quarter "
                             "(overrides dataset.resolution_ratio).")
    args = parser.parse_args()

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
    cfg = apply_resolution_ratio(cfg, args.resolution_ratio, verbose=True)
    ratio = resolution_ratio_of(cfg)
    frontend = frontend_settings(cfg)
    axis = cfg.get("velocity", {}).get("axis_combine", "bind")
    event = cfg.get("velocity", {}).get("event_combine", "bind")
    feature = cfg.get("velocity", {}).get("event_feature", "phi")
    # The label mode fixes the class count, so the head can never disagree with the labels.
    label_mode = resolve_label_mode(cfg)
    n_classes = num_classes_for(label_mode, cfg.get("segmentation", {}).get("num_classes", 32))
    cfg["segmentation"] = {**cfg.get("segmentation", {}), "num_classes": n_classes}
    print(f"label_mode={label_mode}  num_classes={n_classes}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = HDEMS(cfg).to(device)
    print(f"Trainable parameters: {model.num_trainable_params:,}")

    train_cfg = cfg.get("train", {})
    task = train_cfg.get("task", "flow")

    try:
        dataset = build_dataset(cfg)
        if len(dataset) == 0:
            raise FileNotFoundError("Dataset is empty")
        n_full = len(dataset)
        max_n = args.max_samples or train_cfg.get("max_samples")
        if max_n is not None and 0 < max_n < n_full:
            # evenly spread, not the first N: those are the static opening of each recording
            step = [round(i * (n_full - 1) / max(max_n - 1, 1)) for i in range(max_n)]
            dataset = Subset(dataset, sorted(set(step)))
        print(f"Training samples: {len(dataset)}" + (f" (of {n_full})" if len(dataset) != n_full else ""))
        loader = DataLoader(
            dataset,
            batch_size=train_cfg.get("batch_size", 4),
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 4),
        )
    except FileNotFoundError as e:
        print(f"Dataset not ready: {e}")
        print("Place EVIMO2 sequences under root/train/ and root/eval/.")
        return

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.get("lr", 1e-4),
    )

    # Checkpoints: train.out_dir, else checkpoints/[head]_[feature]_[axis]_[event]_[label_mode]
    # so the settings are recorded in the path (last.pt / best.pt live inside).
    head = cfg.get("segmentation", {}).get("head", "cnn")
    ckpt_dir = Path(train_cfg.get(
        "out_dir", f"checkpoints/{head}_{feature}_{axis}_{event}_{label_mode}_r{ratio}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_dice = bool(train_cfg.get("use_dice", False))
    num_classes = n_classes

    # best.pt is chosen on the VALIDATION split (held-out train scenes), not on the
    # training loss -- the lowest training loss is just the most overfitted epoch.
    val_loader = None
    if task == "segmentation":
        from hdems.eval import evaluate_segmentation      # local: eval imports the model stack
        val_ds = build_dataset(cfg, val_split_of(cfg))
        max_val = int(train_cfg.get("max_val_samples", 150) or 0)
        if 0 < max_val < len(val_ds):
            pick = [round(i * (len(val_ds) - 1) / max(max_val - 1, 1)) for i in range(max_val)]
            val_ds = Subset(val_ds, sorted(set(pick)))
        print(f"Validation samples ({val_split_of(cfg)}): {len(val_ds)}")
        if len(val_ds):
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    best_score = float("-inf")

    for epoch in range(train_cfg.get("epochs", 100)):
        loss = train_one_epoch(model, loader, optimizer, device, task,
                               use_dice=use_dice, num_classes=num_classes)
        score, line = -loss, f"Epoch {epoch + 1}: loss={loss:.4f}"
        if val_loader is not None:
            m = evaluate_segmentation(model, val_loader, device, num_classes,
                                      label_mode=label_mode)
            score = m.get("fg_iou", m["miou"])
            line += f"  val_{'fgIoU' if 'fg_iou' in m else 'mIoU'}={score:.4f}"
        print(line)

        # Save after every epoch: always refresh last.pt, keep best.pt too.
        ckpt = {"model": model.state_dict(), "epoch": epoch + 1,
                "loss": loss, "task": task, "head": head,
                "val_score": score if val_loader is not None else None,
                "axis_combine": axis, "event_combine": event, "event_feature": feature,
                "label_mode": label_mode, "num_classes": n_classes,
                "resolution_ratio": ratio, "frontend": frontend,
                "head_config": head_settings(cfg)}
        torch.save(ckpt, ckpt_dir / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(ckpt, ckpt_dir / "best.pt")

    what = "val score" if val_loader is not None else "-train loss"
    print(f"Saved checkpoints to {ckpt_dir.resolve()} "
          f"(last.pt, best.pt @ {what}={best_score:.4f})")


if __name__ == "__main__":
    main()
