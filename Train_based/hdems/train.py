"""Training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from hdems.data.dsec import DSECDataset
from hdems.data.evimo import EVIMODataset
from hdems.losses.flow import epe_loss
from hdems.losses.seg import dice_loss, seg_loss
from hdems.models.hdems import HDEMS
from hdems.seg_features import event_pixel_mask


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, split: str | None = None):
    ds_cfg = cfg.get("dataset", {})
    name = ds_cfg.get("name", "dsec")
    split = split or ds_cfg.get("split", "train")

    if name == "dsec":
        return DSECDataset(ds_cfg["root"], split)
    if name == "evimo":
        return EVIMODataset(
            ds_cfg["root"],
            split,
            ds_cfg.get("version"),
            height=ds_cfg.get("height", 480),
            width=ds_cfg.get("width", 640),
            window_ms=ds_cfg.get("window_ms", 50.0),
            decay=cfg.get("time_surface", {}).get("decay", 0.8),
            remap_mask=ds_cfg.get("remap_mask", True),
            use_classical_fallback=ds_cfg.get("use_classical_fallback", True),
            time_frames=ds_cfg.get("time_frames"),
        )
    raise ValueError(f"Unknown dataset: {name}")


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
            # and would otherwise flood the loss with trivial background.
            valid = event_pixel_mask(surface)
            if not valid.any():
                continue
            loss = seg_loss(logits, target.masked_fill(~valid, 255))
            if use_dice:
                # Dice counters heavy background/foreground imbalance.
                loss = loss + dice_loss(logits, target, num_classes, mask=valid)

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
    parser.add_argument("--event-combine", type=str, choices=["bind", "bundle", "concat"], default=None,
                        help="Override velocity.event_combine (cnn head).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.axis_combine or args.event_combine:                 # CLI overrides config
        vel = dict(cfg.get("velocity", {}))
        if args.axis_combine:
            vel["axis_combine"] = args.axis_combine
        if args.event_combine:
            vel["event_combine"] = args.event_combine
        cfg["velocity"] = vel
    axis = cfg.get("velocity", {}).get("axis_combine", "bind")
    event = cfg.get("velocity", {}).get("event_combine", "bind")
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
            dataset = Subset(dataset, list(range(max_n)))
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

    # Checkpoints: train.out_dir, else checkpoints/[head]_[axis]_[event] so the
    # combine is recorded in the path (last.pt / best.pt live inside).
    head = cfg.get("segmentation", {}).get("head", "cnn")
    ckpt_dir = Path(train_cfg.get("out_dir", f"checkpoints/{head}_{axis}_{event}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    use_dice = bool(train_cfg.get("use_dice", False))
    num_classes = cfg.get("segmentation", {}).get("num_classes", 16)

    for epoch in range(train_cfg.get("epochs", 100)):
        loss = train_one_epoch(model, loader, optimizer, device, task,
                               use_dice=use_dice, num_classes=num_classes)
        print(f"Epoch {epoch + 1}: loss={loss:.4f}")

        # Save after every epoch: always refresh last.pt, keep best.pt too.
        ckpt = {"model": model.state_dict(), "epoch": epoch + 1,
                "loss": loss, "task": task,
                "axis_combine": axis, "event_combine": event}
        torch.save(ckpt, ckpt_dir / "last.pt")
        if loss < best_loss:
            best_loss = loss
            torch.save(ckpt, ckpt_dir / "best.pt")

    print(f"Saved checkpoints to {ckpt_dir.resolve()} "
          f"(last.pt, best.pt @ loss={best_loss:.4f})")


if __name__ == "__main__":
    main()
