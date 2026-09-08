"""Training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from hdems.data.dsec import DSECDataset
from hdems.data.evimo import EVIMODataset
from hdems.losses.flow import epe_loss
from hdems.losses.seg import seg_loss
from hdems.models.hdems import HDEMS


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
        )
    raise ValueError(f"Unknown dataset: {name}")


def train_one_epoch(
    model: HDEMS,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
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
            loss = seg_loss(out["seg_logits"], batch["mask"].to(device).long())

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HD-EMS")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = HDEMS(cfg).to(device)
    print(f"Trainable parameters: {model.num_trainable_params:,}")

    train_cfg = cfg.get("train", {})
    task = train_cfg.get("task", "flow")

    try:
        dataset = build_dataset(cfg)
        if len(dataset) == 0:
            raise FileNotFoundError("Dataset is empty")
        print(f"Training samples: {len(dataset)}")
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

    for epoch in range(train_cfg.get("epochs", 100)):
        loss = train_one_epoch(model, loader, optimizer, device, task)
        print(f"Epoch {epoch + 1}: loss={loss:.4f}")


if __name__ == "__main__":
    main()
