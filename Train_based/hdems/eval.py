"""Evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from hdems.data.dsec import DSECDataset
from hdems.data.evimo import EVIMODataset
from hdems.losses.flow import epe_loss
from hdems.models.hdems import HDEMS


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, split: str = "val"):
    ds_cfg = cfg.get("dataset", {})
    name = ds_cfg.get("name", "dsec")
    if name == "dsec":
        return DSECDataset(ds_cfg["root"], split)
    return EVIMODataset(ds_cfg["root"], split, ds_cfg.get("version", "evimo2"))


@torch.no_grad()
def evaluate_flow(model: HDEMS, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_epe = 0.0
    n = 0
    for batch in loader:
        surface = batch["surface"].to(device)
        target = batch["flow"].to(device)
        out = model(surface, task="flow")
        total_epe += epe_loss(out["flow"], target).item()
        n += 1
    return total_epe / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HD-EMS")
    parser.add_argument("--config", type=str, default="configs/dsec_flow.yaml")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = HDEMS(cfg).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])

    try:
        dataset = build_dataset(cfg, split="val")
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        epe = evaluate_flow(model, loader, device)
        print(f"Mean EPE: {epe:.4f}")
    except FileNotFoundError as e:
        print(f"Dataset not ready: {e}")


if __name__ == "__main__":
    main()
