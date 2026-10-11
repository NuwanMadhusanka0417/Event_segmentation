"""FLOPs, parameter count, and latency benchmark."""

from __future__ import annotations

import argparse
import time

import torch
import yaml

from hdems.models.hdems import HDEMS


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def count_params(model: HDEMS) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


@torch.no_grad()
def measure_latency(
    model: HDEMS,
    surface: torch.Tensor,
    device: torch.device,
    warmup: int = 10,
    repeats: int = 50,
) -> float:
    model.eval()
    surface = surface.to(device)
    for _ in range(warmup):
        model(surface)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(surface)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HD-EMS")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--H", type=int, default=120)
    parser.add_argument("--W", type=int, default=160)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = HDEMS(cfg).to(device)

    params = count_params(model)
    print(f"Parameters: {params['trainable']:,} trainable, {params['frozen']:,} frozen")

    surface = torch.randn(1, 2, args.H, args.W)
    ms = measure_latency(model, surface, device)
    print(f"Latency ({device}): {ms:.2f} ms/frame")


if __name__ == "__main__":
    main()
