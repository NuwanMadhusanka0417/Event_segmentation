"""Simple matplotlib dumps for qualitative results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_heatmap(path: Path, tensor: torch.Tensor, title: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = tensor.detach().cpu().float().numpy()
    if img.ndim == 3:
        img = img[0]
    plt.figure(figsize=(6, 4))
    plt.imshow(img, cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
