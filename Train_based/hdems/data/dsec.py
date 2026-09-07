"""DSEC dataset loader (stub — requires prepared data)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class DSECDataset(Dataset):
    """DSEC optical flow dataset.

    Expects prepared data at ``root/`` with event and flow pairs.
    Run ``scripts/prepare_dsec.sh`` to download and preprocess.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        dt: int = 1,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.dt = dt
        self.samples: list[Path] = []
        if self.root.exists():
            self.samples = sorted((self.root / split).glob("*.pt"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.samples:
            raise FileNotFoundError(
                f"No DSEC samples found at {self.root / self.split}. "
                "Run scripts/prepare_dsec.sh first."
            )
        data = torch.load(self.samples[idx], weights_only=True)
        return data
