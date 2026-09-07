"""EV-IMO / EV-IMO2 segmentation dataset loader (stub)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class EVIMODataset(Dataset):
    """EV-IMO moving-object segmentation dataset.

    Expects prepared data at ``root/``. Run ``scripts/prepare_evimo.sh``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        version: str = "evimo2",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.version = version
        self.samples: list[Path] = []
        if self.root.exists():
            self.samples = sorted((self.root / version / split).glob("*.pt"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.samples:
            raise FileNotFoundError(
                f"No EV-IMO samples at {self.root / self.version / self.split}. "
                "Run scripts/prepare_evimo.sh first."
            )
        return torch.load(self.samples[idx], weights_only=True)
