"""EV-IMO / EVIMO2 segmentation dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from hdems.data.evimo2_reader import (
    build_sample_index,
    find_cached_samples,
    load_frame_sample,
    load_meta,
)


class EVIMODataset(Dataset):
    """EVIMO2 segmentation dataset.

    Layout under ``root``::

        root/train/scene_name/dataset_mask.npz   # raw sequences
        root/eval/scene_name/...
        root/train/sample.pt                     # optional cached shards

    Each sample provides ``surface`` (2, H, W) and ``mask`` (H, W).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        version: str | None = None,
        *,
        height: int = 480,
        width: int = 640,
        window_ms: float = 50.0,
        decay: float = 0.8,
        remap_mask: bool = True,
        use_classical_fallback: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.version = version  # unused, kept for config compatibility
        self.height = height
        self.width = width
        self.window_s = window_ms / 1000.0
        self.decay = decay
        self.remap_mask = remap_mask
        self.use_classical_fallback = use_classical_fallback

        self.cached: list[Path] = find_cached_samples(self.root, split)
        self.index: list[tuple[Path, int]] = (
            [] if self.cached else build_sample_index(self.root, split)
        )

    def __len__(self) -> int:
        return len(self.cached) if self.cached else len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.cached and not self.index:
            raise FileNotFoundError(
                f"No EVIMO2 samples under {self.root / self.split}. "
                "Place sequence folders in root/train and root/eval, "
                "or run: python scripts/prepare_evimo.py --root ../Data/EVIMO2"
            )

        if self.cached:
            return torch.load(self.cached[idx], weights_only=True)

        seq_dir, frame_idx = self.index[idx]
        frame = load_meta(seq_dir)["frames"][frame_idx]
        return load_frame_sample(
            seq_dir,
            frame,
            out_height=self.height,
            out_width=self.width,
            window_s=self.window_s,
            decay=self.decay,
            remap_mask=self.remap_mask,
            use_classical_fallback=self.use_classical_fallback,
        )
