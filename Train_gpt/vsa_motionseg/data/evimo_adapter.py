"""EVIMO2 sequence adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from vsa_motionseg.data.event_dataset import EventDatasetAdapter, EventFrame


def find_sequences(root: Path, split: str) -> list[Path]:
    d = root / split
    if not d.exists():
        return []
    return sorted(
        p for p in d.iterdir() if p.is_dir() and (p / "dataset_mask.npz").exists()
    )


def load_meta(seq_dir: Path) -> dict[str, Any]:
    info = np.load(seq_dir / "dataset_info.npz", allow_pickle=True)
    meta = info["meta"].item()
    if not isinstance(meta, dict):
        raise ValueError(f"Unexpected meta in {seq_dir}")
    return meta


def sensor_size(meta: dict[str, Any], mask_shape: tuple[int, ...]) -> tuple[int, int]:
    inner = meta.get("meta", {})
    if inner.get("res_x") and inner.get("res_y"):
        return int(inner["res_y"]), int(inner["res_x"])
    return int(mask_shape[0]), int(mask_shape[1])


class EVIMO2Adapter(EventDatasetAdapter):
    def __init__(self, root: str | Path, split: str = "train", sequence: int = 0) -> None:
        self.root = Path(root)
        self.sequences = find_sequences(self.root, split)
        if not self.sequences:
            raise FileNotFoundError(f"No EVIMO2 sequences under {self.root / split}")
        self.seq_dir = self.sequences[min(sequence, len(self.sequences) - 1)]
        self.meta = load_meta(self.seq_dir)
        self.frames = self.meta.get("frames", [])
        self._load_event_arrays()

    def _load_event_arrays(self) -> None:
        t_path = self.seq_dir / "dataset_events_t.npy"
        if t_path.exists():
            self._t = np.load(t_path, mmap_mode="r").reshape(-1)
            self._xy = np.load(self.seq_dir / "dataset_events_xy.npy", mmap_mode="r")
            self._p = np.load(self.seq_dir / "dataset_events_p.npy", mmap_mode="r").reshape(-1)
        else:
            self._t = self._xy = self._p = None
        self._masks = np.load(self.seq_dir / "dataset_mask.npz")

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def timestamps(self) -> list[float]:
        return [float(f["ts"]) for f in self.frames]

    def events_in_window(self, t0: float, t1: float) -> torch.Tensor:
        if self._t is None or self._t.size == 0:
            return torch.zeros(0, 4)
        i0 = int(np.searchsorted(self._t, t0, side="left"))
        i1 = int(np.searchsorted(self._t, t1, side="right"))
        if i1 <= i0:
            return torch.zeros(0, 4)
        ev = np.stack(
            [
                self._t[i0:i1].astype(np.float64),
                self._xy[i0:i1, 0].astype(np.float64),
                self._xy[i0:i1, 1].astype(np.float64),
                self._p[i0:i1].astype(np.float64),
            ],
            axis=1,
        )
        return torch.from_numpy(ev)

    def get_frame(self, index: int) -> EventFrame:
        frame = self.frames[index]
        ts = float(frame["ts"])
        fid = int(frame["id"])
        key = f"mask_{fid:010d}"
        inst = None
        h, w = 240, 320
        if key in self._masks.files:
            m = self._masks[key]
            h, w = sensor_size(self.meta, m.shape)
            inst = torch.from_numpy((m.astype(np.int64) // 1000))

        return EventFrame(
            events=None,
            image_height=h,
            image_width=w,
            timestamp=ts,
            instance_masks=inst,
            motion_masks=inst.clone() if inst is not None else None,
            meta={"frame_id": fid, "seq_dir": str(self.seq_dir)},
        )


class DSECAdapterStub(EventDatasetAdapter):
    """Placeholder for future DSEC-MOTS integration."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("DSEC adapter: implement when dataset path is configured")

    def __len__(self) -> int:
        return 0

    def get_frame(self, index: int) -> EventFrame:
        raise NotImplementedError

    @property
    def timestamps(self) -> list[float]:
        return []
