"""Causal 50 ms event windows for supervised FG/BG segmentation (clip-level)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from seg_model import GraphCfg, build_causal_graph

EVENT_EXTS = {".txt", ".csv", ".dat", ".npy"}


def list_event_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in EVENT_EXTS
    )
    if not files:
        raise FileNotFoundError(f"no event files in {root}")
    return files


def load_event_file(path: str | Path) -> dict:
    """Load whitespace-separated `t x y p [label]`. Returns sorted arrays."""
    path = Path(path)
    data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"{path}: expected >=4 columns, got {data.shape[1]}")
    t = data[:, 0].astype(np.float64)
    x = data[:, 1].astype(np.float64)
    y = data[:, 2].astype(np.float64)
    p = data[:, 3].astype(np.float64)
    labels = data[:, 4].astype(np.float64) if data.shape[1] >= 5 else None
    order = np.argsort(t, kind="stable")
    out = {
        "t": t[order], "x": x[order], "y": y[order], "p": p[order],
        "path": path, "clip": path.stem,
    }
    if labels is not None:
        out["label"] = labels[order].astype(np.int64)
    return out


def window_indices(
    t: np.ndarray,
    window_ms: float = 50.0,
    stride_ms: float | None = None,
) -> list[tuple[int, int, float, float]]:
    """Non-overlapping (or strided) causal windows. Returns (i0, i1, t0, t1)."""
    if len(t) == 0:
        return []
    w_s = window_ms * 1e-3
    s_s = w_s if stride_ms is None else stride_ms * 1e-3
    t0 = float(t[0])
    t_end = float(t[-1])
    windows = []
    start = t0
    while start <= t_end + 1e-12:
        stop = start + w_s
        i0 = int(np.searchsorted(t, start, side="left"))
        i1 = int(np.searchsorted(t, stop, side="left"))
        if i1 > i0:
            windows.append((i0, i1, start, stop))
        start += s_s
        if s_s <= 0:
            break
    return windows


def slice_events(ev: dict, i0: int, i1: int) -> dict:
    out = {
        "t": ev["t"][i0:i1].copy(),
        "x": ev["x"][i0:i1].copy(),
        "y": ev["y"][i0:i1].copy(),
        "p": ev["p"][i0:i1].copy(),
    }
    if "label" in ev:
        out["label"] = ev["label"][i0:i1].copy()
    return out


def fg_bg_ratio(labels: np.ndarray | None) -> tuple[int, int, float]:
    if labels is None or len(labels) == 0:
        return 0, 0, float("nan")
    n_fg = int((labels == 1).sum())
    n_bg = int((labels == 0).sum())
    ratio = n_fg / max(n_bg, 1)
    return n_fg, n_bg, ratio


def summarize_dir(directory: str | Path, window_ms: float = 50.0) -> None:
    files = list_event_files(directory)
    print(f"[data] {directory}: {len(files)} clip(s)  (split is by clip/file)")
    if len(files) == 1:
        print("[data] WARNING: single clip only — treat as dev/sanity, "
              "not a generalization result")
    for path in files:
        ev = load_event_file(path)
        wins = window_indices(ev["t"], window_ms=window_ms)
        labels = ev.get("label")
        n_fg, n_bg, ratio = fg_bg_ratio(labels)
        lab = "labeled" if labels is not None else "unlabeled"
        print(
            f"  {path.name}: {len(ev['t'])} events, {len(wins)} windows "
            f"({window_ms:.0f} ms), {lab}, fg={n_fg} bg={n_bg} fg/bg={ratio:.4f}"
        )


class EventWindowDataset(Dataset):
    """One item = one causal window graph (+ labels when present)."""

    def __init__(
        self,
        directory: str | Path,
        cfg: GraphCfg,
        window_ms: float = 50.0,
        stride_ms: float | None = None,
        require_labels: bool = True,
        min_events: int = 8,
    ):
        self.cfg = cfg
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.require_labels = require_labels
        self.min_events = min_events
        self.files = list_event_files(directory)
        self.index: list[tuple[int, int, int, float, float]] = []  # file,i0,i1,t0,t1
        self._cache: dict[int, dict] = {}

        for fi, path in enumerate(self.files):
            ev = load_event_file(path)
            if require_labels and "label" not in ev:
                raise ValueError(f"{path}: training requires a label column")
            self._cache[fi] = ev
            for i0, i1, t0, t1 in window_indices(ev["t"], window_ms, stride_ms):
                if (i1 - i0) >= min_events:
                    self.index.append((fi, i0, i1, t0, t1))

        if not self.index:
            raise RuntimeError(f"no windows with >= {min_events} events in {directory}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        fi, i0, i1, t0, t1 = self.index[idx]
        ev = self._cache[fi]
        sl = slice_events(ev, i0, i1)
        graph = build_causal_graph(sl, self.cfg)
        item = {
            "graph": graph,
            "clip": self.files[fi].stem,
            "window_index": idx,
            "file_window": (fi, i0, i1),
            "t_start_ms": (t0 - float(ev["t"][0])) * 1e3,
            "t0": t0,
            "t1": t1,
            "n_events": i1 - i0,
        }
        if "label" in sl:
            # Remap labels through the same time-sort used in build_causal_graph.
            order = graph["order"]
            item["label"] = torch.from_numpy(sl["label"][order].astype(np.float32))
        return item


def iter_windows_streaming(
    path: str | Path,
    cfg: GraphCfg,
    window_ms: float = 50.0,
    stride_ms: float | None = None,
    min_events: int = 1,
) -> Iterator[dict]:
    """Yield windows in time order without buffering all predictions."""
    ev = load_event_file(path)
    wins = window_indices(ev["t"], window_ms, stride_ms)
    for wi, (i0, i1, t0, t1) in enumerate(wins):
        if (i1 - i0) < min_events:
            continue
        sl = slice_events(ev, i0, i1)
        graph = build_causal_graph(sl, cfg)
        out = {
            "graph": graph,
            "clip": Path(path).stem,
            "window_index": wi,
            "t_start_ms": (t0 - float(ev["t"][0])) * 1e3,
            "t0": t0,
            "t1": t1,
            "has_labels": "label" in sl,
        }
        if "label" in sl:
            out["label"] = sl["label"][graph["order"]].astype(np.int64)
        yield out
