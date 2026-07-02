"""
Pluggable event-camera DETECTION dataset adapters.

Lightweight Prophesee loading: index windows from mmap metadata only;
load one event window at a time on __getitem__ (no full .dat in RAM).

Contract -- every dataset yields, per event window:
    events : (N, 4) float32  [t_us, x, y, p]
    boxes  : (M, 5) float32  [class, xc, yc, w, h]
"""

import os
import numpy as np
from torch.utils.data import Dataset

from model import WINDOW_MS, MS_TO_US, N_CLASSES
from prophesee_io import (
    find_prophesee_pairs,
    probe_dat,
    window_ranges,
    DatWindowReader,
    load_prophesee_bbox,
    boxes_in_window,
    PROPHESEE_CLASS_NAMES,
    PROPHESEE_SENSOR_W,
    PROPHESEE_SENSOR_H,
)


# --------------------------------------------------------------------------- #
#                           raw event stream loader                           #
# --------------------------------------------------------------------------- #
def load_events_txt(path, t_scale_to_us=True):
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    t, x, y, p = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if t_scale_to_us:
        span = t.max() - t.min()
        if span < 1e4:
            t = (t - t.min()) * 1e6
        else:
            t = t - t.min()
    events = np.stack([t.astype(np.float64), x, y, p.clip(0, 1)], axis=1).astype(np.float32)
    return events


def slice_windows(events, window_ms=WINDOW_MS):
    if events.shape[0] == 0:
        return
    win_us = window_ms * MS_TO_US
    t = events[:, 0]
    t0 = t[0]
    edges = np.arange(t0, t[-1] + win_us, win_us)
    idx = np.searchsorted(t, edges)
    for wi, (a, b) in enumerate(zip(idx[:-1], idx[1:])):
        if b > a:
            yield float(edges[wi]), events[a:b]


# --------------------------------------------------------------------------- #
#                       Prophesee Gen1 Automotive Detection                  #
# --------------------------------------------------------------------------- #
class PropheseeDetectionDataset(Dataset):
    """Prophesee Gen1 detection — streaming one window at a time.

    At init: probes each .dat (mmap first/last timestamp only) to build a
    lightweight window index.  On __getitem__: loads just that window's events
    via mmap + binary search; bbox .npy cached per recording (small).
    """

    CLASS_NAMES = PROPHESEE_CLASS_NAMES

    def __init__(self, data_dir, window_ms=WINDOW_MS, max_events=40000,
                 max_recordings=None, max_windows=None, reader_cache_size=1):
        self.data_dir = data_dir
        self.window_ms = window_ms
        self.max_events = max_events
        self.reader_cache_size = max(1, reader_cache_size)
        self.pairs = find_prophesee_pairs(data_dir)
        if max_recordings is not None:
            self.pairs = self.pairs[:max_recordings]
        if not self.pairs:
            raise FileNotFoundError(
                f"No *_td.dat + *_bbox.npy pairs found under {data_dir}. "
                "Extract train_a.7z / val_a.7z first."
            )

        self._rec_meta = []
        self._flat = []
        self._bbox_cache = {}
        self._reader_cache = {}
        self._reader_order = []
        self.sensor = (PROPHESEE_SENSOR_W, PROPHESEE_SENSOR_H)
        self._build_index(max_windows)

    def _build_index(self, max_windows):
        """Build window list from mmap probes only — no event arrays loaded."""
        for dat_path, bbox_path in self.pairs:
            meta = probe_dat(dat_path)
            meta["dat_path"] = dat_path
            meta["bbox_path"] = bbox_path
            rec_i = len(self._rec_meta)
            self._rec_meta.append(meta)
            self.sensor = meta["sensor"]

            for win_i, t_start, t_end in window_ranges(
                    meta["t0"], meta["t_last"], self.window_ms):
                self._flat.append((rec_i, win_i, t_start, t_end))
                if max_windows is not None and len(self._flat) >= max_windows:
                    return

    def _get_boxes(self, rec_i):
        if rec_i not in self._bbox_cache:
            meta = self._rec_meta[rec_i]
            self._bbox_cache[rec_i] = load_prophesee_bbox(
                meta["bbox_path"], t0_us=meta["t0"])
        return self._bbox_cache[rec_i]

    def _get_reader(self, rec_i):
        if rec_i in self._reader_cache:
            self._reader_order.remove(rec_i)
            self._reader_order.append(rec_i)
            return self._reader_cache[rec_i]

        while len(self._reader_cache) >= self.reader_cache_size:
            old = self._reader_order.pop(0)
            r = self._reader_cache.pop(old, None)
            if r is not None:
                r.close()

        meta = self._rec_meta[rec_i]
        reader = DatWindowReader(meta["dat_path"])
        self._reader_cache[rec_i] = reader
        self._reader_order.append(rec_i)
        return reader

    def __len__(self):
        return len(self._flat)

    def __getitem__(self, i):
        rec_i, win_i, t_start, t_end = self._flat[i]
        meta = self._rec_meta[rec_i]
        reader = self._get_reader(rec_i)
        ev = reader.load_window(t_start, t_end)

        if ev.shape[0] > self.max_events:
            sel = np.linspace(0, ev.shape[0] - 1, self.max_events).astype(int)
            ev = ev[sel]

        boxes = boxes_in_window(self._get_boxes(rec_i), t_start, t_end)
        return {
            "events": ev,
            "boxes": boxes.astype(np.float32),
            "sensor": meta["sensor"],
            "name": meta["stem"],
            "window_idx": win_i,
        }

    def close(self):
        for r in self._reader_cache.values():
            r.close()
        self._reader_cache.clear()
        self._reader_order.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
#                              base dataset (legacy txt)                       #
# --------------------------------------------------------------------------- #
class DetectionWindowDataset(Dataset):
    def __init__(self, events_path, window_ms=WINDOW_MS, max_events=40000):
        self.events = load_events_txt(events_path)
        self.window_ms = window_ms
        self.max_events = max_events
        self.windows = list(slice_windows(self.events, window_ms))
        self.sensor = (346, 260)

    def _load_boxes(self, w_start_us, w_end_us):
        raise NotImplementedError

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w_start, ev = self.windows[i]
        if ev.shape[0] > self.max_events:
            sel = np.linspace(0, ev.shape[0] - 1, self.max_events).astype(int)
            ev = ev[sel]
        w_end = w_start + self.window_ms * MS_TO_US
        boxes = self._load_boxes(w_start, w_end)
        return {"events": ev, "boxes": boxes.astype(np.float32), "sensor": self.sensor}


class Gen1ETraMDataset(DetectionWindowDataset):
    CLASS_NAMES = ["car", "pedestrian"][:N_CLASSES]

    def __init__(self, events_path, label_path=None, window_ms=WINDOW_MS, **kw):
        super().__init__(events_path, window_ms=window_ms, **kw)
        self._boxes_raw = self._read_label_file(label_path)

    def _read_label_file(self, label_path):
        if label_path is None or not os.path.exists(label_path):
            return None
        return np.load(label_path, allow_pickle=True)

    def _load_boxes(self, w_start_us, w_end_us):
        if self._boxes_raw is None:
            return np.zeros((0, 5), dtype=np.float32)
        return boxes_in_window(self._boxes_raw, w_start_us, w_end_us)


def make_dataset(args):
    if getattr(args, "dataset", "prophesee") == "prophesee":
        data_dir = args.data_dir
        if not data_dir:
            raise ValueError("--data_dir is required for Prophesee dataset")
        return PropheseeDetectionDataset(
            data_dir,
            window_ms=args.window_ms,
            max_events=args.max_events,
            max_recordings=getattr(args, "max_recordings", None),
            max_windows=getattr(args, "max_windows", None),
            reader_cache_size=getattr(args, "reader_cache_size", 1),
        )
    if not args.events:
        raise ValueError("--events is required for txt dataset")
    return Gen1ETraMDataset(args.events, label_path=args.labels,
                            window_ms=args.window_ms,
                            max_events=args.max_events)


def load_unlabeled(path="events_filtered.txt", window_ms=WINDOW_MS, max_events=40000):
    events = load_events_txt(path)
    first = next(slice_windows(events, window_ms), (0.0, events[:0]))[1]
    if first.shape[0] > max_events:
        sel = np.linspace(0, first.shape[0] - 1, max_events).astype(int)
        first = first[sel]
    return first
