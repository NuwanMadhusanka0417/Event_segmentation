"""
Pluggable event-camera DETECTION dataset adapters.

Contract -- every dataset yields, per event window:
    events : (N, 4) float32  [t_us, x, y, p]            (p in {0,1})
    boxes  : (M, 5) float32  [class, xc, yc, w, h]      (pixels; class in [0,N_CLASSES))

Provided here:
  * PropheseeDetectionDataset : Prophesee Gen1 Automotive Detection (*.td.dat + *_bbox.npy)
  * Gen1ETraMDataset            : legacy single-file txt + npy adapter
  * load_unlabeled              : forward-pass smoke test only
"""

import os
import numpy as np
from torch.utils.data import Dataset

from model import WINDOW_MS, MS_TO_US, N_CLASSES
from prophesee_io import (
    find_prophesee_pairs,
    load_prophesee_recording,
    boxes_in_window,
    PROPHESEE_CLASS_NAMES,
    PROPHESEE_SENSOR_W,
    PROPHESEE_SENSOR_H,
)


# --------------------------------------------------------------------------- #
#                           raw event stream loader                           #
# --------------------------------------------------------------------------- #
def load_events_txt(path, t_scale_to_us=True):
    """Load a space-separated event file: timestamp x y polarity."""
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
    """Split an event array into consecutive time windows. Yields (start_us, ev)."""
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
    """Prophesee Gen1 detection: paired *_td.dat + *_bbox.npy in one folder.

    Scans `data_dir` recursively for recording pairs, slices each into
    `window_ms` time windows, and returns GT boxes active in each window.
    """

    CLASS_NAMES = PROPHESEE_CLASS_NAMES

    def __init__(self, data_dir, window_ms=WINDOW_MS, max_events=40000,
                 max_recordings=None, max_windows=None):
        self.data_dir = data_dir
        self.window_ms = window_ms
        self.max_events = max_events
        self.pairs = find_prophesee_pairs(data_dir)
        if max_recordings is not None:
            self.pairs = self.pairs[:max_recordings]
        if not self.pairs:
            raise FileNotFoundError(
                f"No *_td.dat + *_bbox.npy pairs found under {data_dir}. "
                "Extract train_a.7z / val_a.7z first."
            )

        self._cache = {}
        self._flat = []
        self.sensor = (PROPHESEE_SENSOR_W, PROPHESEE_SENSOR_H)
        self._build_index(max_windows)

    def _load_recording(self, rec_i):
        if rec_i in self._cache:
            return self._cache[rec_i]
        dat_path, bbox_path = self.pairs[rec_i]
        events, boxes, sensor, name = load_prophesee_recording(dat_path, bbox_path)
        windows = list(slice_windows(events, self.window_ms))
        rec = dict(events=events, boxes=boxes, sensor=sensor, name=name, windows=windows)
        self._cache[rec_i] = rec
        self.sensor = sensor
        return rec

    def _build_index(self, max_windows):
        for rec_i in range(len(self.pairs)):
            rec = self._load_recording(rec_i)
            for win_i in range(len(rec["windows"])):
                self._flat.append((rec_i, win_i))
                if max_windows is not None and len(self._flat) >= max_windows:
                    return

    def __len__(self):
        return len(self._flat)

    def __getitem__(self, i):
        rec_i, win_i = self._flat[i]
        rec = self._load_recording(rec_i)
        w_start, ev = rec["windows"][win_i]
        if ev.shape[0] > self.max_events:
            sel = np.linspace(0, ev.shape[0] - 1, self.max_events).astype(int)
            ev = ev[sel]
        w_end = w_start + self.window_ms * MS_TO_US
        boxes = boxes_in_window(rec["boxes"], w_start, w_end)
        return {
            "events": ev,
            "boxes": boxes.astype(np.float32),
            "sensor": rec["sensor"],
            "name": rec["name"],
            "window_idx": win_i,
        }


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
    """Factory: build dataset from CLI args."""
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
        )
    if not args.events:
        raise ValueError("--events is required for txt dataset")
    return Gen1ETraMDataset(args.events, label_path=args.labels,
                            window_ms=args.window_ms,
                            max_events=args.max_events)


# --------------------------------------------------------------------------- #
#                       unlabeled forward-only smoke test                       #
# --------------------------------------------------------------------------- #
def load_unlabeled(path="events_filtered.txt", window_ms=WINDOW_MS, max_events=40000):
    events = load_events_txt(path)
    first = next(slice_windows(events, window_ms), (0.0, events[:0]))[1]
    if first.shape[0] > max_events:
        sel = np.linspace(0, first.shape[0] - 1, max_events).astype(int)
        first = first[sel]
    return first
