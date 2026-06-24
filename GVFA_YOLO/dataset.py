"""
Pluggable event-camera DETECTION dataset adapters.

Contract -- every dataset yields, per event window:
    events : (N, 4) float32  [t_us, x, y, p]            (p in {0,1})
    boxes  : (M, 5) float32  [class, xc, yc, w, h]      (pixels; class in [0,N_CLASSES))

Provided here:
  * DetectionWindowDataset : base class, slices a raw event stream into windows
                             and pairs each window with its GT boxes.
  * Gen1ETraMDataset       : Gen1 / eTraM-style adapter STUB.  Reads the raw
                             events, but box-label parsing is left as a clearly
                             marked TODO (point it at your real label files).
  * load_unlabeled(path, window_ms) : forward-pass smoke test only -- returns
                             the first window of events, NO labels, NO loss.

The Data/ folder shipped with this project contains ONLY raw event files
(no bounding-box annotations), so Gen1ETraMDataset cannot return real boxes
until you wire in label files -- see the TODO in `_load_boxes`.
"""

import os
import numpy as np
from torch.utils.data import Dataset

from model import WINDOW_MS, MS_TO_US, N_CLASSES, SENSOR_W, SENSOR_H


# --------------------------------------------------------------------------- #
#                           raw event stream loader                           #
# --------------------------------------------------------------------------- #
def load_events_txt(path, t_scale_to_us=True):
    """Load a space-separated event file: timestamp x y polarity.

    The reference files store the timestamp in SECONDS (float, e.g.
    1604805390.370436).  We convert to integer microseconds relative to the
    first event so all downstream code uses raw-us dt.
    """
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    t, x, y, p = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if t_scale_to_us:
        # detect seconds vs us: seconds-based timestamps span ~<1e3 over a clip
        span = t.max() - t.min()
        if span < 1e4:                      # looks like seconds
            t = (t - t.min()) * 1e6
        else:                               # already us
            t = t - t.min()
    t = t.astype(np.float64)
    events = np.stack([t, x, y, p.clip(0, 1)], axis=1).astype(np.float32)
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
    for a, b in zip(idx[:-1], idx[1:]):
        if b > a:
            yield float(edges[0]), events[a:b]


# --------------------------------------------------------------------------- #
#                              base dataset                                    #
# --------------------------------------------------------------------------- #
class DetectionWindowDataset(Dataset):
    """Base labeled-detection adapter.

    Subclasses implement `_load_boxes(window_start_us, window_end_us)` returning
    (M,5) [class,xc,yc,w,h] for the boxes active in that window.
    """

    def __init__(self, events_path, window_ms=WINDOW_MS, max_events=40000):
        self.events = load_events_txt(events_path)
        self.window_ms = window_ms
        self.max_events = max_events
        self.windows = list(slice_windows(self.events, window_ms))

    def _load_boxes(self, w_start_us, w_end_us):
        raise NotImplementedError

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w_start, ev = self.windows[i]
        # subsample very dense windows to keep the graph build tractable
        if ev.shape[0] > self.max_events:
            sel = np.linspace(0, ev.shape[0] - 1, self.max_events).astype(int)
            ev = ev[sel]
        w_end = w_start + self.window_ms * MS_TO_US
        boxes = self._load_boxes(w_start, w_end)
        return {"events": ev, "boxes": boxes.astype(np.float32)}


# --------------------------------------------------------------------------- #
#                       Gen1 / eTraM adapter (STUB)                            #
# --------------------------------------------------------------------------- #
class Gen1ETraMDataset(DetectionWindowDataset):
    """Gen1 / eTraM-style detection adapter.

    Gen1 and eTraM ship bounding boxes as a structured numpy array per recording
    (fields: ('t','x','y','w','h','class_id','confidence','track_id'), times in
    microseconds).  Wire `label_path` to that file to get real supervision.
    """

    # eTraM / Gen1 typical traffic-agent classes; override as needed.
    CLASS_NAMES = ["car", "pedestrian"][:N_CLASSES]

    def __init__(self, events_path, label_path=None, window_ms=WINDOW_MS, **kw):
        super().__init__(events_path, window_ms=window_ms, **kw)
        self.label_path = label_path
        self._boxes_raw = self._read_label_file(label_path)

    def _read_label_file(self, label_path):
        if label_path is None or not os.path.exists(label_path):
            return None
        # ---------------------------------------------------------------- #
        # TODO(user): parse your real Gen1/eTraM label file here.
        #   Gen1/eTraM .npy structured array example:
        #     bbox = np.load(label_path)
        #     return structured array with fields t,x,y,w,h,class_id (us, px,
        #     where x,y is the TOP-LEFT corner).
        # ---------------------------------------------------------------- #
        bbox = np.load(label_path, allow_pickle=True)
        return bbox

    def _load_boxes(self, w_start_us, w_end_us):
        if self._boxes_raw is None:
            # No labels wired -> empty GT.  Training on this is meaningless;
            # this stub exists so the interface is exercisable end-to-end.
            return np.zeros((0, 5), dtype=np.float32)
        b = self._boxes_raw
        # assume structured array with us timestamps + top-left x,y
        mask = (b["t"] >= w_start_us) & (b["t"] < w_end_us)
        b = b[mask]
        if b.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        xc = b["x"].astype(np.float32) + b["w"].astype(np.float32) / 2
        yc = b["y"].astype(np.float32) + b["h"].astype(np.float32) / 2
        cls = b["class_id"].astype(np.float32)
        return np.stack([cls, xc, yc, b["w"].astype(np.float32),
                         b["h"].astype(np.float32)], axis=1)


# --------------------------------------------------------------------------- #
#                       unlabeled forward-only smoke test                       #
# --------------------------------------------------------------------------- #
def load_unlabeled(path="events_filtered.txt", window_ms=WINDOW_MS, max_events=40000):
    """Return the FIRST window of an unlabeled event file for a forward-pass
    smoke test.  No labels, no loss, no mAP -- never used for training/eval.

    Returns: events (N,4) float32.
    """
    events = load_events_txt(path)
    first = next(slice_windows(events, window_ms), (0.0, events[:0]))[1]
    if first.shape[0] > max_events:
        sel = np.linspace(0, first.shape[0] - 1, max_events).astype(int)
        first = first[sel]
    return first
