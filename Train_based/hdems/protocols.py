"""Score predictions with a PUBLISHED protocol, so our numbers sit next to a paper's table.

hua2025 -- Hua, Yuan & Fermüller 2025 (arXiv:2507.14500), EVIMO2 IMO test sequences
13-00, 13-05, 14-03, 14-04, 14-05. Their metric, as we read the paper:

    IoU = |P ∩ G| / |P ∪ G|      counted over EVENTS, at full sensor resolution

P / G = the events predicted / labelled as belonging to a moving object. Only frames
that contain a moving object are scored, and the per-frame IoUs are averaged per
sequence. There is no ignore band and no "ambiguous speed" exclusion: every event in
the window counts. Our own eval differs on all of these (pixels at working resolution,
every frame, one pooled mean, boundary band and slow objects ignored).

From pixels to events: the head predicts at working resolution; the map is upsampled
(nearest) to the sensor and every event takes the label of its pixel. Each EVENT counts
once, so a pixel that fired three times weighs three times -- that is what "IoU over
events" means, and it weights busy edges more than our pixel IoU does.

Not stated in the paper, so configurable (config ``benchmark:``):
  event window  which events belong to a frame: the last w ms before the label time
  gt            which objects count as moving:
                  moving       pose speed >= motion_label.move_px (our training label)
                  moving_slow  also the slow ones (static_px < speed < move_px) that our
                               training ignores
                  tracked      every masked object -- includes the static table
Report the setting next to the number, and confirm it with the authors before
claiming a win.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from hdems.data.evimo2_reader import gt_key, load_meta
from hdems.data.motion_labels import MotionParams, sequence_motion
from hdems.grouping import GroupingParams, group_objects
from hdems.seg_features import score_pixel_mask

GT_MODES = ("moving", "moving_slow", "tracked")
PRED_SOURCES = ("fg", "objects")   # fg = the head's moving pixels | objects = after grouping


@dataclass(frozen=True)
class Benchmark:
    name: str
    citation: str
    # (sequence folder prefix, the paper's label, the paper's IoU in %)
    sequences: tuple[tuple[str, str, float], ...]
    split: str = "eval"

    def label_of(self, seq_name: str) -> str | None:
        return next((lab for pre, lab, _ in self.sequences if seq_name.startswith(pre)), None)

    @property
    def reported(self) -> dict[str, float]:
        return {lab: v for _, lab, v in self.sequences}


HUA2025 = Benchmark(
    name="hua2025",
    citation="Hua, Yuan & Fermüller 2025 (arXiv:2507.14500)",
    sequences=(
        ("scene13_dyn_test_00", "13-00", 82.15),
        ("scene13_dyn_test_05", "13-05", 76.24),
        ("scene14_dyn_test_03", "14-03", 79.58),
        ("scene14_dyn_test_04", "14-04", 73.36),
        ("scene14_dyn_test_05", "14-05", 75.67),
    ),
)
BENCHMARKS = {b.name: b for b in (HUA2025,)}

DEFAULTS = {"event_windows_ms": [12.5, 50.0], "gt": "moving", "pred": "fg", "min_gt_events": 1}


def settings(cfg: dict[str, Any], *, gt: str | None = None, pred: str | None = None,
             windows_ms: Iterable[float] | None = None) -> dict[str, Any]:
    """config ``benchmark:`` with CLI overrides (None = keep the config value)."""
    s = {**DEFAULTS, **(cfg.get("benchmark", {}) or {})}
    if gt:
        s["gt"] = gt
    if pred:
        s["pred"] = pred
    if windows_ms:
        s["event_windows_ms"] = list(windows_ms)
    s["event_windows_ms"] = [float(w) for w in (s["event_windows_ms"] if isinstance(
        s["event_windows_ms"], (list, tuple)) else [s["event_windows_ms"]])]
    if s["gt"] not in GT_MODES:
        raise ValueError(f"benchmark.gt must be one of {GT_MODES}, got {s['gt']!r}")
    if s["pred"] not in PRED_SOURCES:
        raise ValueError(f"benchmark.pred must be one of {PRED_SOURCES}, got {s['pred']!r}")
    s["min_gt_events"] = int(s["min_gt_events"])
    return s


def gt_moving_pixels(obj: np.ndarray, moving: Iterable[int], slow: Iterable[int],
                     mode: str) -> np.ndarray:
    """(H, W) bool: pixels of the objects that count as moving under ``mode``."""
    if mode == "tracked":
        return obj > 0
    if mode not in GT_MODES:
        raise ValueError(f"gt mode must be one of {GT_MODES}, got {mode!r}")
    ids = {int(i) for i in moving} | ({int(i) for i in slow} if mode == "moving_slow" else set())
    if not ids:
        return np.zeros(obj.shape, dtype=bool)
    return np.isin(obj, np.array(sorted(ids), dtype=obj.dtype))


def slow_objects(seq_dir: Path, frame_index: int, params: MotionParams,
                 meta: dict[str, Any]) -> set[int]:
    """Objects measured between static_px and move_px. The ambiguous set also holds
    objects with no pose (unmeasurable) -- those are not "slow", so they stay out."""
    sm = sequence_motion(seq_dir, params, meta)
    speeds = sm.speed_px.get(frame_index, {})
    return {int(i) for i in sm.ambiguous.get(frame_index, ()) if int(i) in speeds}


class EventStream:
    """One sequence's events, memory-mapped once."""

    def __init__(self, seq_dir: Path) -> None:
        self.t = np.load(Path(seq_dir) / "dataset_events_t.npy", mmap_mode="r").reshape(-1)
        self.xy = np.load(Path(seq_dir) / "dataset_events_xy.npy", mmap_mode="r")

    def window(self, t0: float, t1: float, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) of the events in [t0, t1] that fall on the sensor."""
        i0 = int(np.searchsorted(self.t, t0, side="left"))
        i1 = int(np.searchsorted(self.t, t1, side="right"))
        xy = np.asarray(self.xy[i0:i1]).astype(np.int64).reshape(-1, 2)
        x, y = xy[:, 0], xy[:, 1]
        ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        return x[ok], y[ok]


def event_iou_counts(pred: np.ndarray, gt: np.ndarray, x: np.ndarray,
                     y: np.ndarray) -> tuple[int, int, int, int]:
    """(|P∩G|, |P∪G|, |G|, |P|) counted over EVENTS; pred / gt are (H, W) bool."""
    p, g = pred[y, x], gt[y, x]
    return int((p & g).sum()), int((p | g).sum()), int(g.sum()), int(p.sum())


@dataclass
class ProtocolScores:
    """Per-frame event IoUs of one event window, grouped by the paper's sequence label."""

    window_ms: float
    frames: dict[str, list[float]] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, label: str, inter: int, union: int, **info: Any) -> None:
        iou = inter / union if union else float("nan")
        self.frames.setdefault(label, []).append(iou)
        self.records.append({"sequence": label, "iou": iou, **info})

    def per_sequence(self) -> dict[str, float]:
        """Mean per-frame IoU of each sequence, in %."""
        return {lab: 100.0 * float(np.nanmean(v)) for lab, v in self.frames.items() if v}

    def mean(self) -> float:
        """Mean over sequences (each sequence weighs the same, like the paper's table)."""
        per = self.per_sequence()
        return float(np.mean(list(per.values()))) if per else float("nan")

    def frame_mean(self) -> float:
        """Mean over all scored frames (long sequences weigh more)."""
        allv = [v for vs in self.frames.values() for v in vs]
        return 100.0 * float(np.nanmean(allv)) if allv else float("nan")


def _upsample(pred: np.ndarray, height: int, width: int) -> np.ndarray:
    if pred.shape == (height, width):
        return pred
    t = torch.from_numpy(pred.astype(np.float32))[None, None]
    return F.interpolate(t, size=(height, width), mode="nearest")[0, 0].numpy() > 0.5


def _spread_cap(items: list[int], cap: int | None) -> list[int]:
    """At most ``cap`` items, evenly spaced (smoke tests keep every part of a sequence)."""
    if not cap or len(items) <= cap:
        return items
    pos = np.linspace(0, len(items) - 1, cap).round().astype(int)
    return [items[i] for i in pos]


@torch.no_grad()
def run_benchmark(
    model: Any,
    dataset: Any,
    bench: Benchmark,
    device: torch.device,
    *,
    windows_ms: Iterable[float],
    gt_mode: str = "moving",
    pred_source: str = "fg",
    grouping: GroupingParams | None = None,
    min_gt_events: int = 1,
    max_per_sequence: int | None = None,
    log_every: int = 100,
) -> tuple[list[ProtocolScores], dict[str, Any]]:
    """Score ``model`` on the benchmark's sequences of ``dataset`` (an EVIMODataset).

    A frame is scored for a window when it has >= ``min_gt_events`` GT-moving events in
    it (the paper's "frames containing moving objects"). Frames without any are found
    from the masks and events first and never reach the model. ``max_per_sequence``
    (smoke tests) keeps that many of the remaining frames, evenly spaced.
    """
    windows = [float(w) for w in windows_ms]
    if pred_source == "objects" and grouping is None:
        raise ValueError("pred='objects' needs grouping parameters")
    params: MotionParams = dataset.motion_params
    model.eval()

    metas: dict[Path, dict] = {}
    masks: dict[Path, Any] = {}
    streams: dict[Path, EventStream] = {}

    def frame_gt(k: int):
        """GT-moving pixels + each window's events of dataset frame k (None = no mover)."""
        seq_dir, fi = dataset.index[k]
        seq_dir = Path(seq_dir)
        if seq_dir not in metas:
            metas[seq_dir] = load_meta(seq_dir)
            masks[seq_dir] = np.load(seq_dir / "dataset_mask.npz")
            streams[seq_dir] = EventStream(seq_dir)
        meta = metas[seq_dir]
        obj = masks[seq_dir][gt_key("mask", fi)] // 1000
        sm = sequence_motion(seq_dir, params, meta)
        gt = gt_moving_pixels(obj, sm.moving.get(fi, ()),
                              slow_objects(seq_dir, fi, params, meta), gt_mode)
        if not gt.any():
            return "no_gt_pixels"
        H, W = obj.shape
        ts = float(meta["frames"][fi]["ts"])
        evs = {w: streams[seq_dir].window(ts - w / 1000.0, ts, H, W) for w in windows}
        if max(int(gt[y, x].sum()) for x, y in evs.values()) < min_gt_events:
            return "no_gt_events"
        return gt, evs, int(fi), ts

    all_frames: dict[str, list[int]] = {}
    for k, (seq_dir, _fi) in enumerate(dataset.index):
        lab = bench.label_of(Path(seq_dir).name)
        if lab is not None:
            all_frames.setdefault(lab, []).append(k)
    stats = {"candidates": sum(map(len, all_frames.values())), "no_gt_pixels": 0,
             "no_gt_events": 0, "run": 0,
             "missing": [lab for lab in bench.reported if lab not in all_frames]}

    scores = [ProtocolScores(w) for w in windows]
    try:
        # 1. frames that contain a moving object (masks + events only -- cheap); the
        #    smoke-test cap is applied to THESE, so a capped run still scores movers
        order: list[tuple[str, int]] = []
        for lab in bench.reported:
            keep = []
            for k in all_frames.get(lab, []):
                r = frame_gt(k)
                if isinstance(r, str):
                    stats[r] += 1
                else:
                    keep.append(k)
            order += [(lab, k) for k in _spread_cap(keep, max_per_sequence)]

        # 2. run the model on them
        for n, (lab, k) in enumerate(order):
            gt, evs, fi, ts = frame_gt(k)
            H, W = gt.shape
            sample = dataset[k]
            surface = sample["surface"].unsqueeze(0).to(device)
            out = model(surface, task="segmentation")
            pred_t = out["seg_logits"].argmax(dim=1)[0] > 0
            if pred_source == "objects":
                if out.get("flow") is None:
                    raise ValueError("pred='objects' needs a head that returns the flow")
                res, _ = model.residual_from_flow(out["flow"], surface)
                ev = score_pixel_mask(sample, surface)[0]
                objects = group_objects(res[0].cpu().numpy(),
                                        (pred_t & ev).cpu().numpy(), grouping)
                pred = objects > 0
            else:
                pred = pred_t.cpu().numpy()
            pred_full = _upsample(pred, H, W)
            stats["run"] += 1

            for sc in scores:
                x, y = evs[sc.window_ms]
                inter, union, g, p = event_iou_counts(pred_full, gt, x, y)
                if g >= min_gt_events:
                    sc.add(lab, inter, union, frame_index=fi, ts=ts,
                           gt_events=g, pred_events=p, events=int(x.size))
            if log_every and (n + 1) % log_every == 0:
                print(f"[benchmark] {n + 1}/{len(order)} frames")
    finally:
        for m in masks.values():
            m.close()
    return scores, stats


def format_report(bench: Benchmark, scores: list[ProtocolScores], stats: dict[str, Any],
                  *, gt_mode: str, pred_source: str) -> str:
    """Our per-sequence numbers next to the paper's table."""
    rep = bench.reported
    lines = [
        f"=== BENCHMARK {bench.name}: {bench.citation} ===",
        "metric : IoU over events at full sensor resolution, only frames with a moving object,",
        "         per-frame IoU averaged per sequence, no ignore band",
        f"setting: gt={gt_mode}  pred={pred_source}  "
        f"(the paper does not state its event window / GT definition)",
        f"frames : {stats['candidates']} in these sequences: {stats['no_gt_pixels']} without "
        f"a moving object, {stats['no_gt_events']} without a GT-moving event, "
        f"{stats['run']} scored",
    ]
    if stats.get("missing"):
        lines.append(f"MISSING: {', '.join(stats['missing'])} -- not in the dataset index")
    for sc in scores:
        per = sc.per_sequence()
        lines.append(f"--- event window {sc.window_ms:g} ms ---")
        lines.append(f"  {'sequence':<10s}{'frames':>8s}{'ours %':>10s}{'paper %':>10s}{'diff':>9s}")
        for lab, paper in rep.items():
            if lab in per:
                lines.append(f"  {lab:<10s}{len(sc.frames[lab]):>8d}{per[lab]:>10.2f}"
                             f"{paper:>10.2f}{per[lab] - paper:>+9.2f}")
            else:
                lines.append(f"  {lab:<10s}{0:>8d}{'n/a':>10s}{paper:>10.2f}")
        if per:
            paper_same = float(np.mean([rep[lab] for lab in per]))
            tag = "" if len(per) == len(rep) else "  (paper over the same sequences)"
            lines.append(f"  {f'mean {len(per)}/{len(rep)}':<18s}{sc.mean():>10.2f}{paper_same:>10.2f}"
                         f"{sc.mean() - paper_same:>+9.2f}{tag}")
            lines.append(f"  {'frame-weighted':<18s}{sc.frame_mean():>10.2f}")
    return "\n".join(lines)


def save_json(path: Path, bench: Benchmark, scores: list[ProtocolScores], stats: dict[str, Any],
              setting: dict[str, Any]) -> None:
    """Per-frame scores + summary, for the paper's tables and later analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "benchmark": bench.name,
        "citation": bench.citation,
        "paper": bench.reported,
        "setting": setting,
        "stats": stats,
        "windows": [{
            "window_ms": sc.window_ms,
            "per_sequence": sc.per_sequence(),
            "mean": sc.mean(),
            "frame_mean": sc.frame_mean(),
            "frames": sc.records,
        } for sc in scores],
    }
    path.write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
