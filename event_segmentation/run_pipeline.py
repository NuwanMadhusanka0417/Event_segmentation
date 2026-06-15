#!/usr/bin/env python3
"""Programmatic wrapper for segment.py + visualize.py (used by the web UI)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

SEG_DIR = Path(__file__).resolve().parent
if str(SEG_DIR) not in sys.path:
    sys.path.insert(0, str(SEG_DIR))

from segment import (  # noqa: E402
    Config,
    Segmenter,
    iter_events,
    write_colormap,
    write_frames,
    write_labeled_events,
    write_summary,
    write_tracks_json,
)
from visualize import (  # noqa: E402
    load_colormap,
    load_labeled_events,
    load_tracks,
    render_animation,
    render_speed_profiles,
    render_trajectories,
)

ProgressFn = Callable[[str, Optional[float]], None]


def run_segmentation(cfg: Config, progress: ProgressFn | None = None) -> dict:
    """Run the segmentation core loop and write labeled outputs."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    in_path = Path(cfg.input)
    if not in_path.exists():
        raise FileNotFoundError(f"input not found: {in_path}")

    def report(msg: str, pct: float | None = None) -> None:
        if progress:
            progress(msg, pct)

    report("Starting segmentation…", 0.0)
    seg = Segmenter(cfg)
    t_start = time.perf_counter()
    n = 0
    limit = cfg.limit or None
    for t, x, y, p in iter_events(str(in_path), limit=limit or 0):
        seg.process_event(t, x, y, p)
        n += 1
        if cfg.progress_every and n % cfg.progress_every == 0:
            el = time.perf_counter() - t_start
            report(f"Processed {n:,} events ({n / el:,.0f} ev/s)", None)
    seg.finalize()
    runtime = time.perf_counter() - t_start

    if n == 0:
        raise ValueError("no events parsed from input")

    report("Resolving instance labels…", 0.85)
    t0 = seg.ev_t[0]
    t1 = seg.ev_t[-1]
    instance, info = seg.resolve_labels()
    n_inst = info["n_confirmed_instances"]

    report(f"Writing outputs ({n_inst} instances)…", 0.88)
    labeled_path = write_labeled_events(out, seg, instance, cfg.csv)
    write_tracks_json(out, seg, instance, info, t0, t1)
    write_colormap(out, n_inst)
    n_frames = 0
    if cfg.render_frames:
        n_frames = write_frames(out, seg, instance, t0, t1)

    n_labeled = int((instance >= 0).sum())
    summary_extra = {
        "duration_s": round(float(t1 - t0), 6),
        "sensor_width": cfg.width or seg.width,
        "sensor_height": cfg.height or seg.height,
        "events_assigned_to_instances": n_labeled,
        "events_background": int(n - n_labeled),
        "frames_written": n_frames,
    }
    write_summary(out, cfg, n, n_inst, runtime, extra=summary_extra)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    report("Segmentation complete.", 0.92)
    return {
        "summary": summary,
        "labeled_path": labeled_path,
        "n_events": n,
        "n_instances": n_inst,
        "runtime_s": runtime,
    }


def run_visualization(
    out: Path,
    fps: int = 25,
    frame_dt: float = 0.02,
    max_time: float = 0.0,
    progress: ProgressFn | None = None,
    skip_animation: bool = False,
) -> dict:
    """Render preview animation and static plots from segmentation output."""
    out = Path(out)
    if not out.exists():
        raise FileNotFoundError(f"output dir not found: {out}")

    def report(msg: str, pct: float | None = None) -> None:
        if progress:
            progress(msg, pct)

    report("Loading labeled events…", 0.93)
    cmap = load_colormap(out)
    tracks = load_tracks(out)

    preview_name = None
    if skip_animation:
        report("Skipping server preview (browser playback)…", 0.94)
    else:
        t, x, y, inst = load_labeled_events(out)
        report("Rendering preview animation…", 0.94)
        preview = render_animation(out, t, x, y, inst, cmap, fps, frame_dt, max_time)
        if preview and preview != "(animation failed)":
            preview_name = Path(preview).name

    report("Rendering trajectories…", 0.97)
    traj_path = render_trajectories(out, tracks, cmap)
    report("Rendering speed profiles…", 0.99)
    speed_path = render_speed_profiles(out, tracks)

    assets = {
        "preview": preview_name,
        "trajectories": traj_path,
        "speed_profiles": speed_path,
    }
    report("Visualization complete.", 1.0)
    return {"assets": assets, "preview_name": preview_name, "n_tracks": len(tracks)}


def run_full_pipeline(
    cfg: Config,
    viz_fps: int = 25,
    viz_frame_dt: float = 0.02,
    viz_max_time: float = 0.0,
    progress: ProgressFn | None = None,
    skip_animation: bool = False,
) -> dict:
    """Segment then visualize; returns combined result dict."""
    seg_result = run_segmentation(cfg, progress=progress)
    viz_result = run_visualization(
        Path(cfg.out),
        fps=viz_fps,
        frame_dt=viz_frame_dt or cfg.frame_dt,
        max_time=viz_max_time,
        progress=progress,
        skip_animation=skip_animation,
    )
    return {**seg_result, **viz_result}
