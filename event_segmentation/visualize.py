#!/usr/bin/env python3
"""
Visualize the streaming instance-segmentation output produced by ``segment.py``.

Reads ``labeled_events`` (parquet or csv), ``tracks.json`` and ``colormap.json``
from an output directory and renders:

  out/preview.mp4        FuncAnimation replay of events coloured by instance
                         over time (falls back to preview.gif if no ffmpeg).
  out/trajectories.png   static centroid trajectory per instance.
  out/speed_profiles.png per-instance speed vs time.

Example:
  python visualize.py --out out/
  python visualize.py --out out/ --fps 30 --frame-dt 0.02 --max-time 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_labeled_events(out: Path):
    """Load t,x,y,p,instance_id from parquet (preferred) or csv."""
    pq = out / "labeled_events.parquet"
    csv = out / "labeled_events.csv"
    if pq.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(pq)
            return (df["t"].to_numpy(), df["x"].to_numpy().astype(np.int32),
                    df["y"].to_numpy().astype(np.int32),
                    df["instance_id"].to_numpy().astype(np.int32))
        except Exception as e:
            print(f"  parquet read failed ({e}); trying csv")
    if csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv)
            return (df["t"].to_numpy(), df["x"].to_numpy().astype(np.int32),
                    df["y"].to_numpy().astype(np.int32),
                    df["instance_id"].to_numpy().astype(np.int32))
        except Exception:
            arr = np.genfromtxt(csv, delimiter=",", names=True)
            return (arr["t"], arr["x"].astype(np.int32),
                    arr["y"].astype(np.int32),
                    arr["instance_id"].astype(np.int32))
    raise FileNotFoundError(f"No labeled_events.* found in {out}")


def load_colormap(out: Path) -> dict[int, tuple]:
    cmap_path = out / "colormap.json"
    cmap = {-1: (110, 110, 110)}
    if cmap_path.exists():
        raw = json.loads(cmap_path.read_text())
        for k, v in raw.items():
            cmap[int(k)] = tuple(v)
    return cmap


def load_tracks(out: Path) -> list[dict]:
    p = out / "tracks.json"
    return json.loads(p.read_text()) if p.exists() else []


def color_array(instances: np.ndarray, cmap: dict[int, tuple]) -> np.ndarray:
    """Map an instance-id array to an (N,3) float RGB array in [0,1]."""
    out = np.empty((instances.shape[0], 3), dtype=np.float32)
    default = np.array(cmap.get(-1, (110, 110, 110)), dtype=np.float32)
    uniq = np.unique(instances)
    lut = {i: np.array(cmap.get(int(i), default), dtype=np.float32) for i in uniq}
    for i in uniq:
        out[instances == i] = lut[i] / 255.0
    return out


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def render_animation(out: Path, t, x, y, inst, cmap, fps: int,
                     frame_dt: float, max_time: float) -> str:
    W = int(x.max()) + 1
    H = int(y.max()) + 1
    t0 = float(t[0])
    t1 = float(t[-1])
    if max_time > 0:
        t1 = min(t1, t0 + max_time)
    n_frames = max(1, int(np.ceil((t1 - t0) / frame_dt)))

    colors = color_array(inst, cmap)

    fig, ax = plt.subplots(figsize=(W / 30, H / 30), dpi=110)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # image coords: y down
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("event instance segmentation", color="white", fontsize=9)

    scat = ax.scatter([], [], s=3, marker="s")
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                  ha="left", color="white", fontsize=8, family="monospace")

    # Precompute frame slice boundaries via searchsorted (events are time-sorted).
    edges = t0 + np.arange(n_frames + 1) * frame_dt
    los = np.searchsorted(t, edges[:-1], "left")
    his = np.searchsorted(t, edges[1:], "left")
    # A short trailing window keeps recent events visible for continuity.
    trail = max(frame_dt, 0.02)

    def update(f):
        b = edges[f + 1]
        a = max(t0, b - trail)
        lo = int(np.searchsorted(t, a, "left"))
        hi = int(his[f])
        if hi > lo:
            pts = np.column_stack([x[lo:hi], y[lo:hi]])
            scat.set_offsets(pts)
            scat.set_facecolors(colors[lo:hi])
        else:
            scat.set_offsets(np.empty((0, 2)))
        n_inst = len(np.unique(inst[lo:hi][inst[lo:hi] >= 0])) if hi > lo else 0
        txt.set_text(f"t={b - t0:6.3f}s  frame {f + 1}/{n_frames}\n"
                     f"events {hi - lo:5d}  instances {n_inst}")
        return scat, txt

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False,
                         interval=1000 / max(1, fps))

    mp4 = out / "preview.mp4"
    gif = out / "preview.gif"
    saved = None
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(str(mp4), writer=writer)
        saved = str(mp4)
    except Exception as e:
        print(f"  ffmpeg unavailable ({type(e).__name__}); writing GIF instead")
        try:
            anim.save(str(gif), writer=PillowWriter(fps=fps))
            saved = str(gif)
        except Exception as e2:
            print(f"  GIF write failed: {e2}")
    plt.close(fig)
    return saved or "(animation failed)"


# ---------------------------------------------------------------------------
# Static plots
# ---------------------------------------------------------------------------

def _top_tracks(tracks: list[dict], k: int) -> list[dict]:
    """The K largest instances by event count (keeps busy plots readable)."""
    return sorted(tracks, key=lambda t: t.get("n_events", 0), reverse=True)[:k]


def render_trajectories(out: Path, tracks: list[dict], cmap: dict,
                        top_k: int = 15) -> str:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    any_pts = False
    shown = _top_tracks(tracks, top_k)
    for tr in shown:
        traj = tr.get("trajectory", [])
        if not traj:
            continue
        cx = [s["cx"] for s in traj]
        cy = [s["cy"] for s in traj]
        col = np.array(tr["color_rgb"]) / 255.0
        ax.plot(cx, cy, "-", color=col, lw=2, alpha=0.9,
                label=f"#{tr['id']} ({tr['n_events']} ev)")
        ax.plot(cx[0], cy[0], "o", color=col, ms=6)
        ax.plot(cx[-1], cy[-1], "s", color=col, ms=6)
        any_pts = True
    extra = max(0, len(tracks) - len(shown))
    title = "instance centroid trajectories"
    if extra:
        title += f"  (top {len(shown)} of {len(tracks)} by event count)"
    ax.set_title(title)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.invert_yaxis()  # image coordinates
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    if any_pts:
        ax.legend(fontsize=8, loc="best", ncol=2)
    else:
        ax.text(0.5, 0.5, "no trajectories", ha="center", va="center",
                transform=ax.transAxes)
    path = out / "trajectories.png"
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return str(path)


def render_speed_profiles(out: Path, tracks: list[dict], top_k: int = 15) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    any_pts = False
    t_ref = None
    shown = _top_tracks(tracks, top_k)
    for tr in shown:
        traj = tr.get("trajectory", [])
        if not traj:
            continue
        ts = np.array([s["t"] for s in traj])
        sp = np.array([np.hypot(s["vx"], s["vy"]) for s in traj])
        if t_ref is None:
            t_ref = ts.min()
        col = np.array(tr["color_rgb"]) / 255.0
        ax.plot(ts - t_ref, sp, "-", color=col, lw=1.8,
                label=f"#{tr['id']}")
        any_pts = True
    extra = max(0, len(tracks) - len(shown))
    title = "per-instance speed vs time"
    if extra:
        title += f"  (top {len(shown)} of {len(tracks)})"
    ax.set_title(title)
    ax.set_xlabel("t (s, relative)"); ax.set_ylabel("speed (px/s)")
    ax.grid(True, alpha=0.2)
    if any_pts:
        ax.legend(fontsize=8, loc="best", ncol=2)
    else:
        ax.text(0.5, 0.5, "no trajectories", ha="center", va="center",
                transform=ax.transAxes)
    path = out / "speed_profiles.png"
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Visualize event-segmentation output.")
    ap.add_argument("--out", default="out", help="output dir written by segment.py")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--frame-dt", type=float, default=0.02,
                    help="seconds of events per animation frame")
    ap.add_argument("--max-time", type=float, default=0.0,
                    help="cap animation to the first N seconds (0 = all)")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if not out.exists():
        print(f"ERROR: output dir not found: {out}")
        return 2

    print(f"Loading segmentation output from {out} …")
    t, x, y, inst = load_labeled_events(out)
    cmap = load_colormap(out)
    tracks = load_tracks(out)
    print(f"  {len(t):,} events, {len(tracks)} instances")

    print("Rendering animation …")
    print(f"  {render_animation(out, t, x, y, inst, cmap, a.fps, a.frame_dt, a.max_time)}")
    print("Rendering trajectories …")
    print(f"  {render_trajectories(out, tracks, cmap)}")
    print("Rendering speed profiles …")
    print(f"  {render_speed_profiles(out, tracks)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
