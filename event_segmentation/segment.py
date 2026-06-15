#!/usr/bin/env python3
"""
Streaming (online) instance segmentation of an event-camera stream by motion
coherence.

The scene content is unknown: events that belong to the same independently
moving object are grouped purely from their space-time motion, not from any
hard-coded object model. The pipeline streams events in time order with
bounded working state (a time-window ring buffer + a spatial hash grid + a
small set of active tracks) and writes visualization-ready outputs.

Core idea (robust to spatially-extended objects):
  * A neighbour graph is built causally over recent events (past -> e) using a
    space-time metric d = sqrt(dx^2 + dy^2 + (beta*dt)^2).
  * A local velocity is fit per event from those neighbours (robust LS).
  * Each event inherits the track label that dominates its neighbours, gated by
    velocity coherence (|v_e - v_track| < vel_tol). This is label propagation
    along the graph, so a single moving blob grows into one instance while two
    objects that merely touch in space are kept apart by their differing motion.
  * Tracks carry an EMA velocity + centroid; idle tracks die, coherent ones
    merge, and a periodic spatial split breaks a label that covers two
    spatially-disconnected blobs.

See README.md for the full algorithm walk-through and parameter tuning guide.

Outputs (written to --out):
  labeled_events.parquet   t,x,y,p,instance_id  (original order; --csv fallback)
  tracks.json              per-instance motion tracks (resampled trajectory)
  colormap.json            instance_id -> [r,g,b]  (-1 -> gray)
  frames/                  one PNG per frame_dt slice (only with --render-frames)
  summary.json             resolved params, totals, runtime, throughput
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# Optional dependencies — degrade gracefully if missing.
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class Config:
    """All tunables. Defaults are tuned to the EED ~847k-event sample."""
    # I/O
    input: str = "events_filtered.txt"
    out: str = "out"
    csv: bool = False
    render_frames: bool = False
    limit: int = 0                     # 0 = no cap; else stop after N events
    seed: int = 0
    progress_every: int = 100_000

    # Neighbour graph / buffer
    dt_window: float = 0.010           # s  neighbour time window / buffer horizon
    k: int = 12                        #    neighbours per event
    beta: float = 2000.0               # px/s  time->space scaling (1 ms ~= 2 px)
    grid_cell: int = 6                 # px  spatial hash cell size
    min_nbrs: int = 5                  #    min neighbours before labeling
    lag: float = 0.002                 # s  look-back lag buffer

    # Track assignment / update
    gate_px: float = 8.0               # px  spatial gate radius
    vel_tol: float = 1500.0            # px/s velocity match tolerance
    lam: float = 0.15                  #    EMA weight for a new event (lambda)
    min_track_events: int = 40         #    events to confirm a candidate track
    track_timeout: float = 0.05        # s  kill an idle track

    # Robust-velocity guards
    v_max: float = 8000.0              # px/s clamp on the local velocity estimate

    # Spatial split / cleanup
    split_px: float = 6.0              # px  spatial-split connectivity radius
    split_every: int = 5000            #    events between spatial splits
    min_track_lifetime_events: int = 200  # drop tiny noise tracks at the end

    # Merge tolerances (housekeeping step 6)
    merge_vel_tol: float = 800.0       # px/s velocity agreement to merge tracks
    merge_px: float = 6.0              # px  centroid agreement to merge tracks

    # Visualization-oriented timing
    traj_dt: float = 0.005             # s  trajectory resample step in tracks.json
    frame_dt: float = 0.02             # s  PNG slice duration for frames/

    # Sensor geometry (0 = auto-detect from the data)
    width: int = 0
    height: int = 0

    def resolved(self) -> dict:
        return asdict(self)


# ===========================================================================
# Colour palette  (Okabe-Ito + Tol bright — colourblind friendly, stable)
# ===========================================================================

_PALETTE = [
    (230, 159, 0),    # orange
    (86, 180, 233),   # sky blue
    (0, 158, 115),    # bluish green
    (240, 228, 66),   # yellow
    (0, 114, 178),    # blue
    (213, 94, 0),     # vermillion
    (204, 121, 167),  # reddish purple
    (170, 68, 153),   # purple
    (51, 187, 238),   # cyan
    (238, 119, 51),   # orange-red
    (102, 204, 153),  # mint
    (187, 187, 51),   # olive
]
BACKGROUND_RGB = (110, 110, 110)  # instance_id == -1


def instance_color(instance_id: int) -> list[int]:
    """Stable, colourblind-friendly colour for an instance id (-1 -> gray)."""
    if instance_id < 0:
        return list(BACKGROUND_RGB)
    return list(_PALETTE[instance_id % len(_PALETTE)])


# ===========================================================================
# Streaming event reader
# ===========================================================================

def iter_events(path: str, look_ahead: int = 512,
                limit: int = 0) -> Iterator[tuple[float, int, int, int]]:
    """
    Stream events from a whitespace-separated ``t x y p`` text file.

    Memory-bounded: reads line-by-line, never materialising the whole stream.
    Malformed lines are skipped. Timestamps are expected to be non-decreasing;
    a small ``look_ahead`` min-heap re-sorts minor local disorder so the core
    loop always sees non-decreasing time.

    Yields ``(t: float, x: int, y: int, p: int)``.
    """
    heap: list[tuple[float, int, int, int, int]] = []
    seq = 0
    emitted = 0
    with open(path, "r", buffering=1 << 20) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                t = float(parts[0])
                x = int(float(parts[1]))
                y = int(float(parts[2]))
                p = int(float(parts[3]))
            except ValueError:
                continue  # defensive: skip malformed line
            heapq.heappush(heap, (t, seq, x, y, p))
            seq += 1
            if len(heap) > look_ahead:
                tt, _, xx, yy, pp = heapq.heappop(heap)
                yield (tt, xx, yy, pp)
                emitted += 1
                if limit and emitted >= limit:
                    return
    while heap:  # drain the look-ahead heap
        tt, _, xx, yy, pp = heapq.heappop(heap)
        yield (tt, xx, yy, pp)
        emitted += 1
        if limit and emitted >= limit:
            return


# ===========================================================================
# Pluggable affinity descriptor
# ===========================================================================
#
# The affinity model sits behind this tiny interface so a learnable HD / VSA
# descriptor can be dropped in later without touching the pipeline. v1 default
# = motion coherence (velocity agreement).

class MotionDescriptor:
    """Default descriptor: a local motion estimate (vx, vy)."""

    def describe(self, vx: float, vy: float) -> tuple:
        return (vx, vy)

    def affinity(self, desc: tuple, track: "Track", cfg: Config) -> float:
        """Velocity-coherence score in [0,1], or -1.0 if it fails the gate."""
        vx, vy = desc
        vel_err = math.hypot(vx - track.vx, vy - track.vy)
        if vel_err > cfg.vel_tol:
            return -1.0
        return 1.0 - vel_err / cfg.vel_tol


# ===========================================================================
# Track
# ===========================================================================

@dataclass
class Track:
    id: int
    vx: float
    vy: float
    cx: float
    cy: float
    t_start: float
    t_last: float
    n_events: int = 0
    confirmed: bool = False
    # Trajectory resampled at cfg.traj_dt for visualization.
    traj_t: list = field(default_factory=list)
    traj_cx: list = field(default_factory=list)
    traj_cy: list = field(default_factory=list)
    traj_vx: list = field(default_factory=list)
    traj_vy: list = field(default_factory=list)
    _last_traj_t: float = -1e18

    def update(self, t: float, x: float, y: float, vx: float, vy: float,
               lam: float, traj_dt: float,
               wmax: float = 1e9, hmax: float = 1e9) -> None:
        """Exponential-moving-average fold of a new event into the track."""
        # Predict centroid forward (dt capped so idle gaps can't fling it away),
        # then EMA toward the observed point and clamp to the sensor.
        dt = t - self.t_last
        dt_pred = dt if dt < 0.005 else 0.005
        self.cx += self.vx * dt_pred
        self.cy += self.vy * dt_pred
        self.cx = (1.0 - lam) * self.cx + lam * x
        self.cy = (1.0 - lam) * self.cy + lam * y
        self.cx = min(max(self.cx, 0.0), wmax)
        self.cy = min(max(self.cy, 0.0), hmax)
        self.vx = (1.0 - lam) * self.vx + lam * vx
        self.vy = (1.0 - lam) * self.vy + lam * vy
        self.t_last = t
        self.n_events += 1
        if t - self._last_traj_t >= traj_dt:
            self.traj_t.append(t)
            self.traj_cx.append(self.cx)
            self.traj_cy.append(self.cy)
            self.traj_vx.append(self.vx)
            self.traj_vy.append(self.vy)
            self._last_traj_t = t

    def seed_traj(self, t: float) -> None:
        self.traj_t.append(t)
        self.traj_cx.append(self.cx)
        self.traj_cy.append(self.cy)
        self.traj_vx.append(self.vx)
        self.traj_vy.append(self.vy)
        self._last_traj_t = t


# ===========================================================================
# Union-Find (track merges + final label resolution)
# ===========================================================================

class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def add(self, a: int) -> None:
        self.parent.setdefault(a, a)

    def find(self, a: int) -> int:
        self.parent.setdefault(a, a)
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:  # path compression
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        lo, hi = (ra, rb) if ra < rb else (rb, ra)  # keep smaller id (stable)
        self.parent[hi] = lo
        return lo


# Growable int64 store with O(1) random write (for lag-buffer relabeling).
class _GrowableInt64:
    def __init__(self) -> None:
        self._buf = np.full(1 << 16, -1, dtype=np.int64)
        self._n = 0

    def append(self, v: int) -> None:
        if self._n >= self._buf.size:
            self._buf = np.concatenate(
                [self._buf, np.full(self._buf.size, -1, dtype=np.int64)])
        self._buf[self._n] = v
        self._n += 1

    def set(self, i: int, v: int) -> None:
        if 0 <= i < self._n:
            self._buf[i] = v

    def to_numpy(self) -> np.ndarray:
        return self._buf[:self._n]


# ===========================================================================
# Segmenter
# ===========================================================================

class Segmenter:
    """Online motion-coherence instance segmenter with bounded working state."""

    def __init__(self, cfg: Config, descriptor: Optional[MotionDescriptor] = None):
        self.cfg = cfg
        self.desc = descriptor or MotionDescriptor()
        np.random.seed(cfg.seed)

        # Ring buffer of recent events (fixed capacity -> bounded memory).
        # Capacity >> events that fit in dt_window (rate * dt_window).
        self.cap = 1 << 17  # 131072 slots
        self.r_t = np.zeros(self.cap, dtype=np.float64)
        self.r_x = np.zeros(self.cap, dtype=np.float32)
        self.r_y = np.zeros(self.cap, dtype=np.float32)
        self.r_cx = np.full(self.cap, -1 << 20, dtype=np.int32)  # owning cell
        self.r_cy = np.full(self.cap, -1 << 20, dtype=np.int32)
        self.r_track = np.full(self.cap, -1, dtype=np.int64)     # provisional id
        self.r_eid = np.full(self.cap, -1, dtype=np.int64)

        # Spatial hash grid: cell -> python list of ring slots.
        self.grid: dict[tuple[int, int], list[int]] = {}

        # Active tracks.
        self.tracks: dict[int, Track] = {}
        self.archive: list[Track] = []  # retired tracks (kept for trajectories)
        self.next_track_id = 0
        self.uf = UnionFind()

        # Per-event output bookkeeping (output scale; only small scalars).
        self.ev_t: list[float] = []
        self.ev_x: list[int] = []
        self.ev_y: list[int] = []
        self.ev_p: list[int] = []
        self.ev_track = _GrowableInt64()

        # Lag buffer: events awaiting more context before labeling.
        self.lag_buf: list[tuple] = []

        self.n_seen = 0
        self.width = 0
        self.height = 0
        self._next_split_at = cfg.split_every

    # ------------------------------------------------------------------
    # Grid + ring buffer
    # ------------------------------------------------------------------
    def _cell(self, x: float, y: float) -> tuple[int, int]:
        c = self.cfg.grid_cell
        return (int(x) // c, int(y) // c)

    def _insert(self, eid: int, t: float, x: int, y: int) -> int:
        slot = eid % self.cap
        cx, cy = self._cell(x, y)
        self.r_t[slot] = t
        self.r_x[slot] = x
        self.r_y[slot] = y
        self.r_cx[slot] = cx
        self.r_cy[slot] = cy
        self.r_track[slot] = -1
        self.r_eid[slot] = eid
        self.grid.setdefault((cx, cy), []).append(slot)
        return slot

    def _compact_grid(self, t_now: float) -> None:
        """Drop stale slots (evicted by time or overwritten by ring wrap)."""
        t_low = t_now - self.cfg.dt_window
        rt, rcx, rcy = self.r_t, self.r_cx, self.r_cy
        for key in list(self.grid.keys()):
            cx, cy = key
            lst = self.grid[key]
            keep = [s for s in lst
                    if rt[s] >= t_low and rcx[s] == cx and rcy[s] == cy]
            if keep:
                self.grid[key] = keep
            else:
                del self.grid[key]

    def _candidate_slots(self, x: int, y: int) -> list[int]:
        """Pure-python gather of ring slots from the 3x3 cells around (x,y)."""
        cx, cy = self._cell(x, y)
        g = self.grid
        cand: list[int] = []
        ext = cand.extend
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                lst = g.get((gx, gy))
                if lst:
                    ext(lst)
        return cand

    # ------------------------------------------------------------------
    # kNN + robust velocity (vectorised over candidates)
    # ------------------------------------------------------------------
    def _neighbours(self, x: int, y: int, t_ref: float, self_slot: int):
        """
        Return (k_slots, vx, vy, n_nbrs). k_slots is a numpy array of the k
        nearest causal neighbour slots under the space-time metric; (vx,vy) is
        a robust local velocity fit; n_nbrs is the neighbour count.
        """
        cfg = self.cfg
        cand = self._candidate_slots(x, y)
        if not cand:
            return None, 0.0, 0.0, 0
        slots = np.fromiter(cand, dtype=np.int64, count=len(cand))

        ts = self.r_t[slots]
        cxs = self.r_cx[slots]
        cys = self.r_cy[slots]
        cellx, celly = self._cell(x, y)
        # Causal + in-window + genuinely within the 3x3 neighbourhood + not self.
        mask = ((ts <= t_ref) & (ts >= t_ref - cfg.dt_window) &
                (slots != self_slot) &
                (np.abs(cxs - cellx) <= 1) & (np.abs(cys - celly) <= 1))
        if not mask.any():
            return None, 0.0, 0.0, 0
        slots = slots[mask]
        xs = self.r_x[slots]
        ys = self.r_y[slots]
        ts = ts[mask]

        dx = x - xs
        dy = y - ys
        dt = t_ref - ts
        bd = cfg.beta * dt
        d2 = dx * dx + dy * dy + bd * bd

        if d2.size > cfg.k:
            idx = np.argpartition(d2, cfg.k)[:cfg.k]
            ksel = slots[idx]
            kdx = dx[idx].astype(np.float64)
            kdy = dy[idx].astype(np.float64)
            kdt = dt[idx].astype(np.float64)
        else:
            ksel = slots
            kdx = dx.astype(np.float64)
            kdy = dy.astype(np.float64)
            kdt = dt.astype(np.float64)
        n_nbrs = int(ksel.size)

        # Robust velocity: trimmed least-squares through the origin
        # minimise sum (dx - vx*dt)^2 over neighbours with dt > eps.
        m = kdt > 1e-6
        if int(m.sum()) < 2:
            return ksel, 0.0, 0.0, n_nbrs
        a = kdt[m]; bx = kdx[m]; by = kdy[m]
        # Need real temporal spread, else dx/dt blows up on a simultaneous burst.
        if float(a.max()) < 0.1 * cfg.dt_window:
            return ksel, 0.0, 0.0, n_nbrs
        denom = float(a @ a)
        if denom <= 0:
            return ksel, 0.0, 0.0, n_nbrs
        vx = float(a @ bx / denom)
        vy = float(a @ by / denom)
        if a.size >= 4:  # one trimmed refit (drop worst ~30% residuals)
            res = (bx - vx * a) ** 2 + (by - vy * a) ** 2
            thr = np.sort(res)[int(0.7 * res.size)]
            keep = res <= thr
            if int(keep.sum()) >= 2:
                a2 = a[keep]; den2 = float(a2 @ a2)
                if den2 > 0:
                    vx = float(a2 @ bx[keep] / den2)
                    vy = float(a2 @ by[keep] / den2)
        # Clamp implausibly large estimates (noise / near-zero dt).
        sp = math.hypot(vx, vy)
        if sp > cfg.v_max:
            s = cfg.v_max / sp
            vx *= s; vy *= s
        return ksel, vx, vy, n_nbrs

    # ------------------------------------------------------------------
    # Assignment: motion-coherent connectivity (single-linkage)
    # ------------------------------------------------------------------
    def _spawn(self, t: float, x: int, y: int, vx: float, vy: float) -> int:
        tid = self.next_track_id
        self.next_track_id += 1
        self.uf.add(tid)
        tr = Track(id=tid, vx=vx, vy=vy, cx=float(x), cy=float(y),
                   t_start=t, t_last=t, n_events=1)
        tr.seed_traj(t)
        self.tracks[tid] = tr
        return tid

    def _merge_track_state(self, keep_id: int, drop_id: int) -> None:
        if keep_id == drop_id:
            return
        keep = self.tracks.get(keep_id)
        drop = self.tracks.get(drop_id)
        if keep is None or drop is None:
            return
        keep.n_events += drop.n_events
        keep.confirmed = keep.confirmed or drop.confirmed
        keep.t_start = min(keep.t_start, drop.t_start)
        keep.t_last = max(keep.t_last, drop.t_last)
        keep.traj_t += drop.traj_t
        keep.traj_cx += drop.traj_cx
        keep.traj_cy += drop.traj_cy
        keep.traj_vx += drop.traj_vx
        keep.traj_vy += drop.traj_vy
        self.tracks.pop(drop_id, None)

    def _assign(self, t: float, x: int, y: int, vx: float, vy: float,
                ksel: np.ndarray) -> int:
        """
        Join the dominant track among the neighbours (spatial-temporal
        connectivity); union into it any other neighbour track that moves
        coherently (|v - v| < vel_tol), so a single blob becomes one instance
        while differently-moving objects stay separate. Spawn if no neighbour
        is labelled yet.
        """
        cfg = self.cfg
        if ksel is None or ksel.size == 0:
            return self._spawn(t, x, y, vx, vy)
        labels = self.r_track[ksel]
        labels = labels[labels >= 0]
        if labels.size == 0:
            return self._spawn(t, x, y, vx, vy)

        # Tally votes per resolved track root, weighted toward the velocity-
        # coherent tracks so a differently-moving object that merely touches
        # this event does not capture it (motion-coherent connectivity).
        votes: dict[int, float] = {}
        desc = self.desc.describe(vx, vy)
        for lab in labels.tolist():
            r = self.uf.find(lab)
            tr = self.tracks.get(r)
            w = 1.0
            if tr is not None:
                aff = self.desc.affinity(desc, tr, cfg)
                w = 1.0 + 2.0 * aff if aff >= 0.0 else 0.25
            votes[r] = votes.get(r, 0.0) + w
        primary = max(votes, key=votes.get)
        ptr = self.tracks.get(primary)
        if ptr is None:
            return self._spawn(t, x, y, vx, vy)

        ptr.update(t, x, y, vx, vy, cfg.lam, cfg.traj_dt,
                   float(self.width), float(self.height))
        if not ptr.confirmed and ptr.n_events >= cfg.min_track_events:
            ptr.confirmed = True
        return primary

    # ------------------------------------------------------------------
    # Housekeeping: timeout + merge
    # ------------------------------------------------------------------
    def _housekeep(self, t_now: float) -> None:
        cfg = self.cfg
        dead = [tid for tid, tr in self.tracks.items()
                if t_now - tr.t_last > cfg.track_timeout]
        for tid in dead:
            # Retire (don't discard): keep the trajectory for the final output.
            self.archive.append(self.tracks.pop(tid))

        # Merge active tracks whose predicted centroid + velocity agree.
        items = list(self.tracks.items())
        for i in range(len(items)):
            ti, tri = items[i]
            if ti not in self.tracks:
                continue
            for j in range(i + 1, len(items)):
                tj, trj = items[j]
                if tj not in self.tracks or tj == ti:
                    continue
                pix = tri.cx + tri.vx * (t_now - tri.t_last)
                piy = tri.cy + tri.vy * (t_now - tri.t_last)
                pjx = trj.cx + trj.vx * (t_now - trj.t_last)
                pjy = trj.cy + trj.vy * (t_now - trj.t_last)
                if (math.hypot(pix - pjx, piy - pjy) <= cfg.merge_px and
                        math.hypot(tri.vx - trj.vx, tri.vy - trj.vy)
                        <= cfg.merge_vel_tol):
                    root = self.uf.union(ti, tj)
                    keep, drop = (tri, trj) if root == ti else (trj, tri)
                    keep.n_events += drop.n_events
                    keep.confirmed = keep.confirmed or drop.confirmed
                    keep.t_start = min(keep.t_start, drop.t_start)
                    keep.traj_t += drop.traj_t
                    keep.traj_cx += drop.traj_cx
                    keep.traj_cy += drop.traj_cy
                    keep.traj_vx += drop.traj_vx
                    keep.traj_vy += drop.traj_vy
                    self.tracks.pop(drop.id, None)

    # ------------------------------------------------------------------
    # Spatial split (connected components within split_px)
    # ------------------------------------------------------------------
    def _spatial_split(self, t_now: float) -> None:
        cfg = self.cfg
        self._compact_grid(t_now)
        # Group live slots by resolved track root.
        groups: dict[int, list[int]] = {}
        for lst in self.grid.values():
            for s in lst:
                tk = int(self.r_track[s])
                if tk < 0:
                    continue
                groups.setdefault(self.uf.find(tk), []).append(s)

        sp2 = cfg.split_px * cfg.split_px
        cell = max(1, int(cfg.split_px))
        for root, slots in groups.items():
            if len(slots) < 2 * cfg.min_nbrs:
                continue
            local = UnionFind()
            buckets: dict[tuple[int, int], list[int]] = {}
            for s in slots:
                local.add(s)
                bx = int(self.r_x[s]) // cell
                by = int(self.r_y[s]) // cell
                buckets.setdefault((bx, by), []).append(s)
            for (bx, by), members in buckets.items():
                neigh: list[int] = []
                for gx in (bx - 1, bx, bx + 1):
                    for gy in (by - 1, by, by + 1):
                        n = buckets.get((gx, gy))
                        if n:
                            neigh.extend(n)
                for a in members:
                    ax, ay = self.r_x[a], self.r_y[a]
                    for b in neigh:
                        if b <= a:
                            continue
                        ddx = ax - self.r_x[b]
                        ddy = ay - self.r_y[b]
                        if ddx * ddx + ddy * ddy <= sp2:
                            local.union(a, b)
            comps: dict[int, list[int]] = {}
            for s in slots:
                comps.setdefault(local.find(s), []).append(s)
            if len(comps) < 2:
                continue
            base = self.tracks.get(root)
            ordered = sorted(comps.values(), key=len, reverse=True)
            for comp in ordered[1:]:  # keep the largest blob on the original id
                if len(comp) < cfg.min_nbrs:
                    continue
                new_id = self.next_track_id
                self.next_track_id += 1
                self.uf.add(new_id)
                mx = float(np.mean([self.r_x[s] for s in comp]))
                my = float(np.mean([self.r_y[s] for s in comp]))
                vx = base.vx if base else 0.0
                vy = base.vy if base else 0.0
                conf = base.confirmed if base else False
                tr = Track(id=new_id, vx=vx, vy=vy, cx=mx, cy=my,
                           t_start=t_now, t_last=t_now, n_events=len(comp),
                           confirmed=conf)
                tr.seed_traj(t_now)
                self.tracks[new_id] = tr
                for s in comp:
                    self.r_track[s] = new_id
                    self.ev_track.set(int(self.r_eid[s]), new_id)

    # ------------------------------------------------------------------
    # Per-event core
    # ------------------------------------------------------------------
    def _label_now(self, eid: int, t: float, x: int, y: int, slot: int) -> None:
        ksel, vx, vy, n_nbrs = self._neighbours(x, y, t, slot)
        if n_nbrs < self.cfg.min_nbrs:
            self.ev_track.set(eid, -1)
            return
        tid = self._assign(t, x, y, vx, vy, ksel)
        self.r_track[slot] = tid
        self.ev_track.set(eid, tid)

    def process_event(self, t: float, x: int, y: int, p: int) -> None:
        cfg = self.cfg
        eid = self.n_seen
        self.n_seen += 1
        if x + 1 > self.width:
            self.width = x + 1
        if y + 1 > self.height:
            self.height = y + 1

        self.ev_t.append(t)
        self.ev_x.append(x)
        self.ev_y.append(y)
        self.ev_p.append(p)
        self.ev_track.append(-1)

        slot = self._insert(eid, t, x, y)

        if self.lag_buf:
            self._drain_lag(t)

        # Decide: label now, or defer for more causal context?
        ksel, vx, vy, n_nbrs = self._neighbours(x, y, t, slot)
        if n_nbrs < cfg.min_nbrs and cfg.lag > 0.0:
            self.lag_buf.append((eid, t, x, y, slot))
        elif n_nbrs < cfg.min_nbrs:
            self.ev_track.set(eid, -1)
        else:
            tid = self._assign(t, x, y, vx, vy, ksel)
            self.r_track[slot] = tid
            self.ev_track.set(eid, tid)

        if (eid & 2047) == 0:
            self._compact_grid(t)  # keep candidate lists tight (and bounded)
            self._housekeep(t)
        if eid >= self._next_split_at:
            self._spatial_split(t)
            self._next_split_at += cfg.split_every

    def _drain_lag(self, t_now: float) -> None:
        cfg = self.cfg
        keep: list[tuple] = []
        for item in self.lag_buf:
            eid, te, xe, ye, slot = item
            if t_now - te >= cfg.lag:
                self._label_now(eid, te, xe, ye, slot)
            else:
                keep.append(item)
        self.lag_buf = keep

    def finalize(self) -> None:
        for (eid, te, xe, ye, slot) in self.lag_buf:
            self._label_now(eid, te, xe, ye, slot)
        self.lag_buf = []

    # ------------------------------------------------------------------
    # Final label resolution
    # ------------------------------------------------------------------
    def resolve_labels(self) -> tuple[np.ndarray, dict]:
        """
        Resolve provisional per-event track ids through the merge union-find,
        drop tiny/unconfirmed tracks, and remap survivors to compact instance
        ids (0..M-1) ordered by start time.
        """
        cfg = self.cfg
        prov = self.ev_track.to_numpy()

        # Resolve provisional ids -> merge roots (cached).
        roots = np.full_like(prov, -1)
        cache: dict[int, int] = {}
        pr = prov.tolist()
        rt_out = roots
        for i, v in enumerate(pr):
            if v < 0:
                continue
            r = cache.get(v)
            if r is None:
                r = self.uf.find(int(v))
                cache[v] = r
            rt_out[i] = r

        valid = roots[roots >= 0]
        if valid.size:
            unique, counts = np.unique(valid, return_counts=True)
            n_by_root = dict(zip(unique.tolist(), counts.tolist()))
        else:
            n_by_root = {}

        # Group every track ever created (alive + retired) by its merge root, so
        # trajectories survive timeouts and merges in the final output.
        root_tracks: dict[int, list[Track]] = {}
        for tr in list(self.tracks.values()) + self.archive:
            r = self.uf.find(tr.id)
            root_tracks.setdefault(r, []).append(tr)

        survivors = []
        for r, n in n_by_root.items():
            if n < cfg.min_track_lifetime_events:
                continue
            grp = root_tracks.get(r, [])
            confirmed = any(t.confirmed for t in grp) or (n >= cfg.min_track_events)
            if confirmed:
                survivors.append(r)

        def start_of(r: int) -> float:
            grp = root_tracks.get(r)
            return min((t.t_start for t in grp), default=0.0) if grp else 0.0
        survivors.sort(key=start_of)
        remap = {r: i for i, r in enumerate(survivors)}

        instance = np.full(prov.shape, -1, dtype=np.int32)
        rl = roots.tolist()
        for i, r in enumerate(rl):
            if r >= 0:
                m = remap.get(r)
                if m is not None:
                    instance[i] = m

        info = {
            "n_confirmed_instances": len(survivors),
            "root_for_instance": {int(remap[r]): int(r) for r in survivors},
            "root_tracks": root_tracks,
        }
        return instance, info


# ===========================================================================
# Output writers
# ===========================================================================

def write_labeled_events(out: Path, seg: Segmenter, instance: np.ndarray,
                         use_csv: bool) -> str:
    t = np.asarray(seg.ev_t, dtype=np.float64)
    x = np.asarray(seg.ev_x, dtype=np.int16)
    y = np.asarray(seg.ev_y, dtype=np.int16)
    p = np.asarray(seg.ev_p, dtype=np.int8)
    inst = instance.astype(np.int32)

    if pd is not None and not use_csv:
        try:
            path = out / "labeled_events.parquet"
            pd.DataFrame({"t": t, "x": x, "y": y, "p": p,
                          "instance_id": inst}).to_parquet(path, index=False)
            return str(path)
        except Exception as e:
            print(f"  parquet write failed ({e}); falling back to CSV")
    path = out / "labeled_events.csv"
    if pd is not None:
        pd.DataFrame({"t": t, "x": x, "y": y, "p": p,
                      "instance_id": inst}).to_csv(path, index=False)
    else:
        arr = np.column_stack([t, x, y, p, inst])
        np.savetxt(path, arr, fmt=["%.6f", "%d", "%d", "%d", "%d"],
                   delimiter=",", header="t,x,y,p,instance_id", comments="")
    return str(path)


def _resample_traj(tracks: list[Track], t0: float, t1: float,
                   dt: float) -> list[dict]:
    """Concatenate + resample the trajectory of every track sharing a root."""
    ts, cxs, cys, vxs, vys = [], [], [], [], []
    for tr in tracks:
        ts += tr.traj_t
        cxs += tr.traj_cx
        cys += tr.traj_cy
        vxs += tr.traj_vx
        vys += tr.traj_vy
    if not ts:
        return []
    tt = np.asarray(ts)
    order = np.argsort(tt)
    tt = tt[order]
    cx = np.asarray(cxs)[order]
    cy = np.asarray(cys)[order]
    vx = np.asarray(vxs)[order]
    vy = np.asarray(vys)[order]
    # Collapse duplicate timestamps (merges can produce them) by averaging.
    uniq, idx = np.unique(tt, return_index=True)
    if uniq.size != tt.size:
        tt, cx, cy, vx, vy = uniq, cx[idx], cy[idx], vx[idx], vy[idx]
    if tt.size == 1:
        return [{"t": round(float(tt[0]), 6), "cx": round(float(cx[0]), 3),
                 "cy": round(float(cy[0]), 3), "vx": round(float(vx[0]), 2),
                 "vy": round(float(vy[0]), 2)}]
    g0 = max(t0, float(tt[0]))
    g1 = min(t1, float(tt[-1]))
    grid = np.arange(g0, g1 + dt * 0.5, dt) if g1 > g0 else tt
    rcx = np.interp(grid, tt, cx)
    rcy = np.interp(grid, tt, cy)
    rvx = np.interp(grid, tt, vx)
    rvy = np.interp(grid, tt, vy)
    return [{"t": round(float(g), 6), "cx": round(float(a), 3),
             "cy": round(float(b), 3), "vx": round(float(c), 2),
             "vy": round(float(d), 2)}
            for g, a, b, c, d in zip(grid, rcx, rcy, rvx, rvy)]


def write_tracks_json(out: Path, seg: Segmenter, instance: np.ndarray,
                      info: dict, t0: float, t1: float) -> None:
    cfg = seg.cfg
    root_tracks: dict[int, list[Track]] = info["root_tracks"]
    inst_t = np.asarray(seg.ev_t)
    tracks_out = []
    for inst_id in range(info["n_confirmed_instances"]):
        root = info["root_for_instance"][inst_id]
        grp = root_tracks.get(root, [])
        mask = instance == inst_id
        n = int(mask.sum())
        if n == 0:
            continue
        ts = inst_t[mask]
        traj = _resample_traj(grp, t0, t1, cfg.traj_dt) if grp else []
        mean_speed = (float(np.mean([math.hypot(s["vx"], s["vy"]) for s in traj]))
                      if traj else 0.0)
        tracks_out.append({
            "id": inst_id,
            "color_rgb": instance_color(inst_id),
            "t_start": round(float(ts.min()), 6),
            "t_end": round(float(ts.max()), 6),
            "n_events": n,
            "mean_speed_px_s": round(mean_speed, 2),
            "trajectory": traj,
        })
    (out / "tracks.json").write_text(json.dumps(tracks_out, indent=2))


def write_colormap(out: Path, n_instances: int) -> None:
    cmap = {"-1": list(BACKGROUND_RGB)}
    for i in range(n_instances):
        cmap[str(i)] = instance_color(i)
    (out / "colormap.json").write_text(json.dumps(cmap, indent=2))


def write_frames(out: Path, seg: Segmenter, instance: np.ndarray,
                 t0: float, t1: float) -> int:
    if Image is None:
        print("  Pillow not available; skipping --render-frames")
        return 0
    cfg = seg.cfg
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    W = cfg.width or seg.width
    H = cfg.height or seg.height
    t = np.asarray(seg.ev_t)
    x = np.asarray(seg.ev_x)
    y = np.asarray(seg.ev_y)
    n_inst = int(instance.max()) + 1 if instance.size else 0
    palette = np.array([BACKGROUND_RGB] + [instance_color(i)
                                           for i in range(max(0, n_inst))],
                       dtype=np.uint8)

    n_frames = max(1, int(math.ceil((t1 - t0) / cfg.frame_dt)))
    written = 0
    for f in range(n_frames):
        a = t0 + f * cfg.frame_dt
        b = a + cfg.frame_dt
        lo = int(np.searchsorted(t, a, "left"))
        hi = int(np.searchsorted(t, b, "left"))
        img = np.zeros((H, W, 3), dtype=np.uint8)
        if hi > lo:
            xs = x[lo:hi]; ys = y[lo:hi]; ins = instance[lo:hi]
            cols = palette[ins + 1]  # -1 -> row 0
            valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            img[ys[valid].astype(np.intp), xs[valid].astype(np.intp)] = cols[valid]
        Image.fromarray(img, "RGB").save(frames_dir / f"frame_{f:05d}.png")
        written += 1
    return written


def write_summary(out: Path, cfg: Config, n_events: int, n_instances: int,
                  runtime: float, extra: dict) -> None:
    summary = {
        "resolved_params": cfg.resolved(),
        "total_events": int(n_events),
        "n_confirmed_instances": int(n_instances),
        "wall_clock_s": round(runtime, 3),
        "throughput_events_per_s": round(n_events / runtime, 1) if runtime > 0 else 0,
        **extra,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


# ===========================================================================
# CLI / main
# ===========================================================================

def parse_args(argv=None) -> Config:
    d = Config()
    ap = argparse.ArgumentParser(
        description="Streaming motion-coherence instance segmentation of an "
                    "event-camera stream.")
    ap.add_argument("--input", default=d.input, help="input events .txt (t x y p)")
    ap.add_argument("--out", default=d.out, help="output directory")
    ap.add_argument("--csv", action="store_true", help="write CSV instead of parquet")
    ap.add_argument("--render-frames", action="store_true",
                    help="write one PNG per frame_dt slice into frames/")
    ap.add_argument("--limit", type=int, default=d.limit,
                    help="process at most N events (0 = all)")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--progress-every", type=int, default=d.progress_every)

    ap.add_argument("--dt-window", type=float, default=d.dt_window)
    ap.add_argument("--k", type=int, default=d.k)
    ap.add_argument("--beta", type=float, default=d.beta)
    ap.add_argument("--grid-cell", type=int, default=d.grid_cell)
    ap.add_argument("--min-nbrs", type=int, default=d.min_nbrs)
    ap.add_argument("--lag", type=float, default=d.lag)

    ap.add_argument("--gate-px", type=float, default=d.gate_px)
    ap.add_argument("--vel-tol", type=float, default=d.vel_tol)
    ap.add_argument("--lam", "--lambda", dest="lam", type=float, default=d.lam)
    ap.add_argument("--min-track-events", type=int, default=d.min_track_events)
    ap.add_argument("--track-timeout", type=float, default=d.track_timeout)
    ap.add_argument("--v-max", type=float, default=d.v_max)

    ap.add_argument("--split-px", type=float, default=d.split_px)
    ap.add_argument("--split-every", type=int, default=d.split_every)
    ap.add_argument("--min-track-lifetime-events", type=int,
                    default=d.min_track_lifetime_events)
    ap.add_argument("--merge-vel-tol", type=float, default=d.merge_vel_tol)
    ap.add_argument("--merge-px", type=float, default=d.merge_px)

    ap.add_argument("--traj-dt", type=float, default=d.traj_dt)
    ap.add_argument("--frame-dt", type=float, default=d.frame_dt)
    ap.add_argument("--width", type=int, default=d.width)
    ap.add_argument("--height", type=int, default=d.height)

    a = ap.parse_args(argv)
    return Config(
        input=a.input, out=a.out, csv=a.csv, render_frames=a.render_frames,
        limit=a.limit, seed=a.seed, progress_every=a.progress_every,
        dt_window=a.dt_window, k=a.k, beta=a.beta, grid_cell=a.grid_cell,
        min_nbrs=a.min_nbrs, lag=a.lag, gate_px=a.gate_px, vel_tol=a.vel_tol,
        lam=a.lam, min_track_events=a.min_track_events,
        track_timeout=a.track_timeout, v_max=a.v_max, split_px=a.split_px,
        split_every=a.split_every,
        min_track_lifetime_events=a.min_track_lifetime_events,
        merge_vel_tol=a.merge_vel_tol, merge_px=a.merge_px,
        traj_dt=a.traj_dt, frame_dt=a.frame_dt, width=a.width, height=a.height,
    )


def main(argv=None) -> int:
    cfg = parse_args(argv)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    in_path = Path(cfg.input)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    print("=" * 68)
    print("Streaming event-stream instance segmentation")
    print("=" * 68)
    print("Resolved configuration:")
    for k, v in cfg.resolved().items():
        print(f"  {k:28s} = {v}")
    print("-" * 68)

    seg = Segmenter(cfg)
    t_start = time.perf_counter()
    n = 0
    for (t, x, y, p) in iter_events(cfg.input, limit=cfg.limit):
        seg.process_event(t, x, y, p)
        n += 1
        if cfg.progress_every and n % cfg.progress_every == 0:
            el = time.perf_counter() - t_start
            print(f"  processed {n:>10,} events  "
                  f"({n/el:,.0f} ev/s, {len(seg.tracks)} active tracks)")
    seg.finalize()
    runtime = time.perf_counter() - t_start

    if n == 0:
        print("ERROR: no events parsed.", file=sys.stderr)
        return 3

    t0 = seg.ev_t[0]
    t1 = seg.ev_t[-1]
    print("-" * 68)
    print(f"Core loop done: {n:,} events in {runtime:.2f}s "
          f"({n/runtime:,.0f} ev/s). Resolving labels…")

    instance, info = seg.resolve_labels()
    n_inst = info["n_confirmed_instances"]

    print(f"Confirmed instances: {n_inst}")
    print("Writing outputs…")
    print(f"  {write_labeled_events(out, seg, instance, cfg.csv)}")
    write_tracks_json(out, seg, instance, info, t0, t1)
    print(f"  {out / 'tracks.json'}")
    write_colormap(out, n_inst)
    print(f"  {out / 'colormap.json'}")
    n_frames = 0
    if cfg.render_frames:
        n_frames = write_frames(out, seg, instance, t0, t1)
        print(f"  {out / 'frames'}/  ({n_frames} PNGs)")

    n_labeled = int((instance >= 0).sum())
    write_summary(out, cfg, n, n_inst, runtime, extra={
        "duration_s": round(float(t1 - t0), 6),
        "sensor_width": cfg.width or seg.width,
        "sensor_height": cfg.height or seg.height,
        "events_assigned_to_instances": n_labeled,
        "events_background": int(n - n_labeled),
        "frames_written": n_frames,
    })
    print(f"  {out / 'summary.json'}")
    print("-" * 68)
    print(f"DONE. {n_inst} instances, {n_labeled:,}/{n:,} events assigned, "
          f"{runtime:.2f}s, {n/runtime:,.0f} ev/s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
