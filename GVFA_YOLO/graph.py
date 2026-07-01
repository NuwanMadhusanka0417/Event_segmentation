"""Causal ellipsoid multigraph + FPE node encoding (aligned with GVFA/segment.py)."""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

SENSOR = (346, 260)

SPATIAL_R_XY_FRAC = 0.04
SPATIAL_R_T_MS = 5.0
SPATIAL_MMAX = 16

TEMPORAL_R_XY_FRAC = 0.01
TEMPORAL_R_T_MS = 40.0
TEMPORAL_MMAX = 12

POS_BW = 1.0
TIME_BW = 0.5
SEED = 0


def _ellipsoid_metric(dx, dy, dt_ms, r_xy, r_t):
    return np.hypot(dx, dy) / r_xy + np.abs(dt_ms) / r_t


def edge_features_spatial(rec, src, x, y, t):
    dx = x[rec] - x[src]
    dy = y[rec] - y[src]
    dt = t[rec] - t[src]
    return np.stack([dx, dy, dt], axis=1).astype(np.float64)


def edge_features_temporal(rec, src, x, y, t, p):
    dx = x[rec] - x[src]
    dy = y[rec] - y[src]
    dt = t[rec] - t[src]
    dp = p[rec] - p[src]
    dt_safe = np.where(np.abs(dt) > 1e-12, dt, np.sign(dt) * 1e-12 + 1e-12)
    return np.stack([dx, dy, dt, dx / dt_safe, dy / dt_safe, dp], axis=1).astype(np.float64)


def _build_causal_ellipsoid_edges(t, x, y, r_xy, r_t, mmax):
    t_ms = (t - t[0]) * 1e3
    xy = np.stack([x, y], axis=1).astype(np.float64)

    nn = NearestNeighbors(radius=r_xy).fit(xy)
    dists_xy, idxs = nn.radius_neighbors(xy, return_distance=True)

    rec, src = [], []
    for j in range(len(t)):
        cand = idxs[j]
        if cand.size == 0:
            continue
        past_mask = t[cand] < t[j]
        past = cand[past_mask]
        if past.size == 0:
            continue

        dx = x[j] - x[past]
        dy = y[j] - y[past]
        dt = t_ms[j] - t_ms[past]
        metric = _ellipsoid_metric(dx, dy, dt, r_xy, r_t)
        keep = metric < 1.0
        past = past[keep]
        metric = metric[keep]
        if past.size == 0:
            continue
        if past.size > mmax:
            order = np.argsort(metric)[:mmax]
            past = past[order]

        rec.append(np.full(past.size, j, dtype=np.int64))
        src.append(past.astype(np.int64))

    if rec:
        rec = np.concatenate(rec)
        src = np.concatenate(src)
    else:
        rec = np.empty(0, np.int64)
        src = np.empty(0, np.int64)

    edge_index = torch.from_numpy(np.stack([rec, src], axis=0)).long()
    return edge_index, rec, src


def build_multigraph(t, x, y, p, sensor=SENSOR,
                     spatial_r_xy_frac=SPATIAL_R_XY_FRAC,
                     spatial_r_t_ms=SPATIAL_R_T_MS,
                     spatial_mmax=SPATIAL_MMAX,
                     temporal_r_xy_frac=TEMPORAL_R_XY_FRAC,
                     temporal_r_t_ms=TEMPORAL_R_T_MS,
                     temporal_mmax=TEMPORAL_MMAX):
    """Build spatial (E_s) and temporal (E_t) causal ellipsoid graphs.

    Spatial edge features  — (Δx, Δy, Δt).
    Temporal edge features — (Δx, Δy, Δt, Δx/Δt, Δy/Δt, Δp).

    `t` must be in seconds (monotone).  Returns edge indices and edge attrs.
    """
    w, _ = sensor
    r_xy_s = spatial_r_xy_frac * w
    r_xy_t = temporal_r_xy_frac * w

    edge_spatial, rec_s, src_s = _build_causal_ellipsoid_edges(
        t, x, y, r_xy_s, spatial_r_t_ms, spatial_mmax)
    edge_temporal, rec_t, src_t = _build_causal_ellipsoid_edges(
        t, x, y, r_xy_t, temporal_r_t_ms, temporal_mmax)

    attr_spatial = edge_features_spatial(rec_s, src_s, x, y, t)
    attr_temporal = edge_features_temporal(rec_t, src_t, x, y, t, p)

    return (edge_spatial, edge_temporal,
            attr_spatial, attr_temporal,
            rec_s, src_s)


def _fpe_channel(values, base_phase, scale, chunk=4096):
    v = (values * scale).astype(np.float32)
    out = np.empty((len(v), base_phase.shape[0]), dtype=np.float32)
    bp = torch.from_numpy(base_phase)
    for s in range(0, len(v), chunk):
        vb = torch.from_numpy(v[s:s + chunk])
        ang = vb[:, None] * bp[None, :]
        code = torch.fft.ifft(torch.exp(1j * ang)).real
        out[s:s + chunk] = code.numpy()
    return out


def fpe_encode(x, y, t, dim, sensor=SENSOR, seed=SEED):
    """FPE-encode node features {x, y, t} -> L2-normalized hypervector [N, D]."""
    rng = np.random.default_rng(seed)
    w, h = sensor
    t_ms = (t - t[0]) * 1e3
    t_span = max(t_ms.max(), 1e-6)

    channels = [
        (x / w, POS_BW),
        (y / h, POS_BW),
        (t_ms / t_span, TIME_BW),
    ]

    bundle = np.zeros((len(x), dim), dtype=np.float32)
    for vals, scale in channels:
        base_phase = rng.uniform(0, 2 * np.pi, size=dim).astype(np.float32)
        bundle += _fpe_channel(np.asarray(vals, dtype=np.float64), base_phase, float(scale))

    h_in = torch.from_numpy(bundle)
    return torch.nn.functional.normalize(h_in, p=2, dim=1)
