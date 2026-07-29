"""
segment.py - Streaming instance segmentation of moving objects from an
event-camera stream, built on the GVFA (Graph Vector Function Architecture)
encoder.

The scene is unlabeled. Grouping comes entirely from space-time-velocity
coherence encoded as hypervectors (VSA / HRR) and refined by a few hops of the
GVFA GraphCNN, then clustered online by cosine similarity inside each spatial
connected component.

NODE FEATURES (both graphs): absolute {x, y, t, p} via FPE codebooks.  Relative
position/motion appears only in edge Δ terms (Eq. 5 / Eq. 6).

PIPELINE
    load_events      -> read txt, take a time WINDOW (fast)
    build_multigraph -> spatial + temporal ellipsoid graphs (causal, past-only)
    fpe_encode       -> FPE codebook node hypervectors from {x, y, t, p}
    encode_nodes     -> FPE codebook edge_H + GVFA per graph; hop-bundle; concat
    assign           -> streaming prototype clustering, factored by component
    save             -> events_labeled.parquet + seg.png + console summary

PARAMETERS (constants below; edit them in place)
    WINDOW_MS   time slice processed, in milliseconds (default 30).
                Increase to see more motion / more objects; cost grows with the
                number of events in the slice. Set to None for the whole file.
    SENSOR      sensor size (W, H) in pixels; spatial R_XY scales from W.
    SPATIAL_*   ellipsoid axes for the spatial graph (4% W, 5 ms, max 16 nbrs).
    TEMPORAL_*  ellipsoid axes for the temporal graph (1% W, 40 ms, max 12 nbrs).
    D           hypervector dimensionality per graph branch (default 4000);
                final node vectors are concat(H_spatial, H_temporal) -> 2*D.
    NUM_LAYERS  GraphCNN layers INCLUDING the input layer; 3 => 2 hops. Kept
                small on purpose so each node's vector stays local to its object.
    TAU         cosine threshold to join an existing object (default 0.15).
                Lower => fewer, larger objects (more merging); higher => more
                objects (more splitting).
    ALPHA       prototype update rate for the decayed bundle (default 0.10).
    MIN_EVENTS  objects smaller than this are treated as noise and relabeled to
                background id -1 (default 80). Set to 0 to keep every object.

    Secondary FPE codebook bandwidths / bundle weights — see FPE CODEBOOK CONFIG.

USAGE
    python segment.py --window-ms 60 --num-layers 3
    python segment.py --input events_filtered.txt --tau 0.12 --num-layers 4
Outputs events_labeled.parquet and seg.png in the working directory.
"""

import argparse
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from fpe_codebook import FPECodebook, bundle_weighted, bind_hv
from gvfa_encoder import encode_graph
from aperture import resolve_flow, extract_constraints
from ego_motion import fit_ego_motion, residual_split
from graph_smoothing import (
    compute_prototypes,
    drop_tiny_clusters,
    smooth_labels,
)
from motion_pooling import (
    bundle_hypervectors,
    diagnose_bundling_cosine,
    diagnose_clustering,
    diagnose_node_flow,
    induce_subgraph,
    motion_coarsen,
    supernode_aggregates,
    unpool,
)
from viz_diagnostics import (
    plot_ego_fit,
    plot_flow_raw,
    plot_flow_smoothed,
    plot_flow_resolved,
    plot_orientation_check,
    plot_constraint_votes,
    plot_vsa_vs_hough,
    plot_resolver_comparison,
    plot_residual_split,
    plot_segmentation,
    plot_summary,
    plot_supernodes,
)

# ----------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------
WINDOW_MS  = 1000.0   # time slice to process (ms); None = whole file
SENSOR     = (346, 260)   # (W, H) in pixels

# Spatial graph: ellipsoid elongated in (x, y) — local spatial structure
SPATIAL_R_XY_FRAC = 0.04  # R_XY = this fraction of sensor width
SPATIAL_R_T_MS    = 5.0   # semi-minor axis along time (ms)
SPATIAL_MMAX      = 16    # max past spatial neighbours per node

# Temporal graph: ellipsoid elongated in t — motion over longer horizons
TEMPORAL_R_XY_FRAC = 0.01  # R_XY = this fraction of sensor width
TEMPORAL_R_T_MS    = 40.0  # semi-major axis along time (ms)
TEMPORAL_MMAX      = 12    # max past temporal neighbours per node

D          = 4000     # hypervector dimensionality per graph branch (tunable)
NUM_LAYERS = 3        # GraphCNN layers incl. input (tunable via --num-layers)
USE_RESERVOIR = False # no tap buffer / Sigma-Pi
TAU        = 0.15     # cosine merge threshold (tunable)
ALPHA      = 0.15     # prototype update rate (tunable)
MIN_EVENTS = 150      # min cluster size (flat path); smaller -> background

# === STAGE 1: flow regularization ===
FLOW_SMOOTH_ITERS = 4      # smoothing iterations
FLOW_KEEP         = 0.5    # self weight per iteration

# === APERTURE RESOLVER (Track A / Track B) ===
MOTION_RESOLVER    = "lk"        # "none" | "lk" | "affine" | "vsa"
COMPARE_METHODS    = ["none", "lk", "vsa"]
ALL_RESOLVERS      = ["none", "lk", "affine", "vsa"]
SEGMENT_MS         = 60.0        # tile size for --stream-segments
U_MIN              = 1e-6
SPEED_V0           = 50.0        # signed-log knee, px/s
BW_R               = 0.1         # node Cartesian motion codebook bandwidth
W_NODE_MOTION      = 1.0
# Track A
LK_MIN_SUPPORT     = 6
AFFINE_MIN_SUPPORT = 12
CONDITION_MIN_EIG  = 1e-3
LK_HUBER_ITERS     = 3
# Track B
D_VEL              = 512
VEL_GRID_N         = 48
VEL_MAX            = 1500.0
VEL_SIGNED_LOG     = True
BW_VEL             = 0.15
BAND_SIGMA         = 0.08
WEIGHT_FLOOR       = 1e-3
SELF_WEIGHT        = 1.0
CLEANUP_TOPK       = 5
CLEANUP_MIN_CONF   = 0.05
READOUT_CHUNK      = 4096
VSA_VALIDATE       = True
VSA_RESOLVE_IMO_ONLY = False  # optional cost cut (not used unless True + pre-mask)

# === STAGE 2: ego-motion + residual split ===
RES_K      = 3.0           # residual threshold in robust sigmas
IRLS_ITERS = 10

# === STAGE 3: motion-coherent pooling on IMO residuals ===
SIGMA_V          = None   # None => auto (median edge ||v_i-v_j||)
W_MIN            = 0.1    # refuse Graclus merges below this motion affinity
N_COARSEN_LEVELS = 7      # coarsening depth on IMO subgraph
SUPER_R_XY       = 40.0   # (kept for --flat / legacy supernode path)
SUPER_R_T_MS     = 30.0
MIN_SUPER_SIZE   = 3      # mark supernodes with fewer members as noise
LAM              = 1.5    # graph label-smoothing strength
SMOOTH_ITERS     = 5
MIN_CLUSTER_SIZE = 200    # final objects smaller than this -> background
OUT_DIR          = "diag"

# === FPE CODEBOOK CONFIG (tunable) ===
# Per-feature bandwidth (Gaussian kernel length-scale = 1/bandwidth)
BW_X, BW_Y = 0.0333, 0.0333      # scale ~30 px
BW_T       = 1e-5                # scale ~100 ms (over 0..1e6 us)
BW_P       = 1.43                # 2 polarities kept distinct
BW_DX, BW_DY = 0.1, 0.1          # scale ~10 px
BW_DT      = 3.3e-4              # scale ~3 ms
BW_VX, BW_VY = 0.1, 0.1          # scale ~1 signed-log unit
BW_DP      = 1.43

# Node bundle weights — motion-dominant; graph carries space (x/y unused)
# Position in the node HV is redundant (edges already connect neighbours) AND
# harmful: proximity bias pulls apart opposite ends of one person.
W_NODE_X, W_NODE_Y = 0.0, 0.0    # kept at 0; re-enable via CLI for ablations
W_NODE_T           = 0.1
W_NODE_P           = 0.0         # polarity is contrast, not identity

# Spatial edge bundle weights
W_EDGE_S_DX, W_EDGE_S_DY, W_EDGE_S_DT = 0.5, 0.5, 0.3  #                Short-range displacement

# Temporal edge bundle weights (velocity weighted up for motion cue)
W_EDGE_T_DX, W_EDGE_T_DY, W_EDGE_T_DT = 0.5, 0.5, 0.3  # 0.5, 0.5, 0.5        Displacement over longer Δt
W_EDGE_T_VX, W_EDGE_T_VY, W_EDGE_T_DP = 5.0, 5.0, 0.5  # 2.5, 2.5, 1.0   Speed + direction (px/s)

# Radix / signed-log grid params
RADIX_S_T = 250           # sqrt-scale fine radix for time (µs)
RADIX_S_DT_SPATIAL = 32   # radix fine for spatial Δt (µs)
RADIX_S_DT_TEMPORAL = 200 # radix fine for temporal Δt (µs)
SIGNED_LOG_V0 = 100.0     # px/s scale for log(1+|v|/v0) on edge velocity
VEL_LOG_GRID = 0.1        # grid step in log-preconditioned velocity units
VEL_LOG_UMAX = 400        # half-range index for signed-log velocity radix

SEED   = 0
DEVICE = "cpu"


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def load_events(path, window_ms=WINDOW_MS, *, t_start_ms=0.0, t_end_ms=None):
    """Read event rows; optionally slice [t0+t_start_ms, t0+t_start_ms+window_ms).

    t_end_ms overrides window_ms when set (absolute end offset from file t0, ms).
    Returns t (seconds), x, y, p sorted by time.
    """
    data = np.loadtxt(path)
    t, x, y, p = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    order = np.argsort(t, kind="stable")
    t, x, y, p = t[order], x[order], y[order], p[order]
    t0 = t[0]
    if t_end_ms is not None:
        t_lo = t0 + float(t_start_ms) * 1e-3
        t_hi = t0 + float(t_end_ms) * 1e-3
        keep = (t >= t_lo) & (t < t_hi)
    elif window_ms is not None:
        t_lo = t0 + float(t_start_ms) * 1e-3
        t_hi = t_lo + float(window_ms) * 1e-3
        keep = (t >= t_lo) & (t < t_hi)
    else:
        keep = np.ones(len(t), dtype=bool)
    t, x, y, p = t[keep], x[keep], y[keep], p[keep]
    return (t.astype(np.float64),
            x.astype(np.float64), y.astype(np.float64),
            p.astype(np.float64))


def file_time_span_ms(path):
    """Return (t0, span_ms) for the full sorted file (no slicing)."""
    data = np.loadtxt(path)
    t = data[:, 0]
    t = np.sort(t)
    t0 = float(t[0])
    span = float((t[-1] - t0) * 1e3) if t.size else 0.0
    return t0, span


def segment_ranges(span_ms, segment_ms):
    """Non-overlapping [start_ms, end_ms) ranges covering [0, span_ms)."""
    segment_ms = float(segment_ms)
    if segment_ms <= 0 or span_ms <= 0:
        return [(0.0, max(span_ms, segment_ms))]
    n = int(np.ceil(span_ms / segment_ms))
    ranges = []
    for i in range(n):
        a = i * segment_ms
        b = min((i + 1) * segment_ms, span_ms)
        if b > a:
            ranges.append((a, b))
    return ranges


# ----------------------------------------------------------------------------
# 2. CAUSAL ELLIPSOID MULTIGRAPH  (spatial E_s + temporal E_t)
# ----------------------------------------------------------------------------
def _ellipsoid_metric(dx, dy, dt_ms, r_xy, r_t):
    """||dxy||/R_XY + |dt|/R_t  (must be < 1 for an edge)."""
    return np.hypot(dx, dy) / r_xy + np.abs(dt_ms) / r_t


def edge_features_spatial(rec, src, x, y, t):
    """Spatial edge features (Eq. 5): e_ij = (Δx, Δy, Δt) with
    Δx = x_j - x_i, Δy = y_j - y_i, Δt = t_j - t_i  (receiver minus neighbour)."""
    dx = x[rec] - x[src]
    dy = y[rec] - y[src]
    dt = t[rec] - t[src]
    return np.stack([dx, dy, dt], axis=1).astype(np.float64)


def edge_features_temporal(rec, src, x, y, t, p):
    """Temporal edge features (Eq. 6): (Δx, Δy, Δt, Δx/Δt, Δy/Δt, Δp)."""
    dx = x[rec] - x[src]
    dy = y[rec] - y[src]
    dt = t[rec] - t[src]
    dp = p[rec] - p[src]
    dt_safe = np.where(np.abs(dt) > 1e-12, dt, np.sign(dt) * 1e-12 + 1e-12)
    return np.stack([dx, dy, dt, dx / dt_safe, dy / dt_safe, dp], axis=1).astype(np.float64)


def _build_causal_ellipsoid_edges(t, x, y, r_xy, r_t, mmax):
    """Build one directed causal graph under the ellipsoid constraint.

    A directed edge (i -> j) is kept when t_i < t_j and
        ||v_i^xy - v_j^xy|| / R_XY + |t_i - t_j| / R_t < 1.

    For GVFA message passing, edge_index[0] = receiver j (later event),
    edge_index[1] = source i (earlier neighbour). Degree capped at `mmax`,
    keeping neighbours with the smallest ellipsoid metric.
    """
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

    Spatial edges  — Eq. 5 features: (Δx, Δy, Δt).
    Temporal edges — Eq. 6 features: (Δx, Δy, Δt, Δx/Δt, Δy/Δt, Δp).

    Returns
        edge_spatial, edge_temporal : torch.LongTensor [2, E]
        edge_attr_spatial  : np.ndarray [E, 3]
        edge_attr_temporal : np.ndarray [E, 6]
        rec_s, src_s : int arrays for spatial connected components
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


def connected_components(num_nodes, rec, src):
    """Union-find over the undirected neighbour pairs. Returns an int label per
    node. Two objects that move alike but are spatially apart end up in different
    components, which keeps them from merging during assignment."""
    parent = np.arange(num_nodes, dtype=np.int64)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:        # path compression
            parent[a], a = root, parent[a]
        return root

    for a, b in zip(rec.tolist(), src.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    roots = np.array([find(i) for i in range(num_nodes)], dtype=np.int64)
    _, comp = np.unique(roots, return_inverse=True)
    return comp


# ----------------------------------------------------------------------------
# 3. NODE FLOW  (local time-surface plane fit -> normal optical flow)
# ----------------------------------------------------------------------------
def _fit_plane(N, rec, src, x, y, tt, active, ridge):
    """Least-squares fit t ~= a*x + b*y + c per node over {self} U {active past
    neighbours}, solved as batched 3x3 normal equations. Returns coef [N,3] and
    the per-node point count."""
    Sxx = x * x; Sxy = x * y; Syy = y * y
    Sx = x.copy(); Sy = y.copy(); S1 = np.ones(N)
    Stx = tt * x; Sty = tt * y; St = tt.copy()
    ri = rec[active]; xj = x[src[active]]; yj = y[src[active]]; tj = tt[src[active]]
    np.add.at(Sxx, ri, xj * xj); np.add.at(Sxy, ri, xj * yj); np.add.at(Syy, ri, yj * yj)
    np.add.at(Sx, ri, xj);       np.add.at(Sy, ri, yj);       np.add.at(S1, ri, 1.0)
    np.add.at(Stx, ri, tj * xj); np.add.at(Sty, ri, tj * yj); np.add.at(St, ri, tj)

    M = np.empty((N, 3, 3))
    M[:, 0, 0] = Sxx + ridge; M[:, 0, 1] = Sxy;         M[:, 0, 2] = Sx
    M[:, 1, 0] = Sxy;         M[:, 1, 1] = Syy + ridge; M[:, 1, 2] = Sy
    M[:, 2, 0] = Sx;          M[:, 2, 1] = Sy;          M[:, 2, 2] = S1 + ridge
    rhs = np.stack([Stx, Sty, St], axis=1)
    coef = np.linalg.solve(M, rhs)
    return coef, S1


def node_flow(t, x, y, edge_index, min_pts=5, ridge=1e-9, clip_pct=99.0):
    """Per-event normal optical flow by fitting a local plane to the time surface.

    For each node we fit t ~= a*x + b*y + c over its causal spatio-temporal
    neighbourhood. The plane gradient g = (a, b) (units s/px) gives the normal
    flow v = g / |g|^2 (px/s) - the motion component perpendicular to the local
    edge, which is what a single edge can observe. One robust re-fit drops
    neighbours whose time residual is an outlier, which kills the nearest-
    neighbour aliasing that makes raw dx/dt unusable.

    Returns vx, vy in px/s; nodes with too few / aperture-degenerate support
    get 0.
    """
    rec = edge_index[0].numpy(); src = edge_index[1].numpy()
    N = len(t)
    tt = (t - t[0])                                   # seconds, shifted for stability
    if rec.shape[0] == 0:
        return np.zeros(N), np.zeros(N)

    active = np.ones(rec.shape[0], dtype=bool)
    coef, _ = _fit_plane(N, rec, src, x, y, tt, active, ridge)

    # robust pass: reject neighbours with large plane residual (per-node scale)
    a, b, c = coef[rec, 0], coef[rec, 1], coef[rec, 2]
    resid = tt[src] - (a * x[src] + b * y[src] + c)
    rss = np.zeros(N); cnt = np.zeros(N)
    np.add.at(rss, rec, resid * resid); np.add.at(cnt, rec, 1.0)
    scale = np.sqrt(rss / np.maximum(cnt, 1.0))       # per-node RMS residual
    active = np.abs(resid) <= (2.5 * scale[rec] + 1e-12)
    coef, npts = _fit_plane(N, rec, src, x, y, tt, active, ridge)

    a, b = coef[:, 0], coef[:, 1]
    g2 = a * a + b * b
    vx = np.zeros(N); vy = np.zeros(N)
    ok = (npts >= min_pts) & (g2 > 1e-12)             # drop aperture/degenerate
    vx[ok] = a[ok] / g2[ok]
    vy[ok] = b[ok] / g2[ok]

    # winsorize extreme speeds so a few blow-ups don't dominate normalization
    spd = np.hypot(vx, vy)
    if ok.any():
        cap = np.percentile(spd[ok], clip_pct)
        scl = np.where((spd > cap) & (spd > 0), cap / np.maximum(spd, 1e-12), 1.0)
        vx *= scl; vy *= scl
    return vx, vy   # px/s


def smooth_flow(vx, vy, edge_index_spatial, n_iters=4, keep=0.5):
    """Graph-neighbour regularization of normal flow (Stage 1).

    Each iteration: v_i <- keep * v_i + (1-keep) * mean(v_j over valid spatial nbrs).
    Events with no valid neighbours keep their value. Vectorized via np.add.at.

    Returns (vx_s, vy_s, n_valid_before, n_valid_after).
    """
    vx_s = np.asarray(vx, dtype=np.float64).copy()
    vy_s = np.asarray(vy, dtype=np.float64).copy()
    n = len(vx_s)
    n_valid_before = int((np.hypot(vx_s, vy_s) > 1e-12).sum())

    if edge_index_spatial is None or edge_index_spatial.numel() == 0 or n_iters <= 0:
        return vx_s, vy_s, n_valid_before, n_valid_before

    rec = edge_index_spatial[0].numpy().astype(np.int64)
    src = edge_index_spatial[1].numpy().astype(np.int64)
    # undirected: both directions
    a = np.concatenate([rec, src])
    b = np.concatenate([src, rec])

    for _ in range(n_iters):
        valid = np.hypot(vx_s, vy_s) > 1e-12
        # only aggregate from valid neighbours
        nbr_ok = valid[b]
        aa, bb = a[nbr_ok], b[nbr_ok]
        sx = np.zeros(n, dtype=np.float64)
        sy = np.zeros(n, dtype=np.float64)
        cnt = np.zeros(n, dtype=np.float64)
        if aa.size:
            np.add.at(sx, aa, vx_s[bb])
            np.add.at(sy, aa, vy_s[bb])
            np.add.at(cnt, aa, 1.0)
        has = cnt > 0
        mean_x = np.zeros(n, dtype=np.float64)
        mean_y = np.zeros(n, dtype=np.float64)
        mean_x[has] = sx[has] / cnt[has]
        mean_y[has] = sy[has] / cnt[has]
        new_x = vx_s.copy()
        new_y = vy_s.copy()
        new_x[has] = keep * vx_s[has] + (1.0 - keep) * mean_x[has]
        new_y[has] = keep * vy_s[has] + (1.0 - keep) * mean_y[has]
        vx_s, vy_s = new_x, new_y

    n_valid_after = int((np.hypot(vx_s, vy_s) > 1e-12).sum())
    print(f"[smooth_flow] iters={n_iters} keep={keep}  "
          f"valid {n_valid_before} -> {n_valid_after}")
    return vx_s, vy_s, n_valid_before, n_valid_after


# ----------------------------------------------------------------------------
# 4. FPE CODEBOOK NODE / EDGE ENCODING
# ----------------------------------------------------------------------------
def make_codebooks(sensor=SENSOR, t_span_s=0.06, seed=SEED):
    """Build node and edge FPE codebooks for one processing window."""
    W, H = sensor
    dt_s_max = int(SPATIAL_R_T_MS * 1000) + 1
    dt_t_max = int(TEMPORAL_R_T_MS * 1000) + 1

    # Signed-log residual components span roughly ±log1p(VEL_MAX/SPEED_V0)
    s_max = float(np.log1p(VEL_MAX / SPEED_V0)) + 0.5
    s_idx = int(np.ceil(s_max / 0.1)) + 2

    node = {
        "x": FPECodebook("x", D, BW_X, "integer", vmin=0, vmax=W - 1,
                         phase_dist="gaussian", seed=seed + 1),
        "y": FPECodebook("y", D, BW_Y, "integer", vmin=0, vmax=H - 1,
                         phase_dist="gaussian", seed=seed + 2),
        "t": FPECodebook("t", D, BW_T, "radix", radix_S=1000, vmin=0,
                         vmax=1_000_000, value_grid_step=1.0,
                         phase_dist="gaussian", seed=seed + 3),
        "p": FPECodebook("p", D, BW_P, "integer", vmin=0, vmax=1,
                         phase_dist="gaussian", seed=seed + 4),
        # Cartesian residual motion in signed-log space (no polar singularity at |r|~0)
        "rx": FPECodebook("rx", D, BW_R, "radix", radix_S=16,
                          vmin=-s_idx, vmax=s_idx, value_grid_step=0.1,
                          phase_dist="gaussian", seed=seed + 5),
        "ry": FPECodebook("ry", D, BW_R, "radix", radix_S=16,
                          vmin=-s_idx, vmax=s_idx, value_grid_step=0.1,
                          phase_dist="gaussian", seed=seed + 6),
    }
    edge_dx = FPECodebook("dx", D, BW_DX, "integer", vmin=-W, vmax=W,
                          phase_dist="gaussian", seed=seed + 10)
    edge_dy = FPECodebook("dy", D, BW_DY, "integer", vmin=-H, vmax=H,
                          phase_dist="gaussian", seed=seed + 11)
    edge_spatial = {
        "dx": edge_dx,
        "dy": edge_dy,
        "dt": FPECodebook("dt_s", D, BW_DT, "radix", radix_S=100,
                          vmin=0, vmax=dt_s_max, value_grid_step=1.0,
                          phase_dist="gaussian", seed=seed + 12),
    }
    edge_temporal = {
        "dx": edge_dx,
        "dy": edge_dy,
        "dt": FPECodebook("dt_t", D, BW_DT, "radix", radix_S=200,
                          vmin=0, vmax=dt_t_max, value_grid_step=1.0,
                          phase_dist="gaussian", seed=seed + 13),
        "vx": FPECodebook("vx", D, BW_VX, "signed_log_radix", radix_S=16,
                          vmin=-25, vmax=25, value_grid_step=0.1,
                          signed_log_v0=100.0, phase_dist="gaussian", seed=seed + 14),
        "vy": FPECodebook("vy", D, BW_VY, "signed_log_radix", radix_S=16,
                          vmin=-25, vmax=25, value_grid_step=0.1,
                          signed_log_v0=100.0, phase_dist="gaussian", seed=seed + 15),
        "dp": FPECodebook("dp", D, BW_DP, "integer", vmin=-1, vmax=1,
                          phase_dist="gaussian", seed=seed + 16),
    }
    w_spatial = {"dx": W_EDGE_S_DX, "dy": W_EDGE_S_DY, "dt": W_EDGE_S_DT}
    w_temporal = {
        "dx": W_EDGE_T_DX, "dy": W_EDGE_T_DY, "dt": W_EDGE_T_DT,
        "vx": W_EDGE_T_VX, "vy": W_EDGE_T_VY, "dp": W_EDGE_T_DP,
    }
    return node, edge_spatial, edge_temporal, w_spatial, w_temporal


def _slog(w, v0=SPEED_V0):
    w = np.asarray(w, dtype=np.float64)
    return np.sign(w) * np.log1p(np.abs(w) / v0)


def fpe_encode(x, y, t, p, codebooks, rx=None, ry=None, motion_valid=None,
               *, w_motion=None, w_t=None, w_x=None, w_y=None, w_p=None):
    """FPE node features -> L2-normalized [N, D].

    Motion uses 2D Cartesian SSP on signed-log residual components:
        z_motion = bind(encode(slog(rx)), encode(slog(ry)))
    Unresolved / invalid motion -> z_motion = 0.
    """
    # REASON: polar (speed, direction) has a singularity at |r|~0 — direction is
    # pure noise for every near-zero-residual BACKGROUND event (largest class).
    # Cartesian SSP has no singularity.
    t_us = (t - t[0]) * 1e6
    wm = W_NODE_MOTION if w_motion is None else w_motion
    wt = W_NODE_T if w_t is None else w_t
    wx = W_NODE_X if w_x is None else w_x
    wy = W_NODE_Y if w_y is None else w_y
    wp = W_NODE_P if w_p is None else w_p

    terms = [
        (codebooks["x"].encode(x), wx),
        (codebooks["y"].encode(y), wy),
        (codebooks["t"].encode(t_us, interpolate=True), wt),
        (codebooks["p"].encode(p), wp),
    ]

    n = len(t)
    if rx is not None and ry is not None and "rx" in codebooks and wm != 0:
        rx = np.asarray(rx, dtype=np.float64)
        ry = np.asarray(ry, dtype=np.float64)
        if motion_valid is None:
            motion_valid = np.isfinite(rx) & np.isfinite(ry)
        motion_valid = np.asarray(motion_valid, dtype=bool)
        z_motion = torch.zeros((n, D), dtype=torch.float32)
        if motion_valid.any():
            sx = _slog(rx[motion_valid])
            sy = _slog(ry[motion_valid])
            zx = codebooks["rx"].encode(sx, interpolate=True)
            zy = codebooks["ry"].encode(sy, interpolate=True)
            z_motion[motion_valid] = bind_hv(zx, zy)
        n_zero = int((~motion_valid).sum())
        print(f"[fpe_encode] Cartesian motion: valid={int(motion_valid.sum())}  "
              f"zeroed={n_zero}  W_NODE_MOTION={wm}")
        terms.append((z_motion, wm))

    return bundle_weighted(terms)


# ----------------------------------------------------------------------------
# 5. GVFA ENCODER  (edge-conditioned, graphcnnVSA_Binding_FULL_new)
# ----------------------------------------------------------------------------
def encode_nodes(x_hv, edge_index, edge_attr, edge_codebooks, edge_weights,
                 graph_kind, num_layers=NUM_LAYERS):
    """Edge-codebook GVFA -> contextual node hypervectors H [N, D]."""
    return encode_graph(
        x_hv, edge_index, edge_attr,
        graph_kind=graph_kind,
        edge_codebooks=edge_codebooks,
        edge_weights=edge_weights,
        num_layers=num_layers,
        edge_feat_dim=edge_attr.shape[1],
        device=DEVICE,
        use_reservoir=USE_RESERVOIR,
        rng_seed=SEED,
    )


def encode_nodes_multigraph(x_hv, edge_spatial, attr_spatial,
                            edge_temporal, attr_temporal,
                            cb_spatial, cb_temporal, w_spatial, w_temporal,
                            num_layers=NUM_LAYERS):
    """GVFA on spatial + temporal graphs; concat -> [N, 2*D]."""
    H_spatial = encode_nodes(
        x_hv, edge_spatial, attr_spatial, cb_spatial, w_spatial, "spatial",
        num_layers)
    H_temporal = encode_nodes(
        x_hv, edge_temporal, attr_temporal, cb_temporal, w_temporal, "temporal",
        num_layers)
    return torch.cat([H_spatial, H_temporal], dim=1)


# ----------------------------------------------------------------------------
# 6. STREAMING ASSIGNMENT (factored by component)
# ----------------------------------------------------------------------------
def assign(H, t, components, tau=TAU, alpha=ALPHA, min_events=MIN_EVENTS):
    """Iterate events in time order. For each event, cosine-match its hypervector
    against the prototypes of its OWN spatial component; join the best if >= tau,
    else open a new object. Update the matched prototype with a decayed bundle
    P <- normalize((1-alpha) P + alpha h). Objects smaller than `min_events` are
    relabeled to background (-1). Returns an object id per event (0..K-1, or -1)."""
    Hc = H - H.mean(dim=0, keepdim=True)   # remove common-mode / consensus
    Hn = torch.nn.functional.normalize(Hc, p=2, dim=1).numpy().astype(np.float32)
    order = np.argsort(t, kind="stable")

    protos = []                 # list of unit vectors [D]
    by_comp = {}                # component id -> list of prototype indices
    obj_id = np.full(len(t), -1, dtype=np.int64)

    for i in order:
        h = Hn[i]
        c = int(components[i])
        cand = by_comp.get(c, [])

        best, best_sim = -1, -1.0
        for pi in cand:
            sim = float(protos[pi] @ h)
            if sim > best_sim:
                best_sim, best = sim, pi

        if best_sim >= tau:
            obj_id[i] = best
            p = (1 - alpha) * protos[best] + alpha * h
            protos[best] = p / (np.linalg.norm(p) + 1e-12)
        else:
            pid = len(protos)
            protos.append(h.copy())
            by_comp.setdefault(c, []).append(pid)
            obj_id[i] = pid

    # noise cleanup: small objects -> background (-1)
    if min_events > 0:
        ids, counts = np.unique(obj_id, return_counts=True)
        small = set(ids[counts < min_events].tolist())
        if small:
            obj_id = np.array([-1 if o in small else o for o in obj_id],
                              dtype=np.int64)

    # compact surviving (non-background) ids to 0..K-1, keep -1 as -1
    keep = obj_id >= 0
    if keep.any():
        _, comp_ids = np.unique(obj_id[keep], return_inverse=True)
        obj_id[keep] = comp_ids
    return obj_id


# ----------------------------------------------------------------------------
# 7. SAVE + SUMMARY
# ----------------------------------------------------------------------------
def save(t, x, y, p, obj_id, parquet="events_labeled.parquet", png="seg.png",
         tau=TAU):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"t": t, "x": x.astype(int), "y": y.astype(int),
                       "p": p.astype(int), "object_id": obj_id})
    try:
        df.to_parquet(parquet, index=False)
    except Exception as e:                      # pyarrow missing -> csv fallback
        parquet = parquet.replace(".parquet", ".csv")
        df.to_csv(parquet, index=False)
        print(f"[warn] parquet unavailable ({e}); wrote {parquet}")

    ids, counts = np.unique(obj_id, return_counts=True)
    # background is label 0 (new path) or -1 (flat path)
    is_bg = ids <= 0
    n_obj = int((~is_bg).sum())
    n_bg = int(counts[is_bg].sum()) if is_bg.any() else 0
    order = np.argsort(-counts)
    print(f"\n#objects found: {n_obj}  (+ {n_bg} background/noise events)  "
          f"over {len(obj_id)} events")
    for k in order:
        tag = "  <- background/noise" if ids[k] <= 0 else ""
        print(f"  object {ids[k]:3d}: {counts[k]:6d} events{tag}")

    # scatter coloured by object id (background drawn first, in grey)
    plt.figure(figsize=(9, 7))
    bg = obj_id <= 0
    if bg.any():
        plt.scatter(x[bg], y[bg], c="0.82", s=2, linewidths=0, label="background")
    fg = ~bg
    if fg.any():
        plt.scatter(x[fg], y[fg], c=obj_id[fg], s=2, cmap="tab20", linewidths=0)
    plt.gca().invert_yaxis()
    plt.title(f"{n_obj} objects  |  {len(obj_id)} events  |  "
              f"window={WINDOW_MS} ms  tau={tau}")
    plt.xlabel("x"); plt.ylabel("y")
    if fg.any():
        plt.colorbar(label="object_id")
    plt.tight_layout()
    plt.savefig(png, dpi=130)
    print(f"wrote {parquet} and {png}")


# ----------------------------------------------------------------------------
# SUPERNODE MULTIGRAPH (reuse causal ellipsoid builders on centroids)
# ----------------------------------------------------------------------------
def build_supernode_multigraph(t_s, x_s, y_s, p_s,
                               super_r_xy=SUPER_R_XY,
                               super_r_t_ms=SUPER_R_T_MS,
                               spatial_mmax=SPATIAL_MMAX,
                               temporal_mmax=TEMPORAL_MMAX):
    """Causal ellipsoid multigraph over supernode centroids.

    Spatial / temporal axes keep the same aspect as the event-level graphs,
    scaled so the spatial R_XY and temporal R_t match the exposed SUPER_* knobs.
    Edge attributes use the same definitions as the event graphs.
    """
    # Preserve event-level ellipsoid aspect: spatial wide+short-t, temporal narrow+long-t
    r_xy_s = super_r_xy
    r_t_s = super_r_t_ms * (SPATIAL_R_T_MS / TEMPORAL_R_T_MS)
    r_xy_t = super_r_xy * (TEMPORAL_R_XY_FRAC / SPATIAL_R_XY_FRAC)
    r_t_t = super_r_t_ms

    edge_spatial, rec_s, src_s = _build_causal_ellipsoid_edges(
        t_s, x_s, y_s, r_xy_s, r_t_s, spatial_mmax)
    edge_temporal, rec_t, src_t = _build_causal_ellipsoid_edges(
        t_s, x_s, y_s, r_xy_t, r_t_t, temporal_mmax)

    attr_spatial = edge_features_spatial(rec_s, src_s, x_s, y_s, t_s)
    attr_temporal = edge_features_temporal(rec_t, src_t, x_s, y_s, t_s, p_s)
    return (edge_spatial, edge_temporal,
            attr_spatial, attr_temporal,
            rec_s, src_s)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def _resolver_cfg(args, *, vsa_validate=None):
    vv = args.vsa_validate if vsa_validate is None else vsa_validate
    return {
        "U_MIN": U_MIN,
        "SPEED_V0": SPEED_V0,
        "LK_MIN_SUPPORT": args.lk_min_support,
        "AFFINE_MIN_SUPPORT": AFFINE_MIN_SUPPORT,
        "CONDITION_MIN_EIG": args.condition_min_eig,
        "LK_HUBER_ITERS": LK_HUBER_ITERS,
        "D_VEL": args.d_vel,
        "VEL_GRID_N": args.vel_grid_n,
        "VEL_MAX": VEL_MAX,
        "VEL_SIGNED_LOG": VEL_SIGNED_LOG,
        "BW_VEL": BW_VEL,
        "BAND_SIGMA": args.band_sigma,
        "WEIGHT_FLOOR": WEIGHT_FLOOR,
        "SELF_WEIGHT": SELF_WEIGHT,
        "CLEANUP_TOPK": args.cleanup_topk,
        "CLEANUP_MIN_CONF": args.cleanup_min_conf,
        "READOUT_CHUNK": READOUT_CHUNK,
        "VSA_VALIDATE": vv,
        "SEED": SEED,
    }


def _segment_dir_name(t_start_ms, t_end_ms):
    return f"t{int(round(t_start_ms)):04d}_{int(round(t_end_ms)):04d}"


def _run_window(
    args,
    t, x, y, p,
    methods,
    out_dir,
    *,
    full_diag,
    compare_resolvers,
    seg_label=None,
    save_labels=False,
):
    """Ego+VSA pipeline on one event window. Returns compare_rows, fig_paths_by_method."""
    import time as _time
    import os

    if len(t) == 0:
        print("[skip] empty event window")
        return [], {}

    print("building spatial + temporal ellipsoid multigraph ...")
    (edge_spatial, edge_temporal,
     attr_spatial, attr_temporal,
     rec, src) = build_multigraph(t, x, y, p)
    print(f"  spatial: {edge_spatial.shape[1]}  temporal: {edge_temporal.shape[1]}")
    comp = connected_components(len(t), rec, src)

    t_span = max((t.max() - t.min()), 1e-9)
    node_cb, cb_spatial, cb_temporal, w_spatial, w_temporal = make_codebooks(
        SENSOR, t_span, seed=SEED)

    sigma_v = args.sigma_v if args.sigma_v is not None else SIGMA_V

    print("STAGE 0: node_flow (RAW normal flow) ...")
    vx_raw, vy_raw = node_flow(t, x, y, edge_temporal)
    diagnose_node_flow(vx_raw, vy_raw)

    p1 = None
    if full_diag:
        p1 = plot_flow_raw(x, y, vx_raw, vy_raw, out_dir)

    compare_rows = []
    fig_paths_by_method = {}
    vsa_validate = args.vsa_validate if full_diag else False

    for method in methods:
        mt0 = _time.time()
        cfg = _resolver_cfg(args, vsa_validate=vsa_validate)
        print(f"\n===== RESOLVER = {method} =====")
        if seg_label:
            print(f"  segment {seg_label}  full_diag={full_diag}")

        if compare_resolvers and not args.stream_segments:
            sub = os.path.join(out_dir, f"resolver_{method}")
        elif args.stream_segments or len(methods) > 1:
            sub = os.path.join(out_dir, method)
            if seg_label:
                sub = os.path.join(sub, seg_label)
        else:
            sub = out_dir
        os.makedirs(sub, exist_ok=True)

        vx_res, vy_res, resolved_mask, rinfo = resolve_flow(
            x, y, vx_raw, vy_raw, [edge_spatial, edge_temporal],
            method=method, cfg=cfg,
        )
        vx_s, vy_s, n_vb, n_va = smooth_flow(
            vx_res, vy_res, edge_spatial,
            n_iters=args.flow_smooth_iters, keep=FLOW_KEEP)

        p2 = p11 = p12 = p3 = p4 = p5 = p13 = p14 = None
        if full_diag:
            p2 = plot_flow_smoothed(
                x, y, vx_s, vy_s, sub,
                n_iters=args.flow_smooth_iters, keep=FLOW_KEEP,
                n_valid_before=n_vb, n_valid_after=n_va)
            p11 = plot_flow_resolved(
                x, y, vx_res, vy_res, resolved_mask, sub, rinfo)

        valid_flow = np.hypot(vx_s, vy_s) > 1e-12
        params, residual, ego_info = fit_ego_motion(
            x, y, vx_s, vy_s, SENSOR, n_iters=IRLS_ITERS, valid_mask=valid_flow)
        is_imo, residual, thresh = residual_split(
            residual, edge_spatial, res_k=args.res_k, valid_mask=valid_flow)
        rx, ry = residual[:, 0], residual[:, 1]

        if full_diag:
            _, n_hat, valid_c = extract_constraints(vx_raw, vy_raw, u_min=U_MIN)
            ori_mask = valid_c & is_imo
            if not ori_mask.any():
                ori_mask = valid_c
            p12 = plot_orientation_check(
                n_hat, vx_res, vy_res, ori_mask, sub, rinfo)
            p3 = plot_ego_fit(
                x, y, params, residual, SENSOR, sub,
                res_k=args.res_k, thresh=thresh,
                inlier_rms=ego_info["inlier_rms"], info=ego_info)
            p4 = plot_residual_split(x, y, is_imo, residual, thresh, sub)
            if method == "vsa" and rinfo.get("C_sparse") is not None:
                p13 = plot_constraint_votes(x, y, is_imo, rinfo, sub)
                if rinfo.get("v_hough") is not None:
                    p14 = plot_vsa_vs_hough(
                        np.stack([vx_res, vy_res], 1), rinfo["v_hough"],
                        resolved_mask, sub, rinfo)
                for k in ("C_sparse", "V_sparse", "Z", "codebook", "grid_coords"):
                    rinfo.pop(k, None)

        motion_ok = resolved_mask & valid_flow & np.isfinite(rx) & np.isfinite(ry)
        x_hv = fpe_encode(
            x, y, t, p, node_cb, rx=rx, ry=ry, motion_valid=motion_ok,
            w_motion=args.w_node_motion, w_t=args.w_node_t,
            w_x=args.w_node_x, w_y=args.w_node_y, w_p=args.w_node_p,
        )
        H_events = encode_nodes_multigraph(
            x_hv, edge_spatial, attr_spatial, edge_temporal, attr_temporal,
            cb_spatial, cb_temporal, w_spatial, w_temporal, args.num_layers)

        labels = np.zeros(len(t), dtype=np.int64)
        cluster_full = np.full(len(t), -1, dtype=np.int64)
        cinfo = {"sigma_v": float("nan"), "w_min": args.w_min,
                 "n_levels": args.n_coarsen_levels, "C": 0,
                 "sizes": np.array([0]), "n_rejected_total": 0}
        n_imo = int(is_imo.sum())
        if n_imo > 0:
            edge_s_imo, sub_idx, _ = induce_subgraph(edge_spatial, is_imo)
            edge_t_imo, _, _ = induce_subgraph(edge_temporal, is_imo)
            cluster_id, cinfo = motion_coarsen(
                n_imo, edge_s_imo, edge_t_imo, rx[sub_idx], ry[sub_idx],
                n_levels=args.n_coarsen_levels, sigma_v=sigma_v,
                w_min=args.w_min, seed=SEED,
            )
            cinfo["w_min"] = args.w_min
            cinfo["n_levels"] = args.n_coarsen_levels
            cluster_full[sub_idx] = cluster_id
            H_imo = H_events[sub_idx]
            H_s_bundle = bundle_hypervectors(H_imo[:, :D], cluster_id)
            H_t_bundle = bundle_hypervectors(H_imo[:, D:], cluster_id)
            H_super = torch.cat([H_s_bundle, H_t_bundle], dim=1)
            diagnose_bundling_cosine(H_super, seed=SEED, label="bundling")
            C = H_super.shape[0]
            if edge_s_imo.numel() > 0:
                rec_s = cluster_id[edge_s_imo[0].numpy()]
                src_s = cluster_id[edge_s_imo[1].numpy()]
                comp_super = connected_components(C, rec_s, src_s)
            else:
                comp_super = np.zeros(C, dtype=np.int64)
            agg = supernode_aggregates(
                x[sub_idx], y[sub_idx], t[sub_idx], p[sub_idx],
                rx[sub_idx], ry[sub_idx], cluster_id)
            labels_super = assign(
                H_super, agg["t"], comp_super, tau=args.tau, min_events=0)
            small = agg["counts"] < MIN_SUPER_SIZE
            if small.any():
                labels_super = labels_super.copy()
                labels_super[small] = -1
            labels_imo = unpool(labels_super, cluster_id)
            out = np.zeros(n_imo, dtype=np.int64)
            keep = labels_imo >= 0
            if keep.any():
                _, compact = np.unique(labels_imo[keep], return_inverse=True)
                out[keep] = compact + 1
            labels[sub_idx] = out

        if full_diag:
            p5 = plot_supernodes(
                x, y, cluster_full[is_imo] if n_imo else np.array([]),
                is_imo, sub, cinfo)

        protos = compute_prototypes(H_events, labels)
        labels = smooth_labels(
            labels, H_events, protos,
            edge_index_list=[edge_spatial, edge_temporal],
            lam=args.lam, n_iters=SMOOTH_ITERS,
        )
        labels = drop_tiny_clusters(labels, MIN_CLUSTER_SIZE, background=0)

        runtime = _time.time() - mt0
        ids, counts = np.unique(labels, return_counts=True)
        n_obj = int((ids > 0).sum())
        largest = float(counts[ids > 0].max() / len(labels)) if n_obj else 0.0
        seg_extra = {"segment": seg_label} if seg_label else {}
        p6 = plot_segmentation(
            x, y, labels, sub,
            tau=args.tau, num_layers=args.num_layers,
            lam=args.lam, smooth_iters=SMOOTH_ITERS, runtime_s=runtime,
            extra_box={
                "MOTION_RESOLVER": method,
                "resolved %": f"{100*rinfo['resolved_frac']:.1f}",
                "orient_corr": f"{rinfo['orientation_corr']:.4f}",
                "W_NODE_MOTION": args.w_node_motion,
                **seg_extra,
            })

        if full_diag:
            paths = [p1, p2, p3, p4, p5, p6, p11, p12]
            if p13:
                paths.append(p13)
            if p14:
                paths.append(p14)
            plot_summary(paths, sub)

        if save_labels and not compare_resolvers:
            save(t, x, y, p, labels, tau=args.tau)

        row = {
            "method": method,
            "segment": seg_label or "",
            "resolved_frac": rinfo["resolved_frac"],
            "orientation_corr": rinfo["orientation_corr"],
            "median_speed": rinfo["median_speed_after"],
            "ego_inlier_rms": ego_info["inlier_rms"],
            "imo_frac": float(is_imo.mean()),
            "n_models_premerge": 0,
            "n_models_final": 0,
            "n_objects": n_obj,
            "largest_object_frac": largest,
            "resolver_runtime_s": rinfo["runtime_s"],
            "total_runtime_s": runtime,
        }
        compare_rows.append(row)
        fig_paths_by_method[method] = {
            "quiver": p11, "split": p4, "seg": p6, "row": row,
        }
        print(f"[summary] method={method}  objects={n_obj}  "
              f"orient_corr={rinfo['orientation_corr']:.4f}  time={runtime:.1f}s")

    return compare_rows, fig_paths_by_method


def main():
    import time as _time
    import csv
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="events_filtered.txt")
    ap.add_argument("--window-ms", type=float, default=WINDOW_MS)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--n-coarsen-levels", type=int, default=N_COARSEN_LEVELS)
    ap.add_argument("--w-min", type=float, default=W_MIN)
    ap.add_argument("--sigma-v", type=float, default=None)
    ap.add_argument("--super-r-xy", type=float, default=SUPER_R_XY)
    ap.add_argument("--super-r-t-ms", type=float, default=SUPER_R_T_MS)
    ap.add_argument("--res-k", type=float, default=RES_K)
    ap.add_argument("--flow-smooth-iters", type=int, default=FLOW_SMOOTH_ITERS)
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--motion-resolver", default=MOTION_RESOLVER,
                    choices=["none", "lk", "affine", "vsa"])
    ap.add_argument("--compare-resolvers", action="store_true",
                    help="A/B all COMPARE_METHODS on the same window")
    ap.add_argument("--lk-min-support", type=int, default=LK_MIN_SUPPORT)
    ap.add_argument("--condition-min-eig", type=float, default=CONDITION_MIN_EIG)
    ap.add_argument("--d-vel", type=int, default=D_VEL)
    ap.add_argument("--vel-grid-n", type=int, default=VEL_GRID_N)
    ap.add_argument("--vsa-validate", type=lambda s: str(s).lower() not in
                    ("0", "false", "no"), default=VSA_VALIDATE)
    ap.add_argument("--band-sigma", type=float, default=BAND_SIGMA)
    ap.add_argument("--cleanup-topk", type=int, default=CLEANUP_TOPK)
    ap.add_argument("--cleanup-min-conf", type=float, default=CLEANUP_MIN_CONF)
    ap.add_argument("--w-node-motion", type=float, default=W_NODE_MOTION)
    ap.add_argument("--w-node-t", type=float, default=W_NODE_T)
    ap.add_argument("--w-node-x", type=float, default=W_NODE_X)
    ap.add_argument("--w-node-y", type=float, default=W_NODE_Y)
    ap.add_argument("--w-node-p", type=float, default=W_NODE_P)
    ap.add_argument(
        "--stream-segments", action="store_true",
        help="Tile the full recording into --segment-ms windows; run all resolvers",
    )
    ap.add_argument(
        "--segment-ms", type=float, default=SEGMENT_MS,
        help="Segment length when --stream-segments (default 60 ms)",
    )
    ap.add_argument(
        "--all-resolvers", action="store_true",
        help="Run none, lk, affine, vsa (default on with --stream-segments)",
    )
    ap.add_argument(
        "--diag-all-segments", action="store_true",
        help="Full diagnostic PNGs on every segment (default: only first segment)",
    )
    args = ap.parse_args()

    if args.stream_segments:
        args.all_resolvers = args.all_resolvers or True

    sigma_v = args.sigma_v if args.sigma_v is not None else SIGMA_V
    torch.manual_seed(SEED)
    t0 = _time.time()

    if args.stream_segments:
        methods = list(ALL_RESOLVERS) if args.all_resolvers else [args.motion_resolver]
        compare_resolvers = False
    elif args.compare_resolvers:
        methods = list(COMPARE_METHODS)
        compare_resolvers = True
    elif args.all_resolvers:
        methods = list(ALL_RESOLVERS)
        compare_resolvers = False
    else:
        methods = [args.motion_resolver]
        compare_resolvers = False

    print(
        f"[config] resolver(s)={methods}  stream={args.stream_segments}  "
        f"segment_ms={args.segment_ms}  window={args.window_ms}ms  "
        f"layers={args.num_layers}  out={args.out_dir}"
    )

    if args.flat:
        print(f"loading {args.input} (window={args.window_ms} ms) ...")
        t, x, y, p = load_events(args.input, args.window_ms)
        print("building spatial + temporal ellipsoid multigraph ...")
        (edge_spatial, edge_temporal,
         attr_spatial, attr_temporal,
         rec, src) = build_multigraph(t, x, y, p)
        comp = connected_components(len(t), rec, src)
        node_cb, cb_spatial, cb_temporal, w_spatial, w_temporal = make_codebooks(
            SENSOR, max((t.max() - t.min()), 1e-9), seed=SEED)
        print("[flat] FPE encode without residual motion ...")
        x_hv = fpe_encode(x, y, t, p, node_cb, w_motion=0.0)
        H_events = encode_nodes_multigraph(
            x_hv, edge_spatial, attr_spatial, edge_temporal, attr_temporal,
            cb_spatial, cb_temporal, w_spatial, w_temporal, args.num_layers)
        obj_id = assign(H_events, t, comp, tau=args.tau)
        save(t, x, y, p, obj_id, tau=args.tau)
        return

    all_compare_rows = []

    if args.stream_segments:
        _, span_ms = file_time_span_ms(args.input)
        ranges = segment_ranges(span_ms, args.segment_ms)
        print(f"streaming {len(ranges)} segments over {span_ms:.1f} ms "
              f"(segment_ms={args.segment_ms})")
        os.makedirs(args.out_dir, exist_ok=True)
        for seg_i, (t_a, t_b) in enumerate(ranges):
            seg_label = _segment_dir_name(t_a, t_b)
            full_diag = args.diag_all_segments or (seg_i == 0)
            print(f"\n======== segment {seg_i+1}/{len(ranges)} "
                  f"[{t_a:.0f}, {t_b:.0f}) ms  full_diag={full_diag} ========")
            t, x, y, p = load_events(
                args.input, window_ms=None,
                t_start_ms=t_a, t_end_ms=t_b,
            )
            print(f"  {len(t)} events  span={(t.max()-t.min())*1e3:.2f} ms")
            rows, _ = _run_window(
                args, t, x, y, p, methods, args.out_dir,
                full_diag=full_diag,
                compare_resolvers=False,
                seg_label=seg_label,
                save_labels=(seg_i == len(ranges) - 1 and len(methods) == 1),
            )
            all_compare_rows.extend(rows)
        csv_path = os.path.join(args.out_dir, "stream_segments.csv")
        if all_compare_rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_compare_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_compare_rows)
            print(f"wrote {csv_path}")
    else:
        print(f"loading {args.input} (window={args.window_ms} ms) ...")
        t, x, y, p = load_events(args.input, args.window_ms)
        print(f"  {len(t)} events  span={(t.max()-t.min())*1e3:.2f} ms")
        compare_rows, fig_paths_by_method = _run_window(
            args, t, x, y, p, methods, args.out_dir,
            full_diag=True,
            compare_resolvers=compare_resolvers,
            save_labels=not compare_resolvers,
        )
        if compare_resolvers:
            csv_path = os.path.join(args.out_dir, "compare_resolvers.csv")
            os.makedirs(args.out_dir, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
                writer.writeheader()
                writer.writerows(compare_rows)
            print(f"wrote {csv_path}")
            print("\n=== resolver comparison ===")
            hdr = f"{'method':8s} {'resol%':>7s} {'orient':>8s} {'imo%':>6s} " \
                  f"{'#obj':>4s} {'t_res':>6s} {'t_tot':>6s}"
            print(hdr)
            for r in compare_rows:
                print(f"{r['method']:8s} {100*r['resolved_frac']:6.1f}% "
                      f"{r['orientation_corr']:8.4f} {100*r['imo_frac']:5.1f}% "
                      f"{r['n_objects']:4d} {r['resolver_runtime_s']:6.2f} "
                      f"{r['total_runtime_s']:6.1f}")
            plot_resolver_comparison(fig_paths_by_method, args.out_dir)

    print(f"done in {_time.time()-t0:.1f}s  -> {args.out_dir}/")


if __name__ == "__main__":
    main()
