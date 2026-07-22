"""
segment.py - Streaming instance segmentation of moving objects from an
event-camera stream, built on the GVFA (Graph Vector Function Architecture)
encoder.

The scene is unlabeled. Grouping comes entirely from space-time-velocity
coherence encoded as hypervectors (VSA / HRR) and refined by a few hops of the
GVFA GraphCNN, then clustered online by cosine similarity inside each spatial
connected component.

NODE FEATURES (both graphs): absolute {x, y, t, p} plus residual motion
{|r|, angle(r)} via FPE codebooks (motion-dominant bundle). Relative
position/motion still appears in edge Δ terms (Eq. 5 / Eq. 6).

PIPELINE
    load_events / build_multigraph
    Stage 1: node_flow -> smooth_flow
    Stage 2: fit_ego_motion -> residual_split -> dilate/erode -> fit_object_models
    fpe_encode(x,y,t,p,rx,ry) -> GVFA
    Stage 3: motion_coarsen(IMO, residual, model_id gate) -> assign -> unpool
    smooth_labels -> drop_tiny_clusters -> diagnostics

USAGE
    python segment.py --window-ms 60 --num-layers 3
    python segment.py --input events_filtered.txt --tau 0.12 --num-layers 4
    python segment.py --flat   # old whole-scene path (no residual / models)
Outputs events_labeled.parquet and diag/*.png in the working directory.
"""

import argparse
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from fpe_codebook import FPECodebook, bundle_weighted, bind_hv, PHASE_INT_KMAX
from gvfa_encoder import encode_graph
from ego_motion import (
    fit_ego_motion,
    fit_ego_motion_ransac,
    residual_split,
    refine_imo_mask,
    fit_object_models,
)
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
    plot_ego_inliers,
    plot_flow_raw,
    plot_flow_smoothed,
    plot_motion_kernels,
    plot_motion_models,
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

# === STAGE 2: ego-motion + residual split ===
RES_K      = 2.0           # residual threshold in robust sigmas
IRLS_ITERS = 10
DILATE_ITERS = 2           # recover motion-parallel contours (aperture)
DILATE_FRAC  = 0.5         # neighbour IMO fraction to flip BG->IMO / erode

# RANSAC ego (largest consensus = background; avoids person contamination)
EGO_RANSAC        = True
RANSAC_HYPOTHESES = 200
RANSAC_SAMPLE     = 8
RANSAC_INLIER_K   = 2.5

# === STAGE 2b: multi-model fitting on IMO ===
OBJECT_MODEL_KIND = "affine"   # "similarity" | "affine"
MAX_MODELS        = 4
MIN_MODEL_INLIERS = 600
MODEL_RES_K       = 2.5
MERGE_COS         = 0.9
MERGE_SPEED_RATIO = 0.5

# === STAGE 3: motion-coherent pooling on IMO residuals ===
SIGMA_V          = None   # None => auto (median edge ||v_i-v_j||)
W_MIN            = 0.1    # refuse Graclus merges below this motion affinity
N_COARSEN_LEVELS = 7      # coarsening depth on IMO subgraph
SUPER_R_XY       = 40.0   # (kept for --flat / legacy supernode path)
SUPER_R_T_MS     = 30.0
MIN_SUPER_SIZE   = 3      # mark supernodes with fewer members as noise
LAM              = 3.0    # graph label-smoothing strength
SMOOTH_ITERS     = 5
USE_GRAPH_CUT    = False  # optional alpha-expansion (needs PyMaxflow)
MIN_CLUSTER_SIZE = 400    # final objects smaller than this -> background
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
BW_SPEED   = 0.1                 # residual speed codebook
SPEED_V0   = 50.0                # px/s signed-log knee for |r|
N_ANGLE_BINS = 720               # 0.5 deg resolution for direction

# Node bundle weights — motion dominant; polarity off (contrast ≠ identity)
W_NODE_X, W_NODE_Y = 0.3, 0.3
W_NODE_T           = 0.5
W_NODE_P           = 0.0
W_NODE_MOTION      = 2.0      # ~3x position: residual motion is the cue

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
def load_events(path, window_ms=WINDOW_MS):
    """Read 'timestamp x y polarity' rows; keep the first window_ms milliseconds.
    Returns t (seconds, float), x, y (int), p (0/1), all sorted by time."""
    data = np.loadtxt(path)
    t, x, y, p = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    order = np.argsort(t, kind="stable")
    t, x, y, p = t[order], x[order], y[order], p[order]
    if window_ms is not None:
        keep = t <= (t[0] + window_ms * 1e-3)
        t, x, y, p = t[keep], x[keep], y[keep], p[keep]
    return (t.astype(np.float64),
            x.astype(np.float64), y.astype(np.float64),
            p.astype(np.float64))


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
        # residual motion: speed (|r|) + periodic direction (angle)
        "speed": FPECodebook(
            "speed", D, BW_SPEED, "signed_log_radix", radix_S=16,
            vmin=0, vmax=60, value_grid_step=0.1,
            signed_log_v0=SPEED_V0, phase_dist="gaussian", seed=seed + 20,
        ),
        "dir": FPECodebook(
            "dir", D, 1.0, "periodic", n_angle_bins=N_ANGLE_BINS,
            phase_dist="integer", phase_int_kmax=PHASE_INT_KMAX,
            seed=seed + 21,
        ),
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


def fpe_encode(x, y, t, p, codebooks, *, rx=None, ry=None, motion_valid=None):
    """FPE node features {x,y,t,p} plus optional residual motion (|r|, angle).

    Keyword-only rx, ry keep existing callers working. When residual is provided,
    speed and direction are BOUND (conjunctive) then bundled with motion-dominant
    weight. Events with undefined residual get a zero motion sub-symbol.
    """
    t_us = (t - t[0]) * 1e6
    terms = [
        (codebooks["x"].encode(x), W_NODE_X),
        (codebooks["y"].encode(y), W_NODE_Y),
        (codebooks["t"].encode(t_us, interpolate=True), W_NODE_T),
        (codebooks["p"].encode(p), W_NODE_P),
    ]

    if rx is not None and ry is not None and "speed" in codebooks and "dir" in codebooks:
        rx = np.asarray(rx, dtype=np.float64).ravel()
        ry = np.asarray(ry, dtype=np.float64).ravel()
        n = len(rx)
        if motion_valid is None:
            motion_valid = np.isfinite(rx) & np.isfinite(ry)
        else:
            motion_valid = np.asarray(motion_valid, dtype=bool).ravel()
        n_undef = int((~motion_valid).sum())
        print(f"[fpe] residual motion channel: {int(motion_valid.sum())}/{n} valid; "
              f"{n_undef} events get zero z_motion")

        speed = np.hypot(rx, ry)
        ang = np.arctan2(ry, rx)
        z_speed = codebooks["speed"].encode(speed, interpolate=True)
        z_dir = codebooks["dir"].encode(ang, interpolate=True)
        z_motion = bind_hv(z_speed, z_dir)
        if n_undef:
            z_motion = z_motion.clone()
            bad = torch.from_numpy(~motion_valid)
            z_motion[bad] = 0.0
        terms.append((z_motion, W_NODE_MOTION))

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
def main():
    import time as _time

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="events_filtered.txt")
    ap.add_argument("--window-ms", type=float, default=WINDOW_MS)
    ap.add_argument("--tau", type=float, default=TAU,
                    help="cosine merge threshold (lower -> fewer objects)")
    ap.add_argument("--num-layers", type=int, default=NUM_LAYERS,
                    help="GVFA layers incl. input (3 => 2 hops)")
    ap.add_argument("--flat", action="store_true",
                    help="bypass ego-motion + pooling; old per-event clustering")
    ap.add_argument("--n-coarsen-levels", type=int, default=N_COARSEN_LEVELS)
    ap.add_argument("--w-min", type=float, default=W_MIN,
                    help="refuse Graclus merges below this motion affinity")
    ap.add_argument("--sigma-v", type=float, default=None,
                    help="velocity length-scale; default=auto median edge ||dv||")
    ap.add_argument("--super-r-xy", type=float, default=SUPER_R_XY)
    ap.add_argument("--super-r-t-ms", type=float, default=SUPER_R_T_MS)
    ap.add_argument("--res-k", type=float, default=RES_K,
                    help="residual threshold in robust sigmas")
    ap.add_argument("--flow-smooth-iters", type=int, default=FLOW_SMOOTH_ITERS)
    ap.add_argument("--lam", type=float, default=LAM,
                    help="graph label-smoothing strength")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="diagnostic figure output directory")
    ap.add_argument("--ego-ransac", type=lambda s: str(s).lower() not in
                    ("0", "false", "no", "off"), default=EGO_RANSAC,
                    help="use RANSAC consensus for global ego fit")
    ap.add_argument("--ransac-hypotheses", type=int, default=RANSAC_HYPOTHESES)
    ap.add_argument("--ransac-sample", type=int, default=RANSAC_SAMPLE)
    ap.add_argument("--ransac-inlier-k", type=float, default=RANSAC_INLIER_K)
    ap.add_argument("--object-model-kind", choices=("affine", "similarity"),
                    default=OBJECT_MODEL_KIND)
    ap.add_argument("--max-models", type=int, default=MAX_MODELS)
    ap.add_argument("--min-model-inliers", type=int, default=MIN_MODEL_INLIERS)
    ap.add_argument("--model-res-k", type=float, default=MODEL_RES_K)
    ap.add_argument("--merge-cos", type=float, default=MERGE_COS)
    ap.add_argument("--merge-speed-ratio", type=float, default=MERGE_SPEED_RATIO)
    ap.add_argument("--use-graph-cut", type=lambda s: str(s).lower() not in
                    ("0", "false", "no", "off"), default=USE_GRAPH_CUT)
    ap.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
    args = ap.parse_args()

    sigma_v = args.sigma_v if args.sigma_v is not None else SIGMA_V
    torch.manual_seed(SEED)
    t0 = _time.time()

    print(
        f"[config] window={args.window_ms}ms layers={args.num_layers} "
        f"tau={args.tau} res_k={args.res_k} ego_ransac={args.ego_ransac} "
        f"obj={args.object_model_kind} max_models={args.max_models} "
        f"min_inl={args.min_model_inliers} merge_cos={args.merge_cos} "
        f"lam={args.lam} graph_cut={args.use_graph_cut} "
        f"min_cluster={args.min_cluster_size} out={args.out_dir} flat={args.flat}"
    )

    print(f"loading {args.input} (window={args.window_ms} ms) ...")
    t, x, y, p = load_events(args.input, args.window_ms)
    print(f"  {len(t)} events  x:[{x.min():.0f},{x.max():.0f}]  "
          f"y:[{y.min():.0f},{y.max():.0f}]  span={ (t.max()-t.min())*1e3:.2f} ms")

    print("building spatial + temporal ellipsoid multigraph ...")
    (edge_spatial, edge_temporal,
     attr_spatial, attr_temporal,
     rec, src) = build_multigraph(t, x, y, p)
    print(f"  spatial:  {edge_spatial.shape[1]} edges  "
          f"(attr dim=3: dx,dy,dt)")
    print(f"  temporal: {edge_temporal.shape[1]} edges  "
          f"(attr dim=6: dx,dy,dt,dx/dt,dy/dt,dp)")

    comp = connected_components(len(t), rec, src)
    print(f"  {comp.max()+1} spatial connected components")

    t_span = max((t.max() - t.min()), 1e-9)
    node_cb, cb_spatial, cb_temporal, w_spatial, w_temporal = make_codebooks(
        SENSOR, t_span, seed=SEED)

    if args.flat:
        # old whole-scene path: encode without residual, skip Stages 1–3
        print("[flat] FPE codebook-encoding nodes (x, y, t, p) — no residual")
        x_hv = fpe_encode(x, y, t, p, node_cb)
        print(f"running FPE-edge GVFA ({args.num_layers} layers) ...")
        H_events = encode_nodes_multigraph(
            x_hv, edge_spatial, attr_spatial, edge_temporal, attr_temporal,
            cb_spatial, cb_temporal, w_spatial, w_temporal, args.num_layers)
        print(f"  concatenated hypervectors: {H_events.shape[1]} dims")
        print("[flat] bypassing ego-motion + pooling — per-event clustering")
        print(f"streaming assignment (tau={args.tau}) ...")
        obj_id = assign(H_events, t, comp, tau=args.tau)
        save(t, x, y, p, obj_id, tau=args.tau)
        return

    # ==================================================================
    # STAGE 1 — flow regularization
    # ==================================================================
    print("STAGE 1: estimating + regularizing normal flow ...")
    vx, vy = node_flow(t, x, y, edge_temporal)
    diagnose_node_flow(vx, vy)
    p1 = plot_flow_raw(x, y, vx, vy, args.out_dir)

    vx_s, vy_s, n_vb, n_va = smooth_flow(
        vx, vy, edge_spatial,
        n_iters=args.flow_smooth_iters, keep=FLOW_KEEP)
    p2 = plot_flow_smoothed(
        x, y, vx_s, vy_s, args.out_dir,
        n_iters=args.flow_smooth_iters, keep=FLOW_KEEP,
        n_valid_before=n_vb, n_valid_after=n_va)

    # ==================================================================
    # STAGE 2 — ego-motion fit + residual split + dilate/erode + models
    # ==================================================================
    print("STAGE 2: ego-motion fit + residual split ...")
    valid_flow = np.hypot(vx_s, vy_s) > 1e-12
    if args.ego_ransac:
        # RANSAC recovers background as LARGEST consensus motion
        params, residual, ego_info = fit_ego_motion_ransac(
            x, y, vx_s, vy_s, SENSOR,
            n_hypotheses=args.ransac_hypotheses,
            sample_size=args.ransac_sample,
            inlier_k=args.ransac_inlier_k,
            n_iters_polish=IRLS_ITERS,
            valid_mask=valid_flow,
            seed=SEED,
        )
    else:
        params, residual, ego_info = fit_ego_motion(
            x, y, vx_s, vy_s, SENSOR, n_iters=IRLS_ITERS, valid_mask=valid_flow)
        ego_info = dict(ego_info)
        ego_info.setdefault("ransac", False)
        ego_info.setdefault("n_ransac_inliers", 0)
        ego_info.setdefault("n_hypotheses", 0)
        ego_info.setdefault("ransac_inlier_mask", valid_flow)

    is_imo, residual, thresh = residual_split(
        residual, edge_spatial, res_k=args.res_k, valid_mask=valid_flow)

    print("STAGE 2b: dilate/erode IMO mask (aperture recovery) ...")
    is_imo, dilate_info = refine_imo_mask(
        is_imo, [edge_spatial], n_dilate=DILATE_ITERS, frac=DILATE_FRAC)

    rx, ry = residual[:, 0], residual[:, 1]
    print(f"STAGE 2c: multi-model fitting on IMO ({args.object_model_kind}) ...")
    model_id, models, merge_info = fit_object_models(
        x, y, rx, ry, is_imo, SENSOR,
        max_models=args.max_models,
        min_inliers=args.min_model_inliers,
        res_k=args.model_res_k,
        n_iters=IRLS_ITERS,
        model_kind=args.object_model_kind,
        merge_cos=args.merge_cos,
        merge_speed_ratio=args.merge_speed_ratio,
    )

    p3 = plot_ego_fit(
        x, y, params, residual, SENSOR, args.out_dir,
        res_k=args.res_k, thresh=thresh,
        inlier_rms=ego_info["inlier_rms"], info=ego_info)
    p10 = plot_ego_inliers(
        x, y, params, ego_info, SENSOR, args.out_dir)
    p4 = plot_residual_split(
        x, y, is_imo, residual, thresh, args.out_dir,
        res_k=args.res_k, dilate_info=dilate_info)
    p9 = plot_motion_models(
        x, y, is_imo, model_id, models, args.out_dir,
        max_models=args.max_models, model_res_k=args.model_res_k,
        min_model_inliers=args.min_model_inliers,
        merge_info=merge_info, model_kind=args.object_model_kind)

    # ==================================================================
    # Encode AFTER residual (motion enters node hypervector)
    # ==================================================================
    print("FPE codebook-encoding nodes (x, y, t, p, |r|, angle(r)) ...")
    x_hv = fpe_encode(
        x, y, t, p, node_cb, rx=rx, ry=ry, motion_valid=valid_flow)
    p8 = plot_motion_kernels(
        node_cb["dir"], node_cb["speed"], args.out_dir,
        bw_speed=BW_SPEED, speed_v0=SPEED_V0,
        n_angle_bins=N_ANGLE_BINS, phase_int_kmax=PHASE_INT_KMAX)

    print(f"running FPE-edge GVFA ({args.num_layers} layers, "
          f"hop-bundle sum, L2 norm) on each graph ...")
    H_events = encode_nodes_multigraph(
        x_hv, edge_spatial, attr_spatial, edge_temporal, attr_temporal,
        cb_spatial, cb_temporal, w_spatial, w_temporal, args.num_layers)
    print(f"  concatenated hypervectors: {H_events.shape[1]} dims "
          f"({D} spatial + {D} temporal)")

    # ==================================================================
    # STAGE 3 — VSA pooling on IMO residuals only
    # ==================================================================
    print("STAGE 3: residual-affinity coarsening on IMO subgraph ...")
    labels = np.zeros(len(t), dtype=np.int64)  # 0 = background
    cluster_full = np.full(len(t), -1, dtype=np.int64)
    cinfo = {
        "sigma_v": float("nan"), "w_min": args.w_min,
        "n_levels": args.n_coarsen_levels, "C": 0,
        "sizes": np.array([0]), "n_rejected_total": 0,
    }

    n_imo = int(is_imo.sum())
    if n_imo == 0:
        print("[stage3] no IMO candidates — all background")
    else:
        edge_s_imo, sub_idx, _ = induce_subgraph(edge_spatial, is_imo)
        edge_t_imo, _, _ = induce_subgraph(edge_temporal, is_imo)
        rx_imo, ry_imo = rx[sub_idx], ry[sub_idx]
        model_imo = model_id[sub_idx]
        t_imo = t[sub_idx]
        H_imo = H_events[sub_idx]

        print(f"[stage3] inducing IMO subgraph: {n_imo} events; "
              f"coarsening with RESIDUAL (rx, ry) + model_id gate")
        cluster_id, cinfo = motion_coarsen(
            n_imo, edge_s_imo, edge_t_imo, rx_imo, ry_imo,
            n_levels=args.n_coarsen_levels,
            sigma_v=sigma_v,
            w_min=args.w_min,
            seed=SEED,
            model_id=model_imo,
            velocity_name="residual",
        )
        cinfo["w_min"] = args.w_min
        cinfo["n_levels"] = args.n_coarsen_levels
        cluster_full[sub_idx] = cluster_id

        print("bundling IMO hypervectors into supernodes ...")
        H_s_bundle = bundle_hypervectors(H_imo[:, :D], cluster_id)
        H_t_bundle = bundle_hypervectors(H_imo[:, D:], cluster_id)
        H_super = torch.cat([H_s_bundle, H_t_bundle], dim=1)
        diagnose_bundling_cosine(H_super, seed=SEED, label="bundling")

        # components on supernodes via projected IMO spatial edges
        C = H_super.shape[0]
        if edge_s_imo.numel() > 0:
            rec_s = cluster_id[edge_s_imo[0].numpy()]
            src_s = cluster_id[edge_s_imo[1].numpy()]
            comp_super = connected_components(C, rec_s, src_s)
        else:
            comp_super = np.zeros(C, dtype=np.int64)

        # supernode times = mean member time
        agg = supernode_aggregates(
            x[sub_idx], y[sub_idx], t_imo, p[sub_idx],
            rx_imo, ry_imo, cluster_id)
        print(f"streaming assignment on IMO supernodes (tau={args.tau}) ...")
        labels_super = assign(
            H_super, agg["t"], comp_super, tau=args.tau, min_events=0)
        # mark tiny supernodes as noise (-1), then shift to 1..K (0=background)
        small = agg["counts"] < MIN_SUPER_SIZE
        if small.any():
            labels_super = labels_super.copy()
            labels_super[small] = -1
            print(f"  marked {int(small.sum())} tiny supernodes as noise")
        diagnose_clustering(H_super, labels_super)

        # map: -1 -> 0 (noise/bg), 0..K-1 -> 1..K
        labels_imo_events = unpool(labels_super, cluster_id)
        out = np.zeros(n_imo, dtype=np.int64)
        keep = labels_imo_events >= 0
        if keep.any():
            _, compact = np.unique(labels_imo_events[keep], return_inverse=True)
            out[keep] = compact + 1  # start at 1
        labels[sub_idx] = out
        assert np.all(labels[sub_idx] == out)
        print("[unpool] IMO events labelled; background stays 0")

    p5 = plot_supernodes(x, y, cluster_full[is_imo] if n_imo else np.array([]),
                         is_imo, args.out_dir, cinfo)

    # graph-smoothing cleanup on full event graph
    print(f"graph label smoothing (lam={args.lam}, iters={SMOOTH_ITERS}, "
          f"graph_cut={args.use_graph_cut}) ...")
    protos = compute_prototypes(H_events, labels)
    labels = smooth_labels(
        labels, H_events, protos,
        edge_index_list=[edge_spatial, edge_temporal],
        lam=args.lam, n_iters=SMOOTH_ITERS,
        use_graph_cut=args.use_graph_cut,
    )
    labels = drop_tiny_clusters(labels, args.min_cluster_size, background=0)
    assert len(labels) == len(t)
    assert not np.isnan(labels.astype(np.float64)).any()
    assert np.all(np.isfinite(labels))

    runtime = _time.time() - t0
    ids, counts = np.unique(labels, return_counts=True)
    object_counts = [(int(i), int(c)) for i, c in zip(ids, counts) if i > 0]
    p6 = plot_segmentation(
        x, y, labels, args.out_dir,
        tau=args.tau, num_layers=args.num_layers,
        lam=args.lam, smooth_iters=SMOOTH_ITERS, runtime_s=runtime,
        w_node_motion=W_NODE_MOTION, n_models=len(models),
        object_counts=object_counts)
    # 3x3 summary: include ego-inliers (10) and motion models (09)
    plot_summary([p1, p2, p3, p4, p5, p6, p8, p9, p10], args.out_dir)

    save(t, x, y, p, labels, tau=args.tau)
    n_obj = int(len(np.unique(labels[labels > 0]))) if (labels > 0).any() else 0
    print(f"done in {runtime:.1f}s  -> diagnostics in {args.out_dir}/")
    print(f"[summary] models={len(models)}  objects={n_obj}  "
          f"bg={int((labels == 0).sum())}")


if __name__ == "__main__":
    main()
