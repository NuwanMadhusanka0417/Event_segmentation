"""
segment.py - Streaming instance segmentation of moving objects from an
event-camera stream, built on the GVFA (Graph Vector Function Architecture)
encoder.

The scene is unlabeled. Grouping comes entirely from space-time-velocity
coherence encoded as hypervectors (VSA / HRR) and refined by a few hops of the
GVFA GraphCNN, then clustered online by cosine similarity inside each spatial
connected component.

NODE FEATURES (both graphs): {x, y, t} via FPE.  Polarity appears only in
temporal edge features (Eq. 6); it is still kept as a column in saved output.

PIPELINE
    load_events      -> read txt, take a time WINDOW (fast)
    build_multigraph -> spatial + temporal ellipsoid graphs (causal, past-only)
    fpe_encode       -> FPE hypervector per node from {x, y, t}
    encode_nodes     -> edge-conditioned GVFA per graph (no reservoir/Sigma-Pi);
                        bundle all hop-level vectors, then concat spatial|temporal
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

    Secondary FPE bandwidths (POS_BW, TIME_BW) control how fast the
    hypervector code de-correlates along each channel.

USAGE
    python segment.py                         # 30 ms window of events_filtered.txt
    python segment.py --window-ms 60          # bigger slice
    python segment.py --input other.txt --window-ms 100
Outputs events_labeled.parquet and seg.png in the working directory.
"""

import argparse
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from gvfa_encoder import encode_graph

# ----------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------
WINDOW_MS  = 60.0     # time slice to process (ms); None = whole file
SENSOR     = (346, 260)   # (W, H) in pixels

# Spatial graph: ellipsoid elongated in (x, y) — local spatial structure
SPATIAL_R_XY_FRAC = 0.04  # R_XY = this fraction of sensor width
SPATIAL_R_T_MS    = 5.0   # semi-minor axis along time (ms)
SPATIAL_MMAX      = 16    # max past spatial neighbours per node

# Temporal graph: ellipsoid elongated in t — motion over longer horizons
TEMPORAL_R_XY_FRAC = 0.01  # R_XY = this fraction of sensor width
TEMPORAL_R_T_MS    = 40.0  # semi-major axis along time (ms)
TEMPORAL_MMAX      = 12    # max past temporal neighbours per node

D          = 4000     # hypervector dimensionality per graph branch
NUM_LAYERS = 3        # GraphCNN layers incl. input (3 => 2 hops)
USE_RESERVOIR = False # no tap buffer / Sigma-Pi; hop vectors are bundled (summed)
TAU        = 0.15     # cosine threshold to join an existing object (lower -> fewer objects)
ALPHA      = 0.15     # prototype decayed-bundle update rate
MIN_EVENTS = 150      # objects below this size -> background id -1 (0 = keep all)

# FPE channel bandwidths for node features {x, y, t}
POS_BW  = 1.0    # x, y
TIME_BW = 0.5    # t
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


# ----------------------------------------------------------------------------
# 4. FRACTIONAL-POWER (HRR) ENCODING
# ----------------------------------------------------------------------------
def _fpe_channel(values, base_phase, scale, chunk=4096):
    """Fractional power encoding for one channel, HRR/FFT style (cf. bind()):
    h_v = real(ifft(base ** v)). The base is a unit-modulus phasor, so base**v
    is a clean rotation exp(i * phase * v) that preserves norm and varies
    smoothly with v. Computed in chunks to bound memory. Returns [N, D] float32.
    """
    v = (values * scale).astype(np.float32)
    out = np.empty((len(v), base_phase.shape[0]), dtype=np.float32)
    bp = torch.from_numpy(base_phase)                      # [D] float32
    for s in range(0, len(v), chunk):
        vb = torch.from_numpy(v[s:s + chunk])             # [b]
        ang = vb[:, None] * bp[None, :]                   # [b, D]
        code = torch.fft.ifft(torch.exp(1j * ang)).real   # [b, D]
        out[s:s + chunk] = code.numpy()
    return out


def fpe_encode(x, y, t, dim=D, sensor=SENSOR, seed=SEED):
    """FPE-encode node features {x, y, t} -> one L2-normalized hypervector [N, D]."""
    rng = np.random.default_rng(seed)
    W, H = sensor
    t_ms = (t - t[0]) * 1e3
    t_span = max(t_ms.max(), 1e-6)

    channels = [
        (x / W,          POS_BW),
        (y / H,          POS_BW),
        (t_ms / t_span,  TIME_BW),
    ]

    bundle = np.zeros((len(x), dim), dtype=np.float32)
    for vals, scale in channels:
        base_phase = (rng.uniform(0, 2 * np.pi, size=dim)).astype(np.float32)
        bundle += _fpe_channel(np.asarray(vals, dtype=np.float64), base_phase, float(scale))

    H_in = torch.from_numpy(bundle)
    H_in = torch.nn.functional.normalize(H_in, p=2, dim=1)
    return H_in   # [N, D] float32


# ----------------------------------------------------------------------------
# 5. GVFA ENCODER  (edge-conditioned, graphcnnVSA_Binding_FULL_new)
# ----------------------------------------------------------------------------
def encode_nodes(x_hv, edge_index, edge_attr, num_layers=NUM_LAYERS):
    """Edge-conditioned GVFA -> contextual node hypervectors H [N, D]."""
    return encode_graph(
        x_hv, edge_index, edge_attr,
        num_layers=num_layers,
        edge_feat_dim=edge_attr.shape[1],
        device=DEVICE,
        use_reservoir=USE_RESERVOIR,
        rng_seed=SEED,
    )


def encode_nodes_multigraph(x_hv, edge_spatial, attr_spatial,
                            edge_temporal, attr_temporal,
                            num_layers=NUM_LAYERS):
    """Apply GVFA separately on spatial and temporal graphs, then concat -> [N, 2*D]."""
    H_spatial = encode_nodes(x_hv, edge_spatial, attr_spatial, num_layers)
    H_temporal = encode_nodes(x_hv, edge_temporal, attr_temporal, num_layers)
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
    Hn = torch.nn.functional.normalize(H, p=2, dim=1).numpy().astype(np.float32)
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
    n_obj = int((ids >= 0).sum())
    n_bg = int(counts[ids == -1].sum()) if (ids == -1).any() else 0
    order = np.argsort(-counts)
    print(f"\n#objects found: {n_obj}  (+ {n_bg} background/noise events)  "
          f"over {len(obj_id)} events")
    for k in order:
        tag = "  <- background/noise" if ids[k] == -1 else ""
        print(f"  object {ids[k]:3d}: {counts[k]:6d} events{tag}")

    # scatter coloured by object id (background drawn first, in grey)
    plt.figure(figsize=(9, 7))
    bg = obj_id < 0
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
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="events_filtered.txt")
    ap.add_argument("--window-ms", type=float, default=WINDOW_MS)
    ap.add_argument("--tau", type=float, default=TAU,
                    help="cosine merge threshold (lower -> fewer objects)")
    args = ap.parse_args()

    torch.manual_seed(SEED)

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

    print("FPE-encoding nodes (x, y, t) ...")
    x_hv = fpe_encode(x, y, t)

    print(f"running edge-conditioned GVFA ({NUM_LAYERS} layers, "
          f"hop-bundle sum, no reservoir) on each graph ...")
    H = encode_nodes_multigraph(
        x_hv, edge_spatial, attr_spatial, edge_temporal, attr_temporal)
    print(f"  concatenated hypervectors: {H.shape[1]} dims ({D} spatial + {D} temporal)")

    print(f"streaming assignment (tau={args.tau}) ...")
    obj_id = assign(H, t, comp, tau=args.tau)

    save(t, x, y, p, obj_id, tau=args.tau)


if __name__ == "__main__":
    main()
