"""Per-region motion by contrast maximisation (CMax) over a long event window. Training-free.

Why (2026-10-21): the flow cues (VSA 35.9, VecKM 36.1 event IoU on the test sequences) are
limited by STRUCTURED errors, not noise -- random flow error of the same size (0.4 px) still
allows 65%. Both flows pool a neighbourhood, so they blur motion across object edges and
under-read slow motion (0.76-0.86 x truth), and a region-level graph cut on top of them does
not help (29.1 vs 28.9 on validation). Here the motion of a region is measured from the
events themselves, along their trajectories over a longer window, the way EMSGC (Zhou et al.
2021) and contrast maximisation (Gallego et al. 2018) do:

  events     window [ts - T, ts]; undistorted; camera rotation removed per event with the
             IMU rate (exact rotation to the label time ts) -> positions at ts, in
             undistorted sensor px
  regions    spatial k-means of the SCORING events (the last score_ms) -> masks at ts
  motion     per region, the 2-D translation v (undistorted sensor px per 12.5 ms, forward
             in time) that makes the image of warped events (IWE)  x + v * (ts - t) / 12.5 ms
             sharpest inside its mask; sharpness = sum I^2 / sum I (independent of how many
             events land in the mask); coarse-to-fine grid search
  background v = 0 after derotation (bg "zero"), or the median region motion (bg "median")
  outputs    per scoring event: speed |v_region - v_bg|; gain = S(v_region) / S(v_bg) - 1
             (how much sharper the region gets); own = per-event two-model check: the IWE
             density at the event's own warped position under v_region vs under v_bg, as
             own = d_region / (d_region + d_bg) -- background events inside a moving region
             stay sharp only under v_bg, so they get own < 0.5

Units match the rest of the project: px per 12.5 ms interval (hdems.veckm.INTERVAL_S), and
the 0.25 px decision scale of scripts/check_protocol_ceiling.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTERVAL_S = 0.0125


@dataclass(frozen=True)
class CMaxParams:
    window_ms: float = 50.0          # events used for the motion of a region
    score_ms: float = 12.5           # events that get a decision (the benchmark window)
    n_regions: int = 30              # k-means regions (Hua et al. over-segment into ~30)
    min_events: int = 40             # scoring events per region, fewer -> merged away
    mask_dilate_px: int = 2          # region mask = its scoring events' pixels, dilated
    vmax: float = 6.0                # search range |v_x|, |v_y| (px per 12.5 ms)
    steps: tuple[float, ...] = (1.0, 0.25, 0.0625)   # grid step per level
    sigmas: tuple[float, ...] = (2.0, 1.0, 0.5)      # IWE blur (px) per level (0.5: sub-pixel)
    max_region_events: int = 30000   # random subsample of a region's window events
    bg: str = "zero"                 # zero | median
    seed: int = 0


# ------------------------------------------------------------------- geometry
def derotate(x: np.ndarray, y: np.ndarray, t: np.ndarray, t_ref: float, omega: np.ndarray | None,
             mm: dict) -> tuple[np.ndarray, np.ndarray]:
    """Event pixels at times t -> undistorted pixel positions (fx, fy, cx, cy of mm) where a
    STATIC point would appear at t_ref, given the camera rate omega (rad/s, camera frame;
    None = no rotation removed). Convention of hdems.imu.rotation_flow: the bearing at the
    earlier time t is R(omega (t_ref - t)) times the bearing at t_ref."""
    from hdems.veckm import undistort_radtan
    fx, fy, cx, cy = (float(mm[k]) for k in ("fx", "fy", "cx", "cy"))
    xn, yn = undistort_radtan(np.asarray(x, np.float64), np.asarray(y, np.float64), mm)
    if omega is None or not np.any(omega):
        return fx * xn + cx, fy * yn + cy
    X = np.stack([xn, yn, np.ones_like(xn)], 1)                    # (n, 3) bearings at t
    rv = -np.outer(t_ref - np.asarray(t, np.float64), np.asarray(omega, np.float64))   # back to t_ref
    th = np.linalg.norm(rv, axis=1, keepdims=True)
    k = rv / np.maximum(th, 1e-12)
    c, s = np.cos(th), np.sin(th)
    Xr = X * c + np.cross(k, X) * s + k * (k * X).sum(1, keepdims=True) * (1 - c)
    z = np.clip(Xr[:, 2], 1e-6, None)
    return fx * Xr[:, 0] / z + cx, fy * Xr[:, 1] / z + cy


# ------------------------------------------------------------------- sharpness
def _iwe(px: np.ndarray, py: np.ndarray, h: int, w: int, sigma: float) -> np.ndarray:
    """(C, n) warped positions (box coordinates) -> (C, h, w) bilinear-voted, blurred IWEs."""
    from scipy.ndimage import gaussian_filter
    C = px.shape[0]
    ix, iy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    ax, ay = px - ix, py - iy
    base = (np.arange(C) * h * w)[:, None]
    idx, wts = [], []
    for dx, dy, wt in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)),
                       (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
        jx, jy = ix + dx, iy + dy
        ok = (jx >= 0) & (jx < w) & (jy >= 0) & (jy < h)
        idx.append((base + jy * w + jx)[ok])
        wts.append(wt[ok])
    I = np.bincount(np.concatenate(idx), np.concatenate(wts), C * h * w).reshape(C, h, w)
    return gaussian_filter(I, (0, sigma, sigma)) if sigma > 0 else I


def sharpness(xr: np.ndarray, yr: np.ndarray, dtau: np.ndarray, V: np.ndarray, box: tuple[int, int, int, int],
              mask: np.ndarray, sigma: float, chunk: int = 48) -> np.ndarray:
    """S(v) = sum I^2 / sum I over the mask, for each candidate velocity V[c] (C, 2).
    dtau = (t_ref - t) / 12.5 ms per event; box = (x0, y0, w, h) of the mask."""
    x0, y0, w, h = box
    out = np.zeros(len(V))
    for a in range(0, len(V), chunk):
        v = V[a:a + chunk]
        I = _iwe(xr[None] - x0 + v[:, :1] * dtau[None], yr[None] - y0 + v[:, 1:] * dtau[None], h, w, sigma)
        m = I[:, mask]
        out[a:a + chunk] = (m * m).sum(1) / np.maximum(m.sum(1), 1e-9)
    return out


def best_velocity(xr, yr, dtau, box, mask, p: CMaxParams, center=(0.0, 0.0), start_level: int = 0,
                  half: float | None = None) -> tuple[np.ndarray, float]:
    """Coarse-to-fine grid search -> (v (2,), S(v) at the finest blur). A refit starts at a
    finer level around the current motion (``center``, +-``half``)."""
    c = np.asarray(center, np.float64)
    half = p.vmax if half is None else half
    s_best = 0.0
    for step, sig in list(zip(p.steps, p.sigmas))[start_level:]:
        g = np.arange(-half, half + 1e-9, step)
        V = np.stack(np.meshgrid(g + c[0], g + c[1]), -1).reshape(-1, 2)
        S = sharpness(xr, yr, dtau, V, box, mask, sig)
        i = int(np.argmax(S))
        c, s_best = V[i], float(S[i])
        half = step                  # the optimum lies within one step of the best grid point
    return c, s_best


# ------------------------------------------------------------------- regions
def _regions(xs: np.ndarray, ys: np.ndarray, p: CMaxParams) -> np.ndarray:
    """Spatial k-means of the scoring events (on their unique pixels) -> label per event."""
    from scipy.cluster.vq import kmeans2
    n = len(xs)
    k = int(min(p.n_regions, max(1, n // p.min_events)))
    if k <= 1:
        return np.zeros(n, np.int64)
    pix = np.round(xs).astype(np.int64) * 100003 + np.round(ys).astype(np.int64)
    upix, inv = np.unique(pix, return_inverse=True)
    pts = np.stack([upix // 100003, upix % 100003], 1).astype(np.float64)
    k = min(k, len(pts))
    _c, lab = kmeans2(pts, k, minit="++", seed=p.seed)
    lab = lab[inv]
    # tiny clusters -> nearest big cluster centre
    cnt = np.bincount(lab, minlength=k)
    big = np.nonzero(cnt >= p.min_events)[0]
    if len(big) == 0:
        return np.zeros(n, np.int64)
    cen = np.stack([[xs[lab == b].mean(), ys[lab == b].mean()] for b in big])
    small = ~np.isin(lab, big)
    if small.any():
        d = ((np.stack([xs[small], ys[small]], 1)[:, None] - cen[None]) ** 2).sum(-1)
        lab[small] = big[np.argmin(d, 1)]
    return np.searchsorted(big, lab)


def _mask(xs, ys, p: CMaxParams, pad: float, H: int, W: int):
    """Region mask from its scoring events -> (box (x0, y0, w, h), mask (h, w) bool)."""
    from scipy.ndimage import binary_dilation
    r = p.mask_dilate_px
    x0 = int(np.floor(xs.min() - r - pad))
    y0 = int(np.floor(ys.min() - r - pad))
    x1 = int(np.ceil(xs.max() + r + pad)) + 2
    y1 = int(np.ceil(ys.max() + r + pad)) + 2
    w, h = x1 - x0, y1 - y0
    m = np.zeros((h, w), bool)
    m[np.clip(np.round(ys).astype(int) - y0, 0, h - 1), np.clip(np.round(xs).astype(int) - x0, 0, w - 1)] = True
    if r > 0:
        m = binary_dilation(m, iterations=r)
    return (x0, y0, w, h), m


def _density_at(xr, yr, dtau, v, box, sigma, qx, qy, qdt):
    """IWE of the region's window events under v, read at the query events' warped positions."""
    from scipy.ndimage import map_coordinates
    x0, y0, w, h = box
    I = _iwe((xr + v[0] * dtau - x0)[None], (yr + v[1] * dtau - y0)[None], h, w, sigma)[0]
    I = I / max(I.sum(), 1e-9)
    return map_coordinates(I, [qy + v[1] * qdt - y0, qx + v[0] * qdt - x0], order=1, mode="constant")


def segment_window(xr: np.ndarray, yr: np.ndarray, t: np.ndarray, t_ref: float, p: CMaxParams = CMaxParams(),
                   labels: np.ndarray | None = None) -> dict:
    """Derotated, undistorted event positions (xr, yr) at times t (window ending at t_ref) ->
    per SCORING event (t >= t_ref - score_ms): {"score_idx" indices into the inputs, "region",
    "speed", "gain", "own"}, plus per region {"v", "v_bg", "n"}. ``labels`` (one region id
    0..L-1 per scoring event) replaces the k-means regions (diagnostics: pure GT regions)."""
    t = np.asarray(t, np.float64)
    dtau = (t_ref - t) / INTERVAL_S
    win = t >= t_ref - p.window_ms / 1000.0
    sc = np.nonzero(t >= t_ref - p.score_ms / 1000.0)[0]
    out = {"score_idx": sc, "region": np.zeros(len(sc), np.int64), "speed": np.zeros(len(sc)),
           "gain": np.zeros(len(sc)), "own": np.full(len(sc), 0.5), "v": np.zeros((0, 2)), "v_bg": np.zeros(2)}
    if len(sc) < p.min_events:
        return out
    rng = np.random.default_rng(p.seed)
    xs, ys = xr[sc], yr[sc]
    lab = _regions(xs, ys, p) if labels is None else np.asarray(labels, np.int64)
    L = int(lab.max()) + 1
    wi = np.nonzero(win)[0]
    pad = p.vmax * float(dtau[wi].max())
    vs, ss, boxes, sel = np.zeros((L, 2)), np.zeros(L), [], []
    for l in range(L):
        box, m = _mask(xs[lab == l], ys[lab == l], p, pad, 0, 0)
        x0, y0, w, h = box
        e = wi[(xr[wi] >= x0) & (xr[wi] < x0 + w) & (yr[wi] >= y0) & (yr[wi] < y0 + h)]
        if len(e) > p.max_region_events:
            e = rng.choice(e, p.max_region_events, replace=False)
        vs[l], ss[l] = best_velocity(xr[e], yr[e], dtau[e], box, m, p)
        boxes.append((box, m))
        sel.append(e)
    v_bg = np.median(vs, 0) if p.bg == "median" else np.zeros(2)
    gain = np.zeros(L)
    for l in range(L):
        (box, m), e = boxes[l], sel[l]
        s_bg = sharpness(xr[e], yr[e], dtau[e], v_bg[None], box, m, p.sigmas[-1])[0]
        gain[l] = ss[l] / max(s_bg, 1e-9) - 1.0
        q = lab == l
        qi = sc[q]
        d_obj = _density_at(xr[e], yr[e], dtau[e], vs[l], box, p.sigmas[-1], xr[qi], yr[qi], dtau[qi])
        d_bg = _density_at(xr[e], yr[e], dtau[e], v_bg, box, p.sigmas[-1], xr[qi], yr[qi], dtau[qi])
        out["own"][q] = d_obj / np.maximum(d_obj + d_bg, 1e-12)
    out["region"] = lab
    # a motion beyond the search range = no real optimum (texture-poor region pulled onto
    # something else): unreliable, scored as static
    edge = np.abs(vs).max(1) > p.vmax
    speed = np.where(edge, 0.0, np.linalg.norm(vs - v_bg, axis=1))
    gain = np.where(edge, 0.0, gain)
    out["speed"] = speed[lab]
    out["gain"] = gain[lab]
    out.update(v=vs, v_bg=v_bg, n=np.bincount(lab, minlength=L), edge=edge)
    return out


# ------------------------------------------------------- iterative reassignment
@dataclass(frozen=True)
class AssignParams:
    iters: int = 3                   # reassign + refit rounds
    lam: float = 1.0                 # Potts smoothness between neighbouring event pixels
    assign_sigma: float = 1.0        # IWE blur (px) for the per-event densities
    reach_px: float = 40.0           # a model is a candidate only within this of its events
    merge_px: float = 0.25           # models closer than this (px per 12.5 ms) are merged
    min_nodes: int = 20              # smaller models are dropped
    trunc: float = 6.0               # cap of the data cost (nats)
    bg_refit: bool = True            # refit the background near 0 (leftover IMU rotation)
    bg_half: float = 0.75            # ... within +- this
    max_events: int = 300000         # events of the window used for the densities


def _nodes(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """Scoring events -> unique pixels: (node of each event, number of nodes)."""
    pix = np.round(y).astype(np.int64) * 100003 + np.round(x).astype(np.int64)
    _u, inv = np.unique(pix, return_inverse=True)
    return inv, int(inv.max()) + 1


def _merge(vs: np.ndarray, sizes: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy merge of velocities, largest model first (row 0 = background, always kept and
    always first) -> (kept velocities, index of the kept model of every input row)."""
    order = [0] + [int(i) for i in np.argsort(-sizes) if i != 0]
    kept, to = [], np.zeros(len(vs), np.int64)
    for i in order:
        d = [np.linalg.norm(vs[i] - vs[j]) for j in kept]
        if kept and min(d) < tol:
            to[i] = int(np.argmin(d))
        else:
            to[i] = len(kept)
            kept.append(i)
    return vs[kept], to


def _box_events(nx, ny, sup, wi, xr, yr, pad, p: CMaxParams, rng):
    """Mask of the nodes ``sup`` and the window events that can warp into it."""
    box, mk = _mask(nx[sup], ny[sup], p, pad, 0, 0)
    bx, by, bw, bh = box
    e = wi[(xr[wi] >= bx) & (xr[wi] < bx + bw) & (yr[wi] >= by) & (yr[wi] < by + bh)]
    if len(e) > p.max_region_events:
        e = rng.choice(e, p.max_region_events, replace=False)
    return box, mk, e


def segment_iterative(xr: np.ndarray, yr: np.ndarray, t: np.ndarray, t_ref: float,
                      p: CMaxParams = CMaxParams(), a: AssignParams = AssignParams()) -> dict:
    """EMSGC-style multi-model segmentation of the scoring events, started from the
    per-region motions of segment_window:

      models   background (v ~ 0 after derotation) + the region motions, merged within merge_px
      assign   every scoring-event pixel takes the model under which the window's IWE is
               densest at its warped position (data cost -log density, blur assign_sigma),
               with a Potts term between neighbouring pixels (alpha-expansion, as in
               hdems.graphcut_seg); a model is a candidate only within reach_px of its events
      refit    each model's v by CMax on the events now assigned to it (finer levels around
               the current v); small models dropped, close ones merged

    -> the keys of segment_window ("region" = final model, 0 = background)."""
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree

    from hdems.graphcut_seg import GCParams, alpha_expansion, knn_graph

    first = segment_window(xr, yr, t, t_ref, p)
    out = dict(first)
    if len(first["v"]) == 0:
        return out
    sc = first["score_idx"]
    t = np.asarray(t, np.float64)
    dtau = (t_ref - t) / INTERVAL_S
    rng = np.random.default_rng(p.seed)
    wi = np.nonzero(t >= t_ref - p.window_ms / 1000.0)[0]
    if len(wi) > a.max_events:
        wi = np.sort(rng.choice(wi, a.max_events, replace=False))
    node, N = _nodes(xr[sc], yr[sc])
    cnt = np.bincount(node, minlength=N).astype(np.float64)
    nx = np.bincount(node, xr[sc], N) / cnt
    ny = np.bincount(node, yr[sc], N) / cnt
    ndt = np.bincount(node, dtau[sc], N) / cnt
    edges, w = knn_graph(nx, ny, GCParams())
    w = a.lam * w

    # initial models and labels: each node takes its region's model (edge regions -> bg)
    reg = np.zeros(N, np.int64)
    reg[node] = first["region"]
    vs0 = np.concatenate([first["v_bg"][None], np.where(first["edge"][:, None], first["v_bg"], first["v"])])
    sizes = np.r_[np.inf, np.bincount(first["region"], minlength=len(first["v"])).astype(np.float64)]
    sizes[1:][first["edge"]] = 0
    V, to = _merge(vs0, sizes, a.merge_px)
    labels = to[reg + 1]

    x0, y0 = int(np.floor(xr[wi].min())) - 2, int(np.floor(yr[wi].min())) - 2
    Wg = int(np.ceil(xr[wi].max())) - x0 + 3
    Hg = int(np.ceil(yr[wi].max())) - y0 + 3
    big = 1e3
    pts = np.stack([nx, ny], 1)
    for _it in range(a.iters):
        L = len(V)
        cost = np.full((N, L), big)
        for m in range(L):
            if m > 0 and not (labels == m).any():
                continue
            I = _iwe((xr[wi] - x0 + V[m, 0] * dtau[wi])[None], (yr[wi] - y0 + V[m, 1] * dtau[wi])[None],
                     Hg, Wg, a.assign_sigma)[0]
            d = map_coordinates(I, [ny - y0 + V[m, 1] * ndt, nx - x0 + V[m, 0] * ndt], order=1, mode="constant")
            c = -np.log(d + 1e-3)
            if m > 0:                       # candidate only near the model's current events
                dist, _j = cKDTree(pts[labels == m]).query(pts, k=1)
                c = np.where(dist <= a.reach_px, c, big)
            cost[:, m] = c
        cmin = cost.min(1, keepdims=True)
        cost = np.where(cost >= big, big, np.minimum(cost - cmin, a.trunc))
        labels = alpha_expansion(cost, edges, w, labels)

        # refit each model on its events; drop small ones
        newV, keep = [], []
        for m in range(L):
            sup = labels == m
            if m > 0 and sup.sum() < a.min_nodes:
                continue
            if (m == 0 and not a.bg_refit) or not sup.any():
                newV.append(V[m])
                keep.append(m)
                continue
            center, half = ((0.0, 0.0), a.bg_half) if m == 0 else (V[m], 1.0)
            reach = (np.abs(center).max() + half + 0.5) * float(dtau[wi].max())     # warps stay in here
            box, mk, e = _box_events(nx, ny, sup, wi, xr, yr, reach, p, rng)
            v, _s = best_velocity(xr[e], yr[e], dtau[e], box, mk, p, center=center, start_level=1, half=half)
            newV.append(v)
            keep.append(m)
        remap = np.full(L, -1)
        remap[keep] = np.arange(len(keep))
        lost = remap[labels] < 0
        labels = remap[labels]
        if lost.any():
            labels[lost] = np.argmin(cost[lost][:, keep], 1)
        V = np.stack(newV)
        sizes = np.bincount(labels, minlength=len(V)).astype(np.float64)
        sizes[0] = np.inf
        V, to = _merge(V, sizes, a.merge_px)
        labels = to[labels]

    # per scoring event; gain of each model over the background on its own events
    lab_ev = labels[node]
    speed_m = np.linalg.norm(V - V[0], axis=1)
    speed_m[0] = 0.0
    gain_m = np.zeros(len(V))
    for m in range(1, len(V)):
        sup = labels == m
        if sup.any():
            reach = (np.abs(V[m]).max() + 0.5) * float(dtau[wi].max())
            box, mk, e = _box_events(nx, ny, sup, wi, xr, yr, reach, p, rng)
            s = sharpness(xr[e], yr[e], dtau[e], np.stack([V[m], V[0]]), box, mk, p.sigmas[-1])
            gain_m[m] = s[0] / max(s[1], 1e-9) - 1.0
    out.update(region=lab_ev, speed=speed_m[lab_ev], gain=gain_m[lab_ev], v=V, v_bg=V[0],
               n=np.bincount(lab_ev, minlength=len(V)), edge=np.zeros(len(V), bool),
               own=(lab_ev > 0).astype(np.float64))
    return out
