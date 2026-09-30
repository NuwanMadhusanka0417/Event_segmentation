"""Training-free motion segmentation: cluster event pixels in hypervector space.

No ego-motion fit, no CNN, no RANSAC. Every event pixel p of the reference surface
is encoded as ONE hypervector that binds where it is to how it moves:

    H(p) = P(x, y) ⊙ V(u, v),   P = X^(x/σs) ⊙ Y^(y/σs),   V = U^(u/σv) ⊙ W^(v/σv)

(FPE with the existing Gaussian position / velocity phases, RAW flow, FIXED scales).
With Gaussian phases  Re<H_i, H_j>/d ≈ exp(-Δpos²/2σs²) · exp(-Δvel²/2σv²), so pixels
that are close AND move alike are similar. Bundling a cluster, M_k = Σ H_i, stores a
position -> velocity lookup, so one prototype can hold a smoothly VARYING motion
field (a rotating object, or the whole background seen by a moving camera).

Pipeline (vsa_group_objects):
  1. kernel k-means   assign each pixel to the most similar prototype, re-bundle
  2. merge            neighbouring clusters whose motion is CONTINUOUS across their
                      shared border are one surface. (Whole-prototype similarity
                      cannot do this: only pixel pairs that are close in space
                      contribute, so two halves of a large background share almost
                      nothing.) Small leftovers join their best neighbour.
  3. background       the cluster most similar to the previous frame's background
                      prototype (per sequence), else a border / extent rule;
                      clusters that move like the background around them join it
  4. smooth labels    majority filter, then objects 1..K by size

Real-valued implementation: H = exp(iθ) is stored as [cos θ | sin θ] (N, 2d), so
Re<H_i, M_k> is a plain dot product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hdems.config import resolution_ratio_of

BACKGROUNDS = ("temporal", "border", "extent")
_MIN_BAND = 10          # pixels per side used to compare motion across a border


@dataclass(frozen=True)
class VSAGroupingParams:
    """All distances / velocities in WORKING-resolution units (see params_from_config)."""

    sigma_s: float = 15.0          # position kernel width (px)
    sigma_v: float = 0.25          # velocity kernel width (px per 12.5 ms interval)
    k_init: int = 24               # initial prototypes: over-segment, then merge
    iters: int = 10                # assign / re-bundle iterations
    max_pixels: int = 60000        # k-means runs on this many pixels; all are assigned
    adjacency_px: int = 3          # clusters closer than this are neighbours
    merge_band_px: int = 4         # band width on each side of a shared border
    merge_gap_px: int = 0          # skipped zone next to the border (flow is blurred there)
    merge_vel: float = 0.25        # merge if the two bands' median flows differ by less
    min_object_px: int = 40        # smaller clusters are not objects
    min_motion: float = 0.125      # an object must move this much relative to the background
    background: str = "temporal"   # temporal | border | extent
    bg_min_sim: float = 0.3        # temporal prior accepted only above this similarity
    bg_fallback: str = "border"    # border | extent
    border_px: int = 5             # "near the image border" (border rule)
    smooth_labels_px: int = 3      # majority-filter window (0 = off)
    use_confidence: bool = False   # weight pixels by the Σ P² match confidence when bundling
    seed: int = 0
    chunk: int = 8192              # pixels per matmul chunk (memory bound)


def params_from_config(cfg: dict[str, Any]) -> VSAGroupingParams:
    """``vsa_grouping:`` section, given in FULL-resolution units, scaled to the working one."""
    g = cfg.get("vsa_grouping", {}) or {}
    r = resolution_ratio_of(cfg)
    background = str(g.get("background", "temporal"))
    fallback = str(g.get("bg_fallback", "border"))
    if background not in BACKGROUNDS or fallback not in ("border", "extent"):
        raise ValueError(f"vsa_grouping.background must be one of {BACKGROUNDS} and "
                         f"bg_fallback border|extent, got {background!r} / {fallback!r}")
    return VSAGroupingParams(
        sigma_s=float(g.get("sigma_s_px", 30.0)) / r,
        sigma_v=float(g.get("sigma_v_px", 0.5)) / r,
        k_init=int(g.get("k_init", 24)),
        iters=int(g.get("iters", 10)),
        max_pixels=int(g.get("max_pixels", 60000)),
        adjacency_px=max(1, round(float(g.get("adjacency_px", 6)) / r)),
        merge_band_px=max(1, round(float(g.get("merge_band_px", 8)) / r)),
        merge_gap_px=max(0, round(float(g.get("merge_gap_px", 0)) / r)),
        merge_vel=float(g.get("merge_vel_px", 0.5)) / r,
        min_object_px=max(8, int(g.get("min_object_px", 160)) // (r * r)),
        min_motion=float(g.get("min_motion_px", 0.25)) / r,
        background=background,
        bg_min_sim=float(g.get("bg_min_sim", 0.3)),
        bg_fallback=fallback,
        border_px=max(1, round(10 / r)),
        smooth_labels_px=int(g.get("smooth_labels_px", 5)),
        use_confidence=bool(g.get("use_confidence", False)),
        seed=int(g.get("seed", 0)),
    )


class BackgroundPrior:
    """The previous frame's background prototype, kept PER SEQUENCE.

    The eval index interleaves sequences, but each sequence's frames stay in time
    order. A prior older than ``max_gap`` frames (or from another sequence) is not used.
    """

    def __init__(self, max_gap: int = 5) -> None:
        self.max_gap = max_gap
        self._last: dict[str, tuple[int, torch.Tensor]] = {}

    def get(self, seq_id: str | None, frame_index: int | None) -> torch.Tensor | None:
        if seq_id is None or frame_index is None or seq_id not in self._last:
            return None
        fi, proto = self._last[seq_id]
        return proto if 0 < frame_index - fi <= self.max_gap else None

    def update(self, seq_id: str | None, frame_index: int | None, proto: torch.Tensor | None) -> None:
        if seq_id is not None and frame_index is not None and proto is not None:
            self._last[seq_id] = (int(frame_index), proto.detach())


def _phases(model):
    return model.matcher.phx, model.matcher.phy, model.phi_vx, model.phi_vy


def encode_pixels(xs, ys, vx, vy, phases, sigma_s: float, sigma_v: float) -> torch.Tensor:
    """(N,) coordinates and flow -> (N, 2d) real codes [cos θ | sin θ] of P(x,y) ⊙ V(u,v)."""
    phx, phy, pvx, pvy = phases
    theta = ((xs / sigma_s)[:, None] * phx + (ys / sigma_s)[:, None] * phy
             + (vx / sigma_v)[:, None] * pvx + (vy / sigma_v)[:, None] * pvy)
    return torch.cat([theta.cos(), theta.sin()], dim=1)


def _codes(idx, xs, ys, vx, vy, phases, p):
    return encode_pixels(xs[idx], ys[idx], vx[idx], vy[idx], phases, p.sigma_s, p.sigma_v)


def _assign(X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """argmax_k Re<X_i, M_k> / |M_k|."""
    return (X @ (M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)).T).argmax(1)


def _kmeanspp(X: torch.Tensor, k: int, d: int, rng: np.random.Generator) -> torch.Tensor:
    """k-means++ seeds on unit-modulus codes: distance = 1 - Re<X_i, c>/d."""
    n = X.shape[0]
    first = int(rng.integers(n))
    centers = [first]
    dmin = (1.0 - (X @ X[first]) / d).clamp_min(0)
    for _ in range(1, k):
        w = (dmin ** 2).cpu().numpy().astype(np.float64)
        if w.sum() <= 0:
            break
        nxt = int(rng.choice(n, p=w / w.sum()))
        centers.append(nxt)
        dmin = torch.minimum(dmin, (1.0 - (X @ X[nxt]) / d).clamp_min(0))
    return X[centers].clone()


def _split_mixed(lab: np.ndarray, flow: np.ndarray, p: VSAGroupingParams,
                 max_depth: int = 2, separation: float = 4.0) -> np.ndarray:
    """Split clusters whose velocities form two clearly separate groups.

    Kernel k-means with bundled prototypes can settle on a MIXED cluster (an object
    region and the background next to it in one prototype; measured 41-66 % object
    pixels on a synthetic moving-camera frame): similarity adds over members, so
    nothing splits it, and in the merge it bridges the object into the background.
    2-means on (u, v) per cluster; split when the centres are > 2*merge_vel apart AND
    > ``separation`` x the spread inside the groups. A smooth field (uniform along a
    line gives ~3.5) is left alone; any over-split is rejoined by the continuity merge.
    """
    lab = lab.copy()
    nxt = int(lab.max()) + 1
    queue = [(int(i), 0) for i in np.unique(lab) if i >= 0]
    while queue:
        i, depth = queue.pop()
        m = lab == i
        if depth >= max_depth or m.sum() < 2 * _MIN_BAND:
            continue
        v = np.stack([flow[0][m], flow[1][m]], 1).astype(np.float64)
        c = v[[np.argmin(v[:, 0] + v[:, 1]), np.argmax(v[:, 0] + v[:, 1])]]   # far-apart start
        for _ in range(10):
            a = np.linalg.norm(v[:, None] - c[None], axis=2).argmin(1)
            if a.min() == a.max():
                break
            c = np.stack([v[a == 0].mean(0), v[a == 1].mean(0)])
        if a.min() == a.max():
            continue
        dist = float(np.linalg.norm(c[0] - c[1]))
        spread = float(np.sqrt(np.mean(np.sum((v - c[a]) ** 2, 1)))) + 1e-6
        if dist > 2 * p.merge_vel and dist / spread > separation:
            ys, xs = np.nonzero(m)
            lab[ys[a == 1], xs[a == 1]] = nxt
            queue += [(i, depth + 1), (nxt, depth + 1)]
            nxt += 1
    return lab


_MOMENTS = ("1", "x", "y", "xx", "xy", "yy", "u", "xu", "yu", "v", "xv", "yv")


def _window_moments(m: np.ndarray, flow: np.ndarray, win: int) -> dict[str, np.ndarray]:
    """Window sums (side ``win``, centred on every pixel) of mask m's pixel moments:
    1, x, y, x², xy, y² and u, v times 1, x, y -- in absolute coordinates, float64.
    Sums over a union of disjoint masks are the sums of the parts."""
    from scipy.ndimage import uniform_filter
    H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    mf = m.astype(np.float64)
    u, v = flow[0].astype(np.float64) * mf, flow[1].astype(np.float64) * mf
    terms = {"1": mf, "x": xx * mf, "y": yy * mf, "xx": xx * xx * mf, "xy": xx * yy * mf,
             "yy": yy * yy * mf, "u": u, "xu": xx * u, "yu": yy * u, "v": v, "xv": xx * v,
             "yv": yy * v}
    area = float(win * win)
    return {k: uniform_filter(t, win) * area for k, t in terms.items()}


def _extrapolate(s: dict[str, np.ndarray], sel: np.ndarray, ridge: float = 1e-2):
    """Local affine fit  flow ≈ f0 + J·(p - c)  of each selected window, returned at
    its centre c -> (n, 2). Moments are shifted from absolute to centred coordinates."""
    H, W = s["1"].shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = xx[sel], yy[sel]
    n0, sx, sy = s["1"][sel], s["x"][sel], s["y"][sel]
    dx, dy = sx - cx * n0, sy - cy * n0                                # Σ(x-cx), Σ(y-cy)
    dxx = s["xx"][sel] - 2 * cx * sx + cx * cx * n0
    dyy = s["yy"][sel] - 2 * cy * sy + cy * cy * n0
    dxy = s["xy"][sel] - cx * sy - cy * sx + cx * cy * n0
    A = np.stack([np.stack([n0, dx, dy], -1),
                  np.stack([dx, dxx + ridge * n0, dxy], -1),
                  np.stack([dy, dxy, dyy + ridge * n0], -1)], axis=1)  # (n, 3, 3)
    out = []
    for c in ("u", "v"):
        f, fx, fy = s[c][sel], s["x" + c][sel], s["y" + c][sel]
        rhs = np.stack([f, fx - cx * f, fy - cy * f], -1)[..., None]
        out.append(np.linalg.solve(A, rhs)[:, 0, 0])                    # value at the centre
    return np.stack(out, 1)


def _merge_continuous(lab: np.ndarray, flow: np.ndarray, proto: dict[int, torch.Tensor],
                      p: VSAGroupingParams) -> np.ndarray:
    """Merge neighbouring clusters whose motion is continuous across the shared border.

    At every point where a window of side 2*(gap+band)+1 holds pixels of both
    clusters A and B, fit a local AFFINE motion to each side's pixels and compare
    the two fits at the SAME point (the window centre); the jump is the median of
    |difference| along the border. Comparing plain means would not work: the two
    sides' pixels sit a few pixels apart, and in a rotating object or the
    background of a moving camera the flow changes with position -- measured
    0.23-0.27 px for a 0.05 rad/interval disc with no noise at all. A smooth field
    gives ~0 here; a real motion boundary does not. Pixels closer than
    ``merge_gap_px`` to the other cluster are left out (Eq.12 pooling blurs the
    flow there).

    Small clusters (< min_object_px) then join their most continuous neighbour if the
    jump is below 3x merge_vel, and are dropped (-1) otherwise.
    """
    from scipy.ndimage import distance_transform_edt

    win = 2 * (p.merge_gap_px + p.merge_band_px) + 1
    lab = lab.copy()
    ids = [int(i) for i in np.unique(lab) if i >= 0]
    masks = {i: lab == i for i in ids}
    size = {i: int(masks[i].sum()) for i in ids}
    dist = {i: distance_transform_edt(~masks[i]) for i in ids}
    stats = ({i: _window_moments(masks[i], flow, win) for i in ids}
             if p.merge_gap_px == 0 else {})
    jumps: dict[tuple[int, int], float] = {}                  # cached; refreshed after a merge

    def jump(a: int, b: int) -> float:
        key = (min(a, b), max(a, b))
        if key in jumps:
            return jumps[key]
        jumps[key] = np.inf
        if dist[b][masks[a]].min() > p.adjacency_px:          # not neighbours
            return np.inf
        if p.merge_gap_px == 0:
            sa, sb = stats[a], stats[b]
        else:                                                 # leave the blurred zone out
            # only the pair's bounding box (+ one window) matters: ~5-10x cheaper than
            # filtering the whole image for every pair
            ys, xs = np.nonzero(masks[a] | masks[b])
            y0, y1 = max(int(ys.min()) - win, 0), min(int(ys.max()) + win + 1, lab.shape[0])
            x0, x1 = max(int(xs.min()) - win, 0), min(int(xs.max()) + win + 1, lab.shape[1])
            box = (slice(y0, y1), slice(x0, x1))
            fbox = flow[:, box[0], box[1]]
            sa = _window_moments((masks[a] & (dist[b] >= p.merge_gap_px))[box], fbox, win)
            sb = _window_moments((masks[b] & (dist[a] >= p.merge_gap_px))[box], fbox, win)
        both = (sa["1"] >= 5.5) & (sb["1"] >= 5.5)            # >= 6 pixels of each for a fit
        if both.sum() < _MIN_BAND:
            return np.inf
        d = _extrapolate(sa, both) - _extrapolate(sb, both)
        jumps[key] = float(np.median(np.hypot(d[:, 0], d[:, 1])))
        return jumps[key]

    def do_merge(a: int, b: int) -> None:
        lab[masks[b]] = a
        masks[a] |= masks.pop(b)
        size[a] += size.pop(b)
        dist[a] = np.minimum(dist[a], dist.pop(b))
        if stats:                                             # window sums add
            sb = stats.pop(b)
            stats[a] = {k: stats[a][k] + sb[k] for k in _MOMENTS}
        proto[a] = proto[a] + proto.pop(b)                    # bundles add
        ids.remove(b)
        for key in [k for k in jumps if a in k or b in k]:    # a changed, b is gone
            del jumps[key]

    for small_pass in (False, True):
        for _ in range(4 * max(len(ids), 1)):                 # bounded
            best, best_j = None, np.inf
            for ai, a in enumerate(ids):
                for b in ids[ai + 1:]:
                    if small_pass and min(size[a], size[b]) >= p.min_object_px:
                        continue
                    j = jump(a, b)
                    if j < best_j:
                        best, best_j = (a, b), j
            limit = 3.0 * p.merge_vel if small_pass else p.merge_vel
            if best is None or best_j >= limit:
                break
            a, b = best
            if size[b] > size[a]:                             # keep the larger id
                a, b = b, a
            do_merge(a, b)
    for i in list(ids):                                       # still too small: not an object
        if masks[i].sum() < p.min_object_px:
            lab[masks[i]] = -1
            proto.pop(i, None)
    return lab


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-9))


def _pick_background(lab, proto, prior, p) -> tuple[int | None, str, float]:
    ids = [i for i in proto if (lab == i).any()]
    if not ids:
        return None, "none", float("nan")
    if p.background == "temporal" and prior is not None:
        sims = {i: _cos(proto[i], prior) for i in ids}
        best = max(sims, key=sims.get)
        if sims[best] >= p.bg_min_sim:
            return best, "temporal", sims[best]
    rule = p.background if p.background != "temporal" else p.bg_fallback
    H, W = lab.shape
    if rule == "border":
        b = p.border_px
        edge = np.zeros((H, W), dtype=bool)
        edge[:b], edge[-b:], edge[:, :b], edge[:, -b:] = True, True, True, True
        score = {i: int(((lab == i) & edge).sum()) for i in ids}
    else:                                                     # extent: widest bounding box
        score = {}
        for i in ids:
            ys, xs = np.nonzero(lab == i)
            score[i] = int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    return max(score, key=score.get), rule, float("nan")


def _absorb_static(lab, flow, bg, p) -> None:
    """Clusters that move like the background around them are background.

    Per pixel: |own flow - mean background flow within sigma_s|, then the median of
    those MAGNITUDES. A median velocity VECTOR would be wrong: a rotating object's
    velocities point every way, so its median vector is ~0 and it would look static.
    """
    from scipy.ndimage import uniform_filter
    bgm = (lab == bg).astype(np.float64)
    if not bgm.any():
        return
    ref_all = np.array([np.median(flow[0][lab == bg]), np.median(flow[1][lab == bg])])
    for win in (2 * int(round(p.sigma_s)) + 1, 4 * int(round(p.sigma_s)) + 1):
        cnt = uniform_filter(bgm, win)
        if (cnt > 0).any():
            break
    ok = cnt > 1e-9
    bu = np.where(ok, uniform_filter(flow[0] * bgm, win) / np.maximum(cnt, 1e-9), ref_all[0])
    bv = np.where(ok, uniform_filter(flow[1] * bgm, win) / np.maximum(cnt, 1e-9), ref_all[1])
    for i in [int(i) for i in np.unique(lab) if i >= 0 and i != bg]:
        m = lab == i
        diff = np.hypot(flow[0][m] - bu[m], flow[1][m] - bv[m])
        if float(np.median(diff)) < p.min_motion:
            lab[m] = bg


def _majority(lab: np.ndarray, events: np.ndarray, w: int) -> np.ndarray:
    """Majority filter of labels (>= 0) over event pixels in a w x w window."""
    from scipy.ndimage import uniform_filter
    ids = [int(i) for i in np.unique(lab[events]) if i >= 0]
    if len(ids) < 2 or w < 2:
        return lab
    votes = np.stack([uniform_filter((lab == i).astype(np.float32), size=w) for i in ids])
    out = lab.copy()
    out[events] = np.array(ids)[votes.argmax(0)][events]
    return out


@torch.no_grad()
def vsa_group_objects(
    flow,
    events,
    model,
    params: VSAGroupingParams,
    *,
    conf=None,
    prior: BackgroundPrior | None = None,
    seq_id: str | None = None,
    frame_index: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """flow (2,H,W) working px per interval, events (H,W) bool
    -> (objects (H,W) int: 0 = background / no event, 1..K objects by size, info).

    info: bg_prototype, prototypes, n_init, n_after_merge, bg_source, bg_similarity.
    ``prior`` (+ seq_id, frame_index) enables and is updated by the temporal background rule.
    """
    p = params
    phases = _phases(model)
    dev = phases[0].device
    d = int(phases[0].numel())
    flow_np = (flow.detach().cpu().numpy() if isinstance(flow, torch.Tensor) else np.asarray(flow))
    flow_np = flow_np.astype(np.float32)
    ev = (events.detach().cpu().numpy() if isinstance(events, torch.Tensor) else np.asarray(events))
    ev = ev.astype(bool)
    H, W = ev.shape
    objects = np.zeros((H, W), dtype=np.int64)
    info: dict[str, Any] = {"bg_prototype": None, "prototypes": {}, "n_init": 0,
                            "n_after_merge": 0, "bg_source": "none", "bg_similarity": float("nan")}
    ys_np, xs_np = np.nonzero(ev)
    n = ys_np.size
    if n < p.min_object_px:
        return objects, info

    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    xs, ys = t(xs_np), t(ys_np)
    vx, vy = t(flow_np[0][ys_np, xs_np]), t(flow_np[1][ys_np, xs_np])
    if p.use_confidence and conf is not None:
        c = conf.detach().cpu().numpy() if isinstance(conf, torch.Tensor) else np.asarray(conf)
        w = t(np.clip(c.reshape(H, W)[ys_np, xs_np], 0, None))
    else:
        w = torch.ones(n, device=dev)
    rng = np.random.default_rng(p.seed)

    # 1. kernel k-means on a subsample
    sub = np.sort(rng.choice(n, p.max_pixels, replace=False)) if n > p.max_pixels else np.arange(n)
    X = torch.cat([_codes(sub[i:i + p.chunk], xs, ys, vx, vy, phases, p)
                   for i in range(0, sub.size, p.chunk)])
    w_sub = w[torch.as_tensor(sub, device=dev)][:, None]
    M = _kmeanspp(X, min(p.k_init, sub.size), d, rng)
    lab_sub = None
    for _ in range(p.iters):
        new = torch.cat([_assign(X[i:i + p.chunk], M) for i in range(0, X.shape[0], p.chunk)])
        if lab_sub is not None and torch.equal(new, lab_sub):
            break
        lab_sub = new
        counts = torch.bincount(lab_sub, minlength=M.shape[0])
        M = torch.zeros_like(M).index_add_(0, lab_sub, X * w_sub)[counts > 0]   # re-bundle
    del X

    # assign ALL pixels, split mixed clusters, then bundle the final prototypes
    lab_all = torch.cat([_assign(_codes(np.arange(i, min(i + p.chunk, n)), xs, ys, vx, vy,
                                        phases, p), M) for i in range(0, n, p.chunk)])
    lab = np.full((H, W), -1, dtype=np.int64)
    lab[ys_np, xs_np] = lab_all.cpu().numpy()
    lab = _split_mixed(lab, flow_np, p)
    lab_pix = torch.as_tensor(lab[ys_np, xs_np], device=dev)
    sums = torch.zeros(int(lab.max()) + 1, 2 * d, device=dev)
    for i in range(0, n, p.chunk):
        idx = np.arange(i, min(i + p.chunk, n))
        sums.index_add_(0, lab_pix[i:i + len(idx)],
                        _codes(idx, xs, ys, vx, vy, phases, p) * w[i:i + len(idx)][:, None])
    proto = {int(k): sums[k] for k in np.unique(lab[lab >= 0])}
    info["n_init"] = len(proto)

    # 2. merge clusters whose motion is continuous across their shared border
    lab = _merge_continuous(lab, flow_np, proto, p)
    info["n_after_merge"] = len(proto)

    # 3. background: temporal prior, else border / extent
    bg, src, sim = _pick_background(lab, proto, prior.get(seq_id, frame_index) if prior else None, p)
    info.update(bg_source=src, bg_similarity=sim)
    if bg is None:
        return objects, info
    _absorb_static(lab, flow_np, bg, p)
    # the final background bundles every cluster absorbed into it
    for i in [i for i in proto if i != bg and not (lab == i).any()]:
        proto[bg] = proto[bg] + proto.pop(i)
    bg_proto = proto[bg]
    info["bg_prototype"] = bg_proto
    if prior is not None:
        prior.update(seq_id, frame_index, bg_proto)

    # 4. smooth labels, drop objects that became too small, number objects by size
    lab = np.where(lab < 0, bg, lab)                          # dropped fragments -> background
    lab = np.where(ev, lab, -1)
    if p.smooth_labels_px >= 2:
        lab = _majority(lab, ev, p.smooth_labels_px)
    ids, counts = np.unique(lab[(lab >= 0) & (lab != bg)], return_counts=True)
    keep = [(i, c) for i, c in zip(ids.tolist(), counts.tolist()) if c >= p.min_object_px]
    for new_id, (old, _) in enumerate(sorted(keep, key=lambda ic: -ic[1]), start=1):
        objects[lab == old] = new_id
    info["prototypes"] = {i: proto[i] for i in proto if i != bg}
    return objects, info
