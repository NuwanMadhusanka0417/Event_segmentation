"""Group moving event pixels into OBJECTS -- one colour per object.

The CNN answers "is this pixel moving?". This module answers "which object?".
Two methods (``grouping.method``):

spatial (default, 2026-10-15) -- LOCATION first, motion only to separate:
  1. spatial groups      moving event pixels within adjacency_px of each other are
                         one group (events are sparse edges, so they are linked
                         through a small gap, not by strict connectivity)
  2. motion split        a group is cut in two only on strong evidence: two affine
                         motions explain its flow much better than one (median error
                         < split_gain x), they differ by > split_min_diff at the
                         median pixel, and both parts stay large -- a rule standing
                         in for the label cost / smoothness of graph-cut methods
                         (Zhou et al. 2021, arXiv 2012.08730) and split-and-merge
                         (0-MMS, Parameshwara et al. 2020). Recursive, split_depth.
  3. motion filter       a group that hardly moves relative to the camera is
                         background the classifier marked by mistake (min_motion)
  Measured on 60 validation frames with ground-truth moving pixels (object mIoU,
  scripts/check_grouping.py): motion-first (below) 0.747 with 19% of the objects
  split into pieces; spatial with a 4 px link, no split, no motion filter 0.845 with
  1% split and 17% of the groups gluing two touching objects. The motion split does
  NOT pay off on this flow (0.806): the pooled flow makes an object's rim look like a
  second motion. The motion filter at the 1 px / 50 ms label threshold drops real
  slow movers (0.795). Defaults in configs/evimo_seg.yaml follow these numbers.

motion (the original) -- motion first:
  1. sequential RANSAC   fit an affine flow model  u,v = A·[x, y, 1]  to the largest
                         consensus set, remove it, repeat -> candidate motions
  2. assignment          every moving pixel joins the model it fits best (if any)
  3. spatial split       one model covering two far-apart regions = two objects
                         (pieces smaller than min_support / 4 are dropped)
  4. merge               neighbouring groups whose JOINT motion still fits one model
                         are one object; small leftovers join their neighbour
  The tight tolerance on noisy, pooled flow puts one object's pixels into several
  models, and the strict merge rarely rejoins them.

Output: an int map, 0 = not an object, 1..K = objects ordered by size.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from hdems.config import resolution_ratio_of


@dataclass(frozen=True)
class GroupingParams:
    """All distances in WORKING-resolution pixels (see params_from_config)."""

    tol: float = 0.35           # a pixel fits a model if its flow is within this (px/interval)
    min_support: int = 40       # pixels: a smaller group is not an object
    max_models: int = 8
    iters: int = 300            # RANSAC hypotheses per model
    score_sample: int = 4000    # pixels used to score a hypothesis (speed)
    merge_tol: float = 0.35     # merge neighbours whose joint affine fit is this good
    adjacency_px: int = 3       # how close two groups must be to count as neighbours
    min_motion: float = 0.0     # a group moving slower than this relative to the camera is
    #                             not an independently moving object (stray CNN pixels)
    seed: int = 0
    method: str = "motion"      # spatial | motion (see the module docstring)
    split_min_diff: float = 0.5   # spatial: the two motions must differ by this (px/interval)
    split_gain: float = 0.5       # spatial: 2-motion median error < gain x 1-motion error
    split_depth: int = 2          # spatial: recursive splits per group (<= 2^depth parts)


METHODS = ("spatial", "motion")


def params_from_config(cfg: dict[str, Any]) -> GroupingParams:
    """``grouping:`` section, given in FULL-resolution units, scaled to the working one."""
    g = cfg.get("grouping", {}) or {}
    r = resolution_ratio_of(cfg)
    method = str(g.get("method", "spatial"))
    if method not in METHODS:
        raise ValueError(f"grouping.method must be one of {METHODS}, got {method!r}")
    return GroupingParams(
        tol=float(g.get("tol_px", 0.7)) / r,
        min_support=max(8, int(g.get("min_object_px", 160)) // (r * r)),
        max_models=int(g.get("max_objects", 8)),
        iters=int(g.get("ransac_iters", 300)),
        merge_tol=float(g.get("merge_tol_px", 0.7)) / r,
        adjacency_px=max(1, int(g.get("adjacency_px", 4)) // r),
        min_motion=float(g.get("min_motion_px", 0.0)) / r,
        method=method,
        split_min_diff=float(g.get("split_min_diff_px", 2.0)) / r,
        split_gain=float(g.get("split_gain", 0.5)),
        split_depth=int(g.get("split_depth", 0)),
    )


def _design(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float64)


def _fit(A: np.ndarray, fu: np.ndarray, fv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cu = np.linalg.lstsq(A, fu, rcond=None)[0]
    cv = np.linalg.lstsq(A, fv, rcond=None)[0]
    return cu, cv


def _err(A: np.ndarray, fu: np.ndarray, fv: np.ndarray, model) -> np.ndarray:
    cu, cv = model
    return np.hypot(A @ cu - fu, A @ cv - fv)


def _ransac_models(A, fu, fv, p: GroupingParams, rng) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sequential RANSAC: repeatedly take the largest set that one affine motion explains."""
    free = np.ones(len(fu), dtype=bool)
    models = []
    for _ in range(p.max_models):
        idx = np.flatnonzero(free)
        if idx.size < p.min_support:
            break
        score = idx if idx.size <= p.score_sample else rng.choice(idx, p.score_sample, replace=False)
        best, best_n = None, -1
        for _ in range(p.iters):
            s = rng.choice(idx, 3, replace=False)
            try:
                model = (np.linalg.solve(A[s], fu[s]), np.linalg.solve(A[s], fv[s]))
            except np.linalg.LinAlgError:                 # collinear sample
                continue
            n_in = int((_err(A[score], fu[score], fv[score], model) < p.tol).sum())
            if n_in > best_n:
                best, best_n = model, n_in
        if best is None:
            break
        inl = idx[_err(A[idx], fu[idx], fv[idx], best) < p.tol]
        if inl.size < p.min_support:
            break
        best = _fit(A[inl], fu[inl], fv[inl])              # refit on the consensus set
        inl = idx[_err(A[idx], fu[idx], fv[idx], best) < p.tol]
        if inl.size < p.min_support:
            break
        models.append(best)
        free[inl] = False
    return models


def _components(mask: np.ndarray, link_px: int) -> tuple[np.ndarray, int]:
    """Connected pieces of ``mask``, treating pixels within link_px as touching.

    Event pixels are sparse (edges only), so plain connectivity would shatter one
    object into many pieces; linking through a small dilation keeps it whole.
    """
    from scipy.ndimage import binary_dilation, label
    linked, n = label(binary_dilation(mask, iterations=link_px) if link_px else mask)
    return np.where(mask, linked, 0), n


def group_objects(flow: np.ndarray, mask: np.ndarray, p: GroupingParams) -> np.ndarray:
    """Moving pixels -> object ids (0 = not an object, 1..K by size).

    flow : (2, H, W)  ego-compensated flow (px per interval, working resolution)
    mask : (H, W) bool  pixels to group -- normally CNN "moving" AND has events
    """
    if p.method == "spatial":
        return _group_spatial(flow, mask, p)
    if p.method != "motion":
        raise ValueError(f"grouping method must be one of {METHODS}, got {p.method!r}")
    return _group_by_motion(flow, mask, p)


# ------------------------------------------------------------------ spatial first
def _two_motions(A: np.ndarray, fu: np.ndarray, fv: np.ndarray, p: GroupingParams,
                 rng: np.random.Generator) -> np.ndarray | None:
    """Split one group's pixels by motion, only on strong evidence -> bool (True = part B),
    or None to keep the group whole.

    Two affine motions from RANSAC (the largest consensus set, then the rest), refined
    by 3 rounds of "each pixel to the closer motion". Kept only if the two motions
    explain the flow much better than one (median error < split_gain x) and differ by
    more than split_min_diff at the median pixel -- flow noise alone can do neither.
    """
    n = len(fu)
    if n < 2 * p.min_support:
        return None
    one = _fit(A, fu, fv)
    cost1 = float(np.median(_err(A, fu, fv, one)))
    first = _ransac_models(A, fu, fv, replace(p, max_models=1), rng)
    if not first:
        return None
    side = _err(A, fu, fv, first[0]) >= p.tol                  # B = not explained by the first
    for _ in range(3):
        if side.sum() < p.min_support or (~side).sum() < p.min_support:
            return None
        ma, mb = _fit(A[~side], fu[~side], fv[~side]), _fit(A[side], fu[side], fv[side])
        ea, eb = _err(A, fu, fv, ma), _err(A, fu, fv, mb)
        side = eb < ea
    if side.sum() < p.min_support or (~side).sum() < p.min_support:
        return None
    cost2 = float(np.median(np.minimum(ea, eb)))
    (au, av), (bu, bv) = ma, mb
    diff = float(np.median(np.hypot(A @ au - A @ bu, A @ av - A @ bv)))
    if cost2 >= p.split_gain * cost1 or diff <= p.split_min_diff:
        return None
    return side


def _split_group(group: np.ndarray, flow: np.ndarray, p: GroupingParams,
                 rng: np.random.Generator, depth: int) -> list[np.ndarray]:
    """One spatial group (H, W bool) -> its parts after recursive motion splits."""
    if depth <= 0:
        return [group]
    ys, xs = np.nonzero(group)
    A = _design(ys.astype(np.float64), xs.astype(np.float64))
    side = _two_motions(A, flow[0][ys, xs].astype(np.float64),
                        flow[1][ys, xs].astype(np.float64), p, rng)
    if side is None:
        return [group]
    parts = []
    for s in (False, True):
        part = np.zeros_like(group)
        part[ys[side == s], xs[side == s]] = True
        # a motion's pixels may be scattered: its spatial pieces are parts of their own
        comp, n = _components(part, p.adjacency_px)
        for c in range(1, n + 1):
            piece = comp == c
            if piece.sum() >= p.min_support:
                parts += _split_group(piece, flow, p, rng, depth - 1)
    return parts or [group]


def _group_spatial(flow: np.ndarray, mask: np.ndarray, p: GroupingParams) -> np.ndarray:
    H, W = mask.shape
    out = np.zeros((H, W), dtype=np.int64)
    if mask.sum() < p.min_support:
        return out
    rng = np.random.default_rng(p.seed)
    comp, n = _components(mask, p.adjacency_px)
    groups: list[np.ndarray] = []
    for c in range(1, n + 1):
        g = comp == c
        if g.sum() >= p.min_support:
            groups += _split_group(g, flow, p, rng, p.split_depth)
    if p.min_motion > 0:
        speed = np.hypot(flow[0], flow[1])
        groups = [g for g in groups if float(np.median(speed[g])) >= p.min_motion]
    groups.sort(key=lambda g: -int(g.sum()))
    for i, g in enumerate(groups[: max(p.max_models, 1)], start=1):
        out[g & (out == 0)] = i
    return out


# ------------------------------------------------------------------- motion first
def _group_by_motion(flow: np.ndarray, mask: np.ndarray, p: GroupingParams) -> np.ndarray:
    H, W = mask.shape
    out = np.zeros((H, W), dtype=np.int64)
    ys, xs = np.nonzero(mask)
    if ys.size < p.min_support:
        return out
    rng = np.random.default_rng(p.seed)
    A = _design(ys.astype(np.float64), xs.astype(np.float64))
    fu, fv = flow[0][ys, xs].astype(np.float64), flow[1][ys, xs].astype(np.float64)

    # 1-2. candidate motions, then every pixel joins the model it fits best
    models = _ransac_models(A, fu, fv, p, rng)
    if not models:
        return out
    errs = np.stack([_err(A, fu, fv, m) for m in models], axis=1)      # (n, K)
    best = errs.argmin(1)
    ok = errs[np.arange(len(best)), best] < 2.0 * p.tol                # generous: final say is the merge
    motion = np.zeros((H, W), dtype=np.int64)
    motion[ys[ok], xs[ok]] = best[ok] + 1

    # 3. one motion over two separate regions = two objects
    groups = np.zeros((H, W), dtype=np.int64)
    nxt = 1
    for k in range(1, len(models) + 1):
        comp, n = _components(motion == k, p.adjacency_px)
        for c in range(1, n + 1):
            piece = comp == c
            if piece.sum() >= max(3, p.min_support // 4):             # tiny bits are dropped
                groups[piece] = nxt
                nxt += 1

    # 4. merge neighbours that one motion explains jointly; absorb small groups
    groups = _merge(groups, flow, p)

    # 5. an object must move relative to the camera: the flow here is already
    #    ego-compensated, so a group with ~zero residual motion is background that
    #    the CNN marked by mistake, not an independently moving object
    if p.min_motion > 0:
        speed = np.hypot(flow[0], flow[1])
        for i in [int(i) for i in np.unique(groups) if i > 0]:
            if float(np.median(speed[groups == i])) < p.min_motion:
                groups[groups == i] = 0

    # relabel 1..K by size (largest object first)
    ids, counts = np.unique(groups[groups > 0], return_counts=True)
    for new, old in enumerate(ids[np.argsort(-counts)], start=1):
        out[groups == old] = new
    return out


def _merge(groups: np.ndarray, flow: np.ndarray, p: GroupingParams) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    groups = groups.copy()
    for _ in range(64):                                  # bounded; usually 1-3 passes
        ids = [int(i) for i in np.unique(groups) if i > 0]
        masks = {i: groups == i for i in ids}
        sizes = {i: int(m.sum()) for i, m in masks.items()}
        grown = {i: binary_dilation(m, iterations=p.adjacency_px) for i, m in masks.items()}
        best_pair, best_cost = None, np.inf
        for ai, a in enumerate(ids):
            ma = masks[a]
            for b in ids[ai + 1:]:
                mb = masks[b]
                if not (grown[a] & mb).any():            # not neighbours
                    continue
                ys, xs = np.nonzero(ma | mb)
                A = _design(ys.astype(np.float64), xs.astype(np.float64))
                fu, fv = flow[0][ys, xs].astype(np.float64), flow[1][ys, xs].astype(np.float64)
                cost = float(np.median(_err(A, fu, fv, _fit(A, fu, fv))))
                small = min(sizes[a], sizes[b]) < p.min_support
                # a small group joins its best-fitting neighbour even if the fit is loose
                limit = 3.0 * p.merge_tol if small else p.merge_tol
                if cost < limit and cost < best_cost:
                    best_pair, best_cost = (a, b), cost
        if best_pair is None:
            break
        groups[groups == best_pair[1]] = best_pair[0]
    # groups still smaller than an object after merging are dropped
    for i in [int(i) for i in np.unique(groups) if i > 0]:
        if (groups == i).sum() < p.min_support:
            groups[groups == i] = 0
    return groups
