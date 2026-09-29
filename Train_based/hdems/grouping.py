"""Group moving event pixels into OBJECTS by motion model -- one colour per object.

The CNN answers "is this pixel moving?". This module answers "which object?",
the way EMSGC (Zhou et al.) and cascaded multi-model fitting (Lu et al.) do:
every independently moving object has ONE rigid motion, so its pixels share one
parametric flow model -- even when the flow DIRECTION varies across the object
(rotation, scaling). Pixels are grouped by the model they fit, not by direction.

Pipeline, on the ego-compensated VSA flow of the moving pixels:
  1. sequential RANSAC   fit an affine flow model  u,v = A·[x, y, 1]  to the largest
                         consensus set, remove it, repeat -> candidate motions
  2. assignment          every moving pixel joins the model it fits best (if any)
  3. spatial split       one model covering two far-apart regions = two objects
  4. merge               neighbouring groups whose JOINT motion still fits one model
                         are one object (stops a single object from splitting into
                         several colours); small leftovers join their neighbour
Output: an int map, 0 = not an object, 1..K = objects ordered by size.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def params_from_config(cfg: dict[str, Any]) -> GroupingParams:
    """``grouping:`` section, given in FULL-resolution units, scaled to the working one."""
    g = cfg.get("grouping", {}) or {}
    r = resolution_ratio_of(cfg)
    return GroupingParams(
        tol=float(g.get("tol_px", 0.7)) / r,
        min_support=max(8, int(g.get("min_object_px", 160)) // (r * r)),
        max_models=int(g.get("max_objects", 8)),
        iters=int(g.get("ransac_iters", 300)),
        merge_tol=float(g.get("merge_tol_px", 0.7)) / r,
        adjacency_px=max(1, int(g.get("adjacency_px", 6)) // r),
        min_motion=float(g.get("min_motion_px", 0.25)) / r,
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
    """Moving pixels -> object ids.

    flow : (2, H, W)  ego-compensated flow (px per interval, working resolution)
    mask : (H, W) bool  pixels to group -- normally CNN "moving" AND has events
    """
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
            if piece.sum() >= max(3, p.min_support // 4):             # tiny bits: merged below
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
