"""Multi-model graph-cut segmentation (hdems/graphcut_seg.py) on synthetic event pixels."""

import numpy as np
import pytest

maxflow = pytest.importorskip("maxflow")

from hdems.graphcut_seg import GCParams, alpha_expansion, segment  # noqa: E402

H, W = 60, 80


def _scene(seed=0, noise=0.2):
    """Sparse 'event' pixels: static background (flow ~0 after derotation) and two movers,
    one translating, one rotating (flow direction varies across it)."""
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    fired = g.random((H, W)) < 0.35
    a = (xx >= 10) & (xx < 30) & (yy >= 10) & (yy < 30)
    c = (xx >= 50) & (xx < 72) & (yy >= 30) & (yy < 52)
    u = np.zeros((2, H, W))
    u[0][a] = 2.5
    u[1][a] = -1.0
    u[0][c] = -(yy[c] - 41) * 0.25
    u[1][c] = (xx[c] - 61) * 0.25
    y, x = np.nonzero(fired)
    flow = u[:, y, x].T + g.normal(0, noise, (len(x), 2))
    truth = np.where(a[y, x], 1, np.where(c[y, x], 2, 0))
    return x, y, flow, truth


def _full_rows(flow):
    n = len(flow)
    row_node = np.repeat(np.arange(n), 2)
    D = np.tile(np.eye(2), (n, 1))
    return row_node, D, flow.reshape(-1)


def _normal_rows(flow, seed=1):
    """One normal-flow constraint per pixel along a random edge direction (aperture)."""
    g = np.random.default_rng(seed)
    ang = g.uniform(0, np.pi, len(flow))
    n0 = np.stack([np.cos(ang), np.sin(ang)], 1)
    return np.arange(len(flow)), n0, (flow * n0).sum(1)


def _check(out, truth, delta=0.8):
    moving = out["motion_score"] > delta
    iou = (moving & (truth > 0)).sum() / max((moving | (truth > 0)).sum(), 1)
    assert iou > 0.9, iou
    # each mover is (mostly) one region, distinct from the other
    la = np.bincount(out["labels"][truth == 1]).argmax()
    lc = np.bincount(out["labels"][truth == 2]).argmax()
    assert la != lc and la != 0 and lc != 0


def test_full_flow_two_movers_one_rotating():
    x, y, flow, truth = _scene()
    rn, D, b = _full_rows(flow)
    _check(segment(x, y, rn, D, b, H, W, GCParams(min_nodes=20)), truth)


def test_normal_flow_constraints_suffice():
    x, y, flow, truth = _scene(seed=2)
    rn, D, b = _normal_rows(flow)
    _check(segment(x, y, rn, D, b, H, W, GCParams(min_nodes=20)), truth, delta=0.8)


def test_smoothness_cleans_isolated_noise():
    # a static scene with a few pixels of large random flow: with smoothness they stay static
    x, y, flow, truth = _scene(seed=3)
    keep = truth == 0
    x, y, flow = x[keep], y[keep], flow[keep].copy()
    g = np.random.default_rng(4)
    spikes = g.choice(len(x), 15, replace=False)
    flow[spikes] += g.normal(0, 3.0, (15, 2))
    rn, D, b = _full_rows(flow)
    out = segment(x, y, rn, D, b, H, W, GCParams(min_nodes=20, lam=2.0))
    assert (out["motion_score"] > 0.8).sum() == 0


def test_alpha_expansion_reaches_the_obvious_optimum():
    # chain of 6 nodes, unaries favour 0 0 0 1 1 1, strong smoothness keeps the two blocks
    cost = np.array([[0, 2], [0, 2], [0.4, 0], [2, 0], [2, 0], [2, 0]], float)
    edges = np.array([[i, i + 1] for i in range(5)])
    lab = alpha_expansion(cost, edges, np.full(5, 1.0), np.zeros(6, int), sweeps=2)
    assert lab.tolist() == [0, 0, 1, 1, 1, 1] or lab.tolist() == [0, 0, 0, 1, 1, 1]
