"""Spatial-first object grouping (grouping.method: spatial)."""

import numpy as np

from hdems.grouping import GroupingParams, group_objects

H, W = 60, 80
P = GroupingParams(method="spatial", min_support=40, adjacency_px=2, tol=0.35,
                   split_min_diff=0.5, split_gain=0.5, min_motion=0.1)


def _noise(seed=0, sigma=0.15):
    return np.random.default_rng(seed).normal(0.0, sigma, (2, H, W))


def _sparse(mask, seed=1, keep=0.6):
    """Events are sparse: keep a random subset of the object's pixels."""
    return mask & (np.random.default_rng(seed).random(mask.shape) < keep)


def test_one_rotating_object_stays_whole_despite_noise():
    # flow direction varies across the object (rotation about its centre) + noise:
    # the old motion-first grouping cut such objects into several models
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    obj = np.zeros((H, W), bool)
    obj[10:50, 15:65] = True
    flow = np.stack([-(yy - 30) * 0.08, (xx - 40) * 0.08]) + _noise()
    out = group_objects(flow, _sparse(obj), P)
    assert len(np.unique(out[out > 0])) == 1


def test_two_touching_objects_with_different_motion_are_split():
    a = np.zeros((H, W), bool)
    b = np.zeros((H, W), bool)
    a[10:50, 10:40] = True
    b[10:50, 40:70] = True                           # touching a
    flow = _noise()
    flow[0][a] += 2.0
    flow[1][b] -= 2.0
    out = group_objects(flow, _sparse(a | b), P)
    ids = [int(i) for i in np.unique(out[out > 0])]
    assert len(ids) == 2
    for m in (a, b):                                 # each object mostly one id
        vals = out[m & (out > 0)]
        assert np.bincount(vals).max() / vals.size > 0.9


def test_separate_objects_with_the_same_motion_are_two_objects():
    a = np.zeros((H, W), bool)
    b = np.zeros((H, W), bool)
    a[5:25, 5:30] = True
    b[35:55, 45:75] = True
    flow = _noise()
    flow[0] += 1.5
    out = group_objects(flow, _sparse(a | b), P)
    assert len(np.unique(out[out > 0])) == 2


def test_a_static_blob_is_not_an_object():
    a = np.zeros((H, W), bool)
    a[10:40, 10:40] = True
    out = group_objects(_noise(sigma=0.02), _sparse(a), P)       # no residual motion
    assert not out.any()
