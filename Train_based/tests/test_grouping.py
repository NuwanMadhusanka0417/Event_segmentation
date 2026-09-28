"""Object grouping: one colour per independently moving object."""

import math

import numpy as np

from hdems.data.motion_labels import rigid_groups
from hdems.grouping import GroupingParams, group_objects

P = GroupingParams(tol=0.35, min_support=40, adjacency_px=3, merge_tol=0.35)
H, W = 120, 160


def _flow_for(regions):
    """regions: list of (mask, fn(y, x) -> (u, v)) -> (flow (2,H,W), union mask)."""
    rng = np.random.default_rng(0)
    flow = np.zeros((2, H, W))
    mask = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    for m, fn in regions:
        u, v = fn(yy, xx)
        flow[0][m], flow[1][m] = u[m], v[m]
        mask |= m
    flow += rng.normal(0, 0.05, flow.shape)
    # event data is sparse: keep ~40% of the pixels, like edges
    mask &= rng.random((H, W)) < 0.4
    return flow, mask


def _box(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def _one_label(lbl, region, mask):
    vals = np.unique(lbl[region & mask])
    return vals.size == 1 and vals[0] > 0


def test_two_objects_moving_differently_get_two_colours():
    a, b = _box(20, 60, 20, 60), _box(20, 60, 90, 130)
    flow, mask = _flow_for([(a, lambda y, x: (np.full_like(x, 2.0), np.zeros_like(x))),
                            (b, lambda y, x: (np.zeros_like(x), np.full_like(x, -2.0)))])
    lbl = group_objects(flow, mask, P)
    assert _one_label(lbl, a, mask) and _one_label(lbl, b, mask)
    assert np.unique(lbl[a & mask])[0] != np.unique(lbl[b & mask])[0]


def test_rotating_object_is_one_colour_although_direction_varies():
    # flow direction goes all the way round the centre -- still ONE rigid motion
    a = _box(30, 90, 40, 100)
    cy, cx, w = 60.0, 70.0, 0.06
    flow, mask = _flow_for([(a, lambda y, x: (-w * (y - cy), w * (x - cx)))])
    lbl = group_objects(flow, mask, P)
    assert _one_label(lbl, a, mask)


def test_same_motion_far_apart_is_two_objects():
    a, b = _box(10, 40, 10, 40), _box(80, 110, 110, 150)
    same = lambda y, x: (np.full_like(x, 1.5), np.full_like(x, 1.0))
    flow, mask = _flow_for([(a, same), (b, same)])
    lbl = group_objects(flow, mask, P)
    assert _one_label(lbl, a, mask) and _one_label(lbl, b, mask)
    assert np.unique(lbl[a & mask])[0] != np.unique(lbl[b & mask])[0]


def test_touching_parts_with_same_motion_are_one_object():
    a, b = _box(30, 60, 30, 70), _box(60, 90, 30, 70)          # share an edge
    same = lambda y, x: (np.full_like(x, -1.0), np.full_like(x, 2.0))
    flow, mask = _flow_for([(a, same), (b, same)])
    lbl = group_objects(flow, mask, P)
    assert _one_label(lbl, a | b, mask)


def test_nothing_moving_gives_no_objects():
    flow = np.zeros((2, H, W))
    assert group_objects(flow, np.zeros((H, W), bool), P).max() == 0


def _pose(R, t):
    w = math.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q = {"w": w, "x": (R[2, 1] - R[1, 2]) / (4 * w), "y": (R[0, 2] - R[2, 0]) / (4 * w),
         "z": (R[1, 0] - R[0, 1]) / (4 * w)}
    return {"pos": {"q": q, "t": {"x": t[0], "y": t[1], "z": t[2]}}}


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_parts_moving_rigidly_together_are_one_ground_truth_object():
    frames = []
    for k in range(30):
        t = k / 60
        Ra, ta = _rz(0.8 * t), np.array([0.2 * t, 0.0, 1.0])       # object 5: moves + turns
        off = np.array([0.1, 0.05, 0.0])                             # object 6: bolted onto 5
        Rb, tb = Ra, ta + Ra @ off
        Rc, tc = np.eye(3), np.array([0.0, -0.3 * t, 1.2])           # object 7: independent
        frames.append({"ts": t, "id": k, "cam": _pose(np.eye(3), np.zeros(3)),
                       "5": _pose(Ra, ta), "6": _pose(Rb, tb), "7": _pose(Rc, tc)})
    g = rigid_groups({"frames": frames}, 15, [5, 6, 7])
    assert g[5] == g[6]            # attached parts -> one object
    assert g[7] != g[5]            # independent motion -> its own object
