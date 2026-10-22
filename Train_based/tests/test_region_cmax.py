"""Per-region contrast maximisation (hdems/region_cmax.py) on synthetic events."""

import numpy as np

from hdems.imu import so3_exp
from hdems.region_cmax import INTERVAL_S, AssignParams, CMaxParams, best_velocity, derotate, segment_iterative, segment_window

T_REF = 1.0
MM = {"fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0}


def _scene(v_obj=(1.6, -0.7), seed=0, window_s=0.05, n_bg=1500, n_obj=250, rate=12):
    """Point features firing events at random times in the window: a static background
    texture and a 30x30 px object moving at v_obj (px per 12.5 ms). -> x, y, t, is_obj."""
    g = np.random.default_rng(seed)
    bg = g.uniform([0, 0], [320, 240], (n_bg, 2))
    obj = g.uniform([140, 100], [170, 130], (n_obj, 2))       # positions at T_REF
    pts = np.concatenate([bg, obj])
    is_obj = np.r_[np.zeros(n_bg, bool), np.ones(n_obj, bool)]
    k = np.repeat(np.arange(len(pts)), rate)
    t = T_REF - g.uniform(0, window_s, len(k))
    vel = np.where(is_obj[k, None], np.asarray(v_obj), 0.0)
    xy = pts[k] - vel * ((T_REF - t) / INTERVAL_S)[:, None] + g.normal(0, 0.2, (len(k), 2))
    # the object hides the background behind it (no background events there at T_REF)
    hidden = ~is_obj[k] & (xy[:, 0] > 138) & (xy[:, 0] < 172) & (xy[:, 1] > 98) & (xy[:, 1] < 132)
    keep = ~hidden
    return xy[keep, 0], xy[keep, 1], t[keep], is_obj[k][keep]


def test_derotate_inverts_camera_rotation():
    g = np.random.default_rng(1)
    omega = np.array([0.4, -0.9, 0.3])                          # rad/s
    xs, ys = g.uniform(0, 320, 200), g.uniform(0, 240, 200)    # where static points are at T_REF
    t = T_REF - g.uniform(0, 0.1, 200)
    X = np.stack([(xs - MM["cx"]) / MM["fx"], (ys - MM["cy"]) / MM["fy"], np.ones(200)], 1)
    Xe = np.stack([so3_exp(omega * (T_REF - ti)) @ Xi for ti, Xi in zip(t, X)])   # bearing at t
    xe = MM["fx"] * Xe[:, 0] / Xe[:, 2] + MM["cx"]
    ye = MM["fy"] * Xe[:, 1] / Xe[:, 2] + MM["cy"]
    assert np.abs(xe - xs).max() > 1.0                          # the rotation moved them
    xr, yr = derotate(xe, ye, t, T_REF, omega, MM)
    assert np.abs(xr - xs).max() < 1e-6 and np.abs(yr - ys).max() < 1e-6


def test_best_velocity_is_subpixel():
    x, y, t, obj = _scene(v_obj=(1.6, -0.7))
    p = CMaxParams()
    sel = obj
    dtau = (T_REF - t[sel]) / INTERVAL_S
    box = (130, 90, 50, 50)
    m = np.zeros((50, 50), bool)
    m[8:42, 8:42] = True
    v, _s = best_velocity(x[sel], y[sel], dtau, box, m, p)
    assert np.abs(v - [1.6, -0.7]).max() < 0.1


def test_slow_motion_is_resolved():
    """0.3 px per 12.5 ms -- the slow movers the pooled flows under-read."""
    x, y, t, obj = _scene(v_obj=(0.3, 0.0), seed=2)
    dtau = (T_REF - t[obj]) / INTERVAL_S
    m = np.zeros((50, 50), bool)
    m[8:42, 8:42] = True
    v, _s = best_velocity(x[obj], y[obj], dtau, (130, 90, 50, 50), m, CMaxParams())
    assert abs(v[0] - 0.3) < 0.1 and abs(v[1]) < 0.1


def test_segment_window_separates_mover_from_background():
    x, y, t, obj = _scene(v_obj=(1.6, -0.7), seed=3)
    o = segment_window(x, y, t, T_REF, CMaxParams(n_regions=20))
    sc = o["score_idx"]
    lab = obj[sc]
    speed, own = o["speed"], o["own"]
    assert np.median(speed[lab]) > 1.0
    assert np.median(speed[~lab]) < 0.2
    # region decision + the per-event check classify most events right
    pred = (speed > 0.5) & (own > 0.5)
    iou = (pred & lab).sum() / (pred | lab).sum()
    assert iou > 0.7


def test_empty_window_is_safe():
    o = segment_window(np.zeros(0), np.zeros(0), np.zeros(0), T_REF)
    assert len(o["speed"]) == 0


def test_reassignment_fixes_mixed_regions():
    """Few large regions -> the mover is mixed with background in its region; the iterative
    per-event reassignment must still cut it out and measure its motion."""
    x, y, t, obj = _scene(v_obj=(1.6, -0.7), seed=4)
    p = CMaxParams(n_regions=4)
    lab = obj[segment_window(x, y, t, T_REF, p)["score_idx"]]
    o0 = segment_window(x, y, t, T_REF, p)
    o = segment_iterative(x, y, t, T_REF, p, AssignParams())

    def iou(pred):
        return (pred & lab).sum() / max((pred | lab).sum(), 1)

    before = max(iou(o0["speed"] > th) for th in (0.25, 0.5, 1.0))
    after = iou(o["speed"] > 0.5)
    assert after > 0.7 and after > before + 0.1
    vm = o["v"][np.bincount(o["region"][lab]).argmax()]
    assert np.abs(vm - [1.6, -0.7]).max() < 0.15
