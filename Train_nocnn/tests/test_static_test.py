"""Static-scene test (hdems/static_test.py) on synthetic scenes with a known answer:
a camera that rotates (removed with the 'IMU') and translates over a scene of random
depth, plus one independently moving patch."""

import numpy as np
import pytest

from hdems.imu import rotation_flow
from hdems.static_test import StaticTestParams, derotate, moving_score, params_from_config

H, W, R = 120, 160, 4                      # working grid of a 480x640 sensor at ratio 4
K = (520.0, 520.0, 324.0, 239.0)


def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    ranks = np.empty(allv.size)
    ranks[allv.argsort(kind="mergesort")] = np.arange(1, allv.size + 1)
    return (ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def scene(T, mover_v, omega=(0.0, 0.0, 0.0), seed=0):
    """Flow (2,H,W) = earlier position - now (working px) of a translating (T, metres per
    interval) and rotating (omega, rad/s) camera over random depth, plus a mover."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    qx = ((xx + 0.5) * R - 0.5 - K[2]) / K[0]
    qy = ((yy + 0.5) * R - 0.5 - K[3]) / K[1]
    Z = rng.uniform(0.4, 2.0, (H, W))
    X, Y = qx * Z + T[0], qy * Z + T[1]
    Ze = Z + T[2]
    d = np.stack([(X / Ze - qx) * K[0] / R, (Y / Ze - qy) * K[1] / R])
    rot = rotation_flow(np.asarray(omega), *K, 0.0125, H, W, R).astype(np.float64)
    mv = np.zeros((H, W), bool)
    mv[40:65, 100:130] = True
    d[:, mv] += np.asarray(mover_v, dtype=float)[:, None]
    d = d + rot + rng.normal(0, 0.03, d.shape)
    ev = rng.random((H, W)) < 0.4
    return d, rot, ev, mv


CASES = {
    "sideways+forward, mover perpendicular": ((0.004, 0.0, 0.002), (0.0, 0.6)),
    "forward, mover sideways": ((0.0, 0.0, 0.006), (0.4, 0.0)),
    "sideways, mover against the scene": ((0.005, 0.001, 0.0), (-0.8, 0.0)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_epipolar_finds_the_mover_where_flow_magnitude_cannot(name):
    T, mover = CASES[name]
    d, rot, ev, mv = scene(T, mover, omega=(0.3, -0.5, 0.2))
    p = StaticTestParams(model="epipolar", static_px=0.06, tau=0.08, min_rot=0.0)
    score, derot, info = moving_score(d, ev, p, rot_flow=rot, intrinsics=K, ratio=R)
    assert info["derotated"] and info["mode"] == "epipolar"
    assert auc(score[ev & mv], score[ev & ~mv]) > 0.95
    mag = np.hypot(*derot)
    assert auc(mag[ev & mv], mag[ev & ~mv]) < 0.9          # magnitude alone is not enough


def test_rotation_is_removed_exactly():
    d, rot, ev, mv = scene((0.0, 0.0, 0.0), (0.5, 0.0), omega=(0.4, 0.6, -0.3))
    flat, used = derotate(d, rot, min_rot=0.0)
    assert used
    assert np.median(np.hypot(*flat)[ev & ~mv]) < 0.06     # static scene ~0 after derotation
    assert np.median(np.hypot(*d)[ev & ~mv]) > 0.3          # ... but not before


def test_small_rotation_is_left_in():
    d, rot, ev, _ = scene((0.0, 0.0, 0.0), (0.0, 0.0), omega=(0.01, 0.0, 0.0))
    _, used = derotate(d, rot, min_rot=0.2)
    assert not used


def test_still_camera_scores_the_flow_magnitude():
    d, rot, ev, mv = scene((0.0, 0.0, 0.0), (0.4, 0.2))
    p = StaticTestParams(model="epipolar", static_px=0.1)
    score, _, info = moving_score(d, ev, p, rot_flow=rot, intrinsics=K, ratio=R)
    assert info["mode"] == "static"
    assert auc(score[ev & mv], score[ev & ~mv]) > 0.99


def test_quadratic_fits_a_plane_seen_by_a_translating_camera():
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:H, 0:W]
    x, y = (xx - W / 2) / W, (yy - H / 2) / H
    d = np.stack([0.3 + 0.4 * x + 0.2 * x * x - 0.1 * x * y, -0.2 + 0.3 * y + 0.15 * y * y])
    mv = np.zeros((H, W), bool)
    mv[10:60, 10:90] = True                               # a LARGE mover (21% of the frame)
    d[:, mv] += np.array([0.0, 0.8])[:, None]
    d += rng.normal(0, 0.02, d.shape)
    ev = rng.random((H, W)) < 0.5
    score, _, info = moving_score(d, ev, StaticTestParams(model="quadratic", static_px=0.05))
    assert info["mode"] == "quadratic"
    assert auc(score[ev & mv], score[ev & ~mv]) > 0.99


def test_params_are_scaled_to_the_working_resolution():
    cfg = {"dataset": {"resolution_ratio": 2},
           "moving_test": {"model": "quadratic", "threshold_px": 1.0, "min_rot_px": 0.5}}
    p = params_from_config(cfg)
    assert (p.model, p.threshold, p.min_rot) == ("quadratic", 0.5, 0.25)
    with pytest.raises(ValueError):
        params_from_config({"moving_test": {"model": "bogus"}})
