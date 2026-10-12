"""IMU rotation compensation (hdems/imu.py; ported from the MS_paper1 branch)."""

import numpy as np
import pytest
import torch

from hdems.data.evimo import flip_sample
from hdems.imu import procrustes, rotation_flow, so3_exp, so3_log

FX = FY = 500.0
CX, CY = 319.5, 239.5


def test_so3_round_trip():
    v = np.array([0.3, -0.2, 0.5])
    assert np.allclose(so3_log(so3_exp(v)), v, atol=1e-9)
    assert np.allclose(so3_exp(np.zeros(3)), np.eye(3))


def test_rotation_flow_matches_projecting_a_rotated_point():
    w = np.array([0.4, -1.1, 0.7])                          # rad/s, camera frame
    dt = 0.0125
    flow = rotation_flow(w, FX, FY, CX, CY, dt, 480, 640, ratio=1)
    for (x, y) in ((100.0, 50.0), (500.0, 400.0), (320.0, 240.0)):
        X = np.array([(x - CX) / FX, (y - CY) / FY, 1.0]) * 2.7      # any depth
        Xe = so3_exp(w * dt) @ X                            # where it was dt EARLIER
        u, v = FX * Xe[0] / Xe[2] + CX - x, FY * Xe[1] / Xe[2] + CY - y
        assert flow[0, int(y), int(x)] == pytest.approx(u, abs=1e-3)
        assert flow[1, int(y), int(x)] == pytest.approx(v, abs=1e-3)
    assert np.abs(rotation_flow(np.zeros(3), FX, FY, CX, CY, dt, 48, 64)).max() == 0


def test_rotation_flow_at_half_resolution_is_the_block_centre_value_halved():
    w = np.array([0.2, 0.9, -0.3])
    full = rotation_flow(w, FX, FY, CX, CY, 0.0125, 480, 640, ratio=1)
    half = rotation_flow(w, FX, FY, CX, CY, 0.0125, 240, 320, ratio=2)
    centre = 0.5 * (full[:, 1::2, 1::2] + full[:, ::2, ::2])
    assert np.allclose(half, centre / 2, atol=2e-3)


def test_flip_negates_the_rotation_field_and_mirrors_the_principal_point():
    rot = torch.randn(2, 4, 6)
    s = {"mask": torch.zeros(4, 6), "rot_flow": rot,
         "intrinsics": torch.tensor([500.0, 500.0, 1.0, 1.0, 6.0, 4.0])}
    h = flip_sample(s, (-1,))
    assert torch.equal(h["rot_flow"][0], -torch.flip(rot[0], (-1,)))
    assert torch.equal(h["rot_flow"][1], torch.flip(rot[1], (-1,)))
    assert h["intrinsics"].tolist() == [500.0, 500.0, 4.0, 1.0, 6.0, 4.0]
    b = flip_sample(s, (-2, -1))
    assert torch.equal(b["rot_flow"], -torch.flip(rot, (-2, -1)))
    assert b["intrinsics"].tolist() == [500.0, 500.0, 4.0, 2.0, 6.0, 4.0]


def test_procrustes_recovers_a_rotation():
    rng = np.random.default_rng(0)
    R = so3_exp(np.array([0.1, 1.4, -0.3]))
    a = rng.normal(size=(200, 3))
    assert np.allclose(procrustes(a, a @ R.T), R, atol=1e-9)
