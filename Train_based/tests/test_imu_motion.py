"""IMU rotation compensation, the quadratic / IMU-anchored ego fit, fine flow, the VSA
ego-consistency maps, event-IoU validation and old-checkpoint compatibility."""

import json

import numpy as np
import pytest
import torch

from hdems.config import _LEGACY_FRONTEND, frontend_settings, restore_frontend
from hdems.data.evimo import EVIMODataset, flip_sample
from hdems.data.motion_labels import MotionParams
from hdems.imu import RotationSource, procrustes, rotation_flow, so3_exp, so3_log
from hdems.models.hdems import HDEMS
from hdems.models.motion import ego_residual_joint
from hdems.vsa.velocity import MOTION_SCALARS

from test_mfunet import _tiny
from test_protocols import H, W, _FixedModel, _fake_sequence

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
    for (x, y) in ((100.0, 50.0), (500.0, 400.0), (320.0, 240.0)):     # pixel centres
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
    # working pixel (i, j) covers sensor rows 2i, 2i+1 -> its centre is between them
    centre = 0.5 * (full[:, 1::2, 1::2] + full[:, ::2, ::2])
    assert np.allclose(half, centre / 2, atol=2e-3)


def test_flip_negates_the_rotation_field():
    rot = torch.randn(2, 4, 6)
    s = {"mask": torch.zeros(4, 6), "rot_flow": rot}
    h = flip_sample(s, (-1,))["rot_flow"]
    assert torch.equal(h[0], -torch.flip(rot[0], (-1,))) and torch.equal(h[1], torch.flip(rot[1], (-1,)))
    b = flip_sample(s, (-2, -1))["rot_flow"]
    assert torch.equal(b, -torch.flip(rot, (-2, -1)))


def test_procrustes_recovers_a_rotation():
    rng = np.random.default_rng(0)
    R = so3_exp(np.array([0.1, 1.4, -0.3]))
    a = rng.normal(size=(200, 3))
    assert np.allclose(procrustes(a, a @ R.T), R, atol=1e-9)


def _scene(big_mover: bool):
    """Background flow = smooth quadratic; a mover covering 60% of the pixels."""
    Hh, Ww = 30, 40
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, Hh), torch.linspace(-1, 1, Ww), indexing="ij")
    bg = torch.stack([0.05 + 0.1 * xs * xs, -0.05 + 0.08 * xs * ys])
    flow = bg.clone()
    mover = torch.zeros(Hh, Ww, dtype=torch.bool)
    if big_mover:
        mover[:, : int(0.6 * Ww)] = True
        flow[0][mover] += 3.0
    return flow[None], mover


def test_prior_keeps_a_large_mover_out_of_the_background_fit():
    flow, mover = _scene(big_mover=True)
    plain, _ = ego_residual_joint(flow, model="quadratic")
    anchored, bg = ego_residual_joint(flow, model="quadratic", prior_px=0.5)
    # anchored: background explained, mover left in the residual
    assert anchored[0][:, ~mover].abs().max() < 0.05
    assert anchored[0, 0][mover].mean() > 2.9
    assert torch.allclose(bg + anchored, flow)
    # started from plain least squares, the mover (60% of the pixels) bends the fit
    assert plain[0][:, ~mover].abs().max() > 0.5


def test_quadratic_fits_rotation_like_flow_exactly():
    flow, _ = _scene(big_mover=False)
    res, _ = ego_residual_joint(flow, model="quadratic")
    assert res.abs().max() < 1e-3


def _new_cfg(**motion):
    cfg = _tiny("mfunet")
    cfg["motion"] = {"rot_source": "imu", "ego_model": "quadratic", "ego_prior_px": 1.0,
                     "fine_flow": True, "fine_alpha": 0.6, "fine_smooth": 3,
                     "extra_inputs": True, "vsa_consistency": True, **motion}
    return cfg


def test_new_motion_pipeline_forward_shapes_and_ablation():
    model = HDEMS(_new_cfg()).eval()
    assert model.n_motion_ch == MOTION_SCALARS + 3 + 3 + 3
    g = torch.Generator().manual_seed(4)
    s = (torch.rand(1, 4, 2, 16, 20, generator=g) > 0.7).float()
    rot = torch.randn(1, 2, 16, 20, generator=g) * 0.3
    with torch.no_grad():
        assert model.compute_motion_all(s).shape == (1, 5, 16, 20)       # + fine flow
        m = model.motion_features(s, rot)
        assert m["consistency"].shape == (1, 3, 16, 20)
        assert m["consistency"].abs().max() <= 1.0 + 1e-4
        assert torch.allclose(m["background"] + m["residual"], m["flow"], atol=1e-5)
        ch = model.motion_channels(m, s)
        assert ch.shape == (1, model.n_motion_ch, 16, 20)
        out = model(s, task="segmentation", rot_flow=rot)
        assert out["seg_logits"].shape == (1, 2, 16, 20)
        model.ablate = "motion"
        m0 = model.motion_features(s, rot)
        assert model.motion_channels(m0, s)[:, MOTION_SCALARS:].abs().max() == 0
        assert torch.equal(m0["flow"], m["flow"])                 # measured flow kept for output
        model.ablate = None


def test_without_imu_the_fit_is_not_anchored():
    model = HDEMS(_new_cfg(rot_source="none")).eval()
    s = (torch.rand(1, 4, 2, 16, 20, generator=torch.Generator().manual_seed(5)) > 0.7).float()
    with torch.no_grad():
        a = model.motion_features(s, torch.randn(1, 2, 16, 20))   # ignored: rot_source none
        b = model.motion_features(s, None)
    assert torch.equal(a["residual"], b["residual"]) and a["rot"] is None


def test_old_checkpoints_rebuild_the_old_pipeline():
    saved = frontend_settings(_tiny("mfunet"))
    saved.pop("motion", None)                                   # trained before the section
    cfg = restore_frontend(_new_cfg(), saved)
    model = HDEMS(cfg)
    assert model.n_motion_ch == MOTION_SCALARS and model.rot_source == "none"
    assert model.ego_model == "affine" and not model.fine_flow and not model.vsa_consistency
    assert ("motion", "rot_source") in _LEGACY_FRONTEND


def test_fine_flow_has_its_own_cache_key(tmp_path):
    from hdems.flow_cache import FlowCache
    dirs = []
    for fine in (False, True):
        cfg = _new_cfg(fine_flow=fine)
        cfg["flow_cache"] = {"enabled": True, "dir": str(tmp_path)}
        model = HDEMS(cfg)
        model.flow_cache.bind(model)
        dirs.append(model.flow_cache._dir)
    assert dirs[0] != dirs[1]


def _with_imu(seq, rate):
    """Add a constant-rate gyro (identity calibration) to a fake sequence."""
    info = np.load(seq / "dataset_info.npz", allow_pickle=True)["meta"].item()
    ts = np.arange(0.0, 1.0, 0.001)
    info["imu"] = {"/prophesee/left/imu": [
        {"ts": float(t), "angular_velocity": dict(zip("xyz", rate)),
         "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.8}} for t in ts]}
    np.savez(seq / "dataset_info.npz", meta=np.array(info, dtype=object))
    return info


def test_dataset_provides_the_rotation_field(tmp_path):
    seq = _fake_sequence(tmp_path)
    rate = (0.2, -0.5, 0.1)
    info = _with_imu(seq, rate)
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"left": {"R": np.eye(3).tolist(), "dt": 0.0}}))
    ds = EVIMODataset(tmp_path, "eval", height=H, width=W, window_ms=50.0,
                      time_frames=[1.0, 0.75, 0.5, 0.0], motion_params=MotionParams(window_s=0.05),
                      rotation=RotationSource("imu", calib_path=calib), res_ratio=1)
    s = ds[0]
    mm = info["meta"]
    expect = rotation_flow(np.array(rate), mm["fx"], mm["fy"], mm["cx"], mm["cy"], 0.0125, H, W, 1)
    assert s["rot_flow"].shape == (2, H, W)
    assert np.allclose(s["rot_flow"].numpy(), expect, atol=1e-5)


def test_validate_event_iou_threshold_sweep(tmp_path):
    from hdems import protocols as bm
    _fake_sequence(tmp_path)
    ds = EVIMODataset(tmp_path, "eval", height=H, width=W, window_ms=50.0,
                      motion_params=MotionParams(window_s=0.05))
    pred = np.zeros((H, W), bool)
    pred[0:8, 0:5] = True
    pred[8:16, 10:15] = True
    got, n = bm.validate_event_iou(_FixedModel(pred), ds, torch.device("cpu"),
                                   thresholds=[0.5, 0.99])
    assert n == 13
    assert got[0.5] == pytest.approx(1 / 3)         # same as the benchmark test
    assert got[0.99] == pytest.approx(0.0)          # P(moving) = e/(1+e) = 0.73 < 0.99
