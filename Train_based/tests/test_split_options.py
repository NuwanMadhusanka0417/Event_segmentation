"""Held-out takes (val_sequences), IMU shift / gain, and the shortcut-ablation options
(hv_input none, density_input false)."""

import numpy as np
import pytest
import torch

from hdems.config import frontend_settings, restore_frontend
from hdems.data.build import HOLDOUT, build_dataset
from hdems.data.evimo2_reader import build_sample_index
from hdems.imu import RotationSource
from hdems.models.hdems import HDEMS

from test_imu_motion import _new_cfg
from test_mfunet import _tiny
from test_protocols import H, W, _fake_sequence

A, B = "scene13_dyn_test_01_000000", "scene14_dyn_test_02_000000"


def test_sequences_can_be_selected_or_excluded(tmp_path):
    _fake_sequence(tmp_path, A)
    _fake_sequence(tmp_path, B)
    names = lambda idx: sorted({p.name for p, _ in idx})            # noqa: E731
    assert names(build_sample_index(tmp_path, "eval", window_s=0.05)) == [A, B]
    assert names(build_sample_index(tmp_path, "eval", window_s=0.05,
                                    only_sequences=["scene14_dyn_test_02"])) == [B]
    assert names(build_sample_index(tmp_path, "eval", window_s=0.05,
                                    exclude_sequences=["scene14_dyn_test_02"])) == [A]


def _cfg(root, **ds):
    return {"dataset": {"name": "evimo", "root": str(root), "split": "eval", "eval_split": "test",
                        "height": H, "width": W, "window_ms": 50.0, "val_split": "holdout",
                        "val_scenes": ["scene13"], **ds},
            "motion": {"rot_source": "none"}}


def test_held_out_takes_replace_held_out_scenes(tmp_path):
    _fake_sequence(tmp_path, A)
    _fake_sequence(tmp_path, B)
    # val_sequences: B is validation and never trained on; A (same scene13 as val_scenes!) trains
    cfg = _cfg(tmp_path, val_sequences=["scene14_dyn_test_02"])
    train = {p.name for p, _ in build_dataset(cfg, "eval").index}
    val = {p.name for p, _ in build_dataset(cfg, HOLDOUT).index}
    assert train == {A} and val == {B}
    # without val_sequences the old behaviour: the whole val scene is held out
    cfg = _cfg(tmp_path)
    assert {p.name for p, _ in build_dataset(cfg, "eval").index} == {B}
    assert {p.name for p, _ in build_dataset(cfg, HOLDOUT).index} == {A}


def test_holdout_needs_scenes_or_sequences(tmp_path):
    _fake_sequence(tmp_path, A)
    with pytest.raises(ValueError):
        build_dataset(_cfg(tmp_path, val_scenes=[]), HOLDOUT)


def _meta_with_step():
    """Gyro rate 0 before t = 0.5 s and (1, 2, 3) rad/s after (identity calibration)."""
    ts = np.arange(0.0, 1.0, 0.001)
    rate = [{"x": 0.0, "y": 0.0, "z": 0.0} if t < 0.5 else {"x": 1.0, "y": 2.0, "z": 3.0} for t in ts]
    return {"imu": {"/prophesee/left/imu": [
        {"ts": float(t), "angular_velocity": r, "linear_acceleration": {"x": 0, "y": 0, "z": 0}}
        for t, r in zip(ts, rate)]}}


def test_imu_shift_moves_the_window_and_gain_scales(tmp_path):
    import json
    calib = tmp_path / "c.json"
    calib.write_text(json.dumps({"left": {"R": np.eye(3).tolist(), "dt": 0.0}}))
    meta = _meta_with_step()
    plain = RotationSource("imu", calib_path=calib, avg_ms=10.0)
    assert np.allclose(plain.omega("s", meta, 0.52), [1, 2, 3])        # window 0.51-0.52
    late = RotationSource("imu", calib_path=calib, avg_ms=10.0, shift_ms=-30.0)
    assert np.allclose(late.omega("s", meta, 0.52), [0, 0, 0])         # window 0.48-0.49
    gain = RotationSource("imu", calib_path=calib, avg_ms=10.0, gain=1.5)
    assert np.allclose(gain.omega("s", meta, 0.52), [1.5, 3.0, 4.5])


def test_sequences_without_a_gyro_fall_back_to_motion_capture(tmp_path):
    import json
    from hdems.imu import vicon_rates
    calib = tmp_path / "c.json"
    calib.write_text(json.dumps({"left": {"R": np.eye(3).tolist(), "dt": 0.0}}))
    # a camera turning about y at 0.5 rad/s, no IMU recording at all
    from hdems.imu import so3_exp
    traj = []
    for t in np.arange(0.0, 1.0, 0.005):
        R = so3_exp(np.array([0.0, 0.5 * t, 0.0]))
        w, x, y, z = _quat_from_matrix(R)
        traj.append({"ts": float(t), "cam": {"pos": {"q": {"w": w, "x": x, "y": y, "z": z},
                                                     "t": {"x": 0.0, "y": 0.0, "z": 0.0}}}})
    meta = {"full_trajectory": traj, "frames": traj}
    assert np.allclose(vicon_rates(meta)[1].mean(0), [0, 0.5, 0], atol=1e-3)
    off = RotationSource("imu", calib_path=calib, fallback="none")
    assert off.omega("seq_no_imu", meta, 0.5) is None
    fb = RotationSource("imu", calib_path=calib, fallback="vicon")
    assert np.allclose(fb.omega("seq_no_imu2", meta, 0.5), [0, 0.5, 0], atol=1e-3)
    assert "seq_no_imu2" in fb.no_imu


def _quat_from_matrix(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)


def test_the_test_split_never_gets_the_fallback(tmp_path):
    from hdems.data.build import rotation_source
    import json
    calib = tmp_path / "c.json"
    calib.write_text(json.dumps({"left": {"R": np.eye(3).tolist(), "dt": 0.0}}))
    cfg = {"motion": {"rot_source": "imu", "imu_calib": str(calib), "rot_fallback": "vicon"}}
    assert rotation_source(cfg, allow_fallback=True).fallback == "vicon"
    assert rotation_source(cfg, allow_fallback=False).fallback == "none"


def test_no_hypervector_and_no_density_options():
    model = HDEMS(_new_cfg(density_input=False, vsa_consistency=False, fine_flow=False)
                  | {"real_fpe": {"input": "none"}}).eval()
    s = (torch.rand(1, 4, 2, 16, 20, generator=torch.Generator().manual_seed(6)) > 0.7).float()
    with torch.no_grad():
        assert model.pixel_hv(torch.randn(1, 2, 16, 20)).abs().max() == 0
        conf = torch.rand(1, 1, 16, 20)
        ev = model.evidence_channels(s, conf)
        assert ev[:, 0].abs().max() == 0 and torch.equal(ev[:, 1], conf[:, 0])
        assert model(s, task="segmentation")["seg_logits"].shape == (1, 2, 16, 20)


def test_checkpoints_without_the_new_keys_keep_the_old_behaviour():
    saved = frontend_settings(_new_cfg())
    for k in ("imu_shift_ms", "imu_gain", "density_input"):
        saved["motion"].pop(k, None)
    cfg = restore_frontend(_new_cfg(density_input=False, imu_gain=1.3), saved)
    model = HDEMS(cfg)
    assert model.density_input is True
    assert cfg["motion"]["imu_gain"] == 1.0 and cfg["motion"]["imu_shift_ms"] == 0.0
