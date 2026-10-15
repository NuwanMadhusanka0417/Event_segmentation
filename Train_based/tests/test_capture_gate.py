"""Capture fallback (motion.ego_gate_px): a frame whose ego fit modelled a large mover
keeps only the rotation as background."""

import torch

from hdems.config import frontend_settings, restore_frontend
from hdems.models.hdems import HDEMS

from test_imu_motion import _new_cfg

H, W = 16, 20


def _frame(mover_share_cols: int, speed: float = 1.5):
    """Everything fires; derotated flow is 0 on the background and `speed` px on a
    mover covering the first `mover_share_cols` columns."""
    s = torch.ones(1, 4, 2, H, W)
    flow = torch.zeros(1, 2, H, W)
    flow[0, 0, :, :mover_share_cols] = speed
    return s, flow


def test_a_captured_fit_falls_back_to_the_rotation_only():
    model = HDEMS(_new_cfg(ego_gate_px=0.5, ego_prior_px=0.0)).eval()   # plain start: easy to capture
    rot = torch.full((1, 2, H, W), 0.3)
    s, d = _frame(14)                                  # the mover is 70% of the frame
    flow = d + rot
    res, bg = model.background_fit(flow, s, rot)
    assert torch.allclose(bg, rot)                     # fit dropped: background = rotation
    assert torch.allclose(res, d)                      # residual = derotated flow
    off = HDEMS(_new_cfg(ego_gate_px=0.0, ego_prior_px=0.0)).eval()
    res_off, _ = off.background_fit(flow, s, rot)
    assert res_off[0, 0, :, :14].abs().mean() < 0.2    # without it: the mover looks static


def test_a_normal_frame_keeps_its_fit():
    on = HDEMS(_new_cfg(ego_gate_px=0.5)).eval()
    off = HDEMS(_new_cfg(ego_gate_px=0.0)).eval()
    rot = torch.full((1, 2, H, W), 0.3)
    s, d = _frame(4)                                   # small mover: the fallback must not fire
    res_on, bg_on = on.background_fit(d + rot, s, rot)
    res_off, bg_off = off.background_fit(d + rot, s, rot)
    assert torch.equal(res_on, res_off) and torch.equal(bg_on, bg_off)
    assert not torch.allclose(bg_on, rot)


def test_old_checkpoints_have_no_fallback():
    saved = frontend_settings(_new_cfg())
    saved["motion"].pop("ego_gate_px", None)
    assert HDEMS(restore_frontend(_new_cfg(ego_gate_px=0.5), saved)).ego_gate_px == 0.0
