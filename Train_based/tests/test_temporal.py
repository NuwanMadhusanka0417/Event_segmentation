"""Temporal consistency filter (hdems.temporal_filter) and its use in the benchmark."""

import numpy as np
import pytest
import torch

from hdems import protocols as bm
from hdems.data.evimo import EVIMODataset
from hdems.data.motion_labels import MotionParams
from hdems.temporal_filter import TemporalFilter, warp_backward

from test_protocols import H, W, _FixedModel, _fake_sequence


def test_warp_backward_samples_at_the_displaced_position():
    img = torch.zeros(10, 12)
    img[5, 7] = 1.0
    disp = torch.zeros(2, 10, 12)
    disp[0] = 2.0                                   # sample 2 px to the right
    out, valid = warp_backward(img, disp)
    assert out[5, 5] == pytest.approx(1.0) and out[5, 7] == pytest.approx(0.0)
    assert valid[:, :10].min() == pytest.approx(1.0) and valid[:, -1].max() == 0.0   # off the image


def test_static_scene_blends_with_the_previous_frame():
    tf = TemporalFilter(0.5, interval_s=0.0125)
    p1, p2 = torch.full((4, 5), 0.8), torch.full((4, 5), 0.2)
    flow = torch.zeros(2, 4, 5)
    assert torch.equal(tf("s", p1, flow, 1.0), p1)                     # first frame: as is
    assert torch.allclose(tf("s", p2, flow, 1.0 + 1 / 60), torch.full((4, 5), 0.5))


def _moving_object_scene():
    # previous frame: object at columns 4-5, moving 3 px right per interval, so now at
    # 7-8; the flow points 3 px LEFT on the object (towards where it was earlier)
    prev, cur = torch.zeros(6, 14), torch.zeros(6, 14)
    prev[:, 4:6] = 1.0
    cur[:, 7:9] = 0.4                                                # weak evidence now
    f_prev, f_cur = torch.zeros(2, 6, 14), torch.zeros(2, 6, 14)
    f_prev[0, :, 4:6] = -3.0
    f_cur[0, :, 7:9] = -3.0
    return prev, cur, f_prev, f_cur


def test_a_moving_object_is_followed_along_the_flow_without_a_ghost():
    prev, cur, f_prev, f_cur = _moving_object_scene()
    tf = TemporalFilter(0.5, interval_s=0.0125, sigma=1.0)
    tf("s", prev, f_prev, 0.0)
    out = tf("s", cur, f_cur, 0.0125)                                # dt = one interval
    assert torch.allclose(out[:, 7:9], torch.full((6, 2), 0.7))      # reinforced where it IS
    # where it WAS the background is revealed (flow 0 now, -3 then): the ghost is
    # suppressed by exp(-3^2 / 2) -> 0.5 * 0.011 (0.5 without the check)
    assert out[:, 4:6].max() < 0.01


def test_without_the_occlusion_check_a_ghost_is_left_behind():
    prev, cur, f_prev, f_cur = _moving_object_scene()
    tf = TemporalFilter(0.5, interval_s=0.0125, sigma=0.0)
    tf("s", prev, f_prev, 0.0)
    out = tf("s", cur, f_cur, 0.0125)
    assert out[:, 4:6].min() == pytest.approx(0.5)


def test_reset_on_a_gap_new_sequence_or_beta_zero():
    p1, p2 = torch.ones(3, 3), torch.zeros(3, 3)
    tf = TemporalFilter(0.5, max_gap_s=0.05)
    tf("a", p1, None, 0.0)
    assert torch.equal(tf("a", p2, None, 0.2), p2)                   # 200 ms gap
    assert torch.equal(tf("b", p2, None, 0.21), p2)                  # other sequence
    off = TemporalFilter(0.0)
    off("a", p1, None, 0.0)
    assert torch.equal(off("a", p2, None, 0.01), p2)
    with pytest.raises(ValueError):
        TemporalFilter(1.0)


def _ds(tmp_path):
    _fake_sequence(tmp_path)
    return EVIMODataset(tmp_path, "eval", height=H, width=W, window_ms=50.0,
                        motion_params=MotionParams(window_s=0.05))


def _pred():
    pred = np.zeros((H, W), bool)
    pred[0:8, 0:5] = True
    pred[8:16, 10:15] = True
    return pred


def test_benchmark_with_the_filter_runs_every_frame_in_order(tmp_path):
    ds = _ds(tmp_path)
    model = _FixedModel(_pred())
    plain, _ = bm.run_benchmark(model, ds, bm.HUA2025, torch.device("cpu"), windows_ms=[12.5],
                                temporal_beta=0.0)
    filt, stats = bm.run_benchmark(model, ds, bm.HUA2025, torch.device("cpu"), windows_ms=[12.5],
                                   temporal_beta=0.6)
    # a constant prediction is a fixed point of the filter: the score must not change
    assert filt[0].per_sequence()["13-00"] == pytest.approx(plain[0].per_sequence()["13-00"])
    assert stats["forward"] == 13 and stats["temporal_beta"] == 0.6
    report = bm.format_report(bm.HUA2025, filt, stats, gt_mode="moving", pred_source="fg")
    assert "temporal beta=0.6" in report


def test_temporal_beta_comes_from_the_model_by_default(tmp_path):
    ds = _ds(tmp_path)
    model = _FixedModel(_pred())
    model.temporal_beta = 0.3
    _, stats = bm.run_benchmark(model, ds, bm.HUA2025, torch.device("cpu"), windows_ms=[12.5],
                                max_per_sequence=3)
    assert stats["temporal_beta"] == 0.3 and stats["run"] == 3


def test_validation_with_the_filter(tmp_path):
    ds = _ds(tmp_path)
    got, n = bm.validate_event_iou(_FixedModel(_pred()), ds, torch.device("cpu"),
                                   thresholds=[0.5], temporal_beta=0.5)
    assert n == 13 and got[0.5] == pytest.approx(1 / 3)
