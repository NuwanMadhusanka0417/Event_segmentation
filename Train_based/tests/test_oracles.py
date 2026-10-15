"""Phase-1 diagnostics: ground-truth (oracle) flow / background inputs and the
training-free cue baselines."""

import numpy as np
import pytest
import torch

from hdems import protocols as bm
from hdems.baselines import CueBaseline
from hdems.data.build import oracle_inputs
from hdems.data.evimo import EVIMODataset, flip_sample
from hdems.data.evimo2_reader import gt_key
from hdems.data.gt_flow import gt_flow_between, gt_static_flow_between, to_working
from hdems.data.motion_labels import MotionParams
from hdems.imu import so3_exp
from hdems.models.hdems import HDEMS

from test_imu_motion import _new_cfg
from test_protocols import FX, H, W, _fake_sequence
from test_split_options import _quat_from_matrix

TF = [1.0, 0.75, 0.5, 0.0]


def _with_depth(seq, n_frames=13):
    """Depth 1 m on both objects, none elsewhere (as EVIMO2: depth only on tracked objects)."""
    m = np.load(seq / "dataset_mask.npz")
    depth = {gt_key("depth", i): np.where(m[gt_key("mask", i)] > 0, 1000, 0).astype(np.uint16)
             for i in range(n_frames)}
    np.savez_compressed(seq / "dataset_depth.npz", **depth)


def _ds(root, **kw):
    return EVIMODataset(root, "eval", height=H, width=W, window_ms=50.0, time_frames=TF,
                        motion_params=MotionParams(window_s=0.05), **kw)


def test_dataset_attaches_ground_truth_flow_and_background(tmp_path):
    seq = _fake_sequence(tmp_path)
    _with_depth(seq)
    s = _ds(tmp_path, oracle_inputs=("gt_flow", "gt_background"))[5]
    g, b = s["gt_flow"].numpy(), s["gt_background"].numpy()
    assert g.shape == b.shape == (2, H, W)
    # object 1 moves +1 m/s in x at 1 m: 100 px/s -> -1.25 px over 12.5 ms toward EARLIER times
    assert np.allclose(g[0, 0:8, 0:10], -1.25, atol=1e-4) and np.allclose(g[1, 0:8, 0:10], 0, atol=1e-6)
    assert np.allclose(g[:, 8:16, 10:20], 0, atol=1e-6)                 # static object 2
    assert np.allclose(b[:, 0:8, 0:10], 0) and np.allclose(b[:, 8:16, 10:20], 0)   # still camera
    assert np.isnan(g[:, 0:8, 10:20]).all() and np.isnan(b[:, 0:8, 10:20]).all()  # no depth there
    # a flip mirrors the vector field as well as the image
    f = flip_sample(s, (-1,))
    assert torch.allclose(f["gt_flow"][0, 0:8, -10:], torch.tensor(1.25))


def _pose_dict(R, t):
    w, x, y, z = _quat_from_matrix(R)
    return {"pos": {"q": {"w": w, "x": x, "y": y, "z": z},
                    "t": dict(zip("xyz", map(float, t)))}}


def test_static_world_flow_matches_a_static_object_under_camera_motion():
    """Camera translating AND rotating: a world-static object's GT flow (from its own
    camera-frame poses) must equal the static-world flow from the camera poses alone;
    a moving object's must not."""
    frames = []
    for ts in (0.0, 1 / 60):
        Rc = so3_exp(np.array([0.0, 0.8 * ts, 0.1 * ts]))           # camera -> world
        tc = np.array([0.5 * ts, 0.0, 0.2 * ts])
        f = {"ts": ts, "cam": _pose_dict(Rc, tc)}
        for oid, tw in ((1, np.array([0.1, 0.0, 2.0])),                  # static in the world
                        (2, np.array([-0.2 + 2.0 * ts, 0.0, 1.5]))):     # moving 2 m/s in x
            f[str(oid)] = _pose_dict(Rc.T, Rc.T @ (tw - tc))             # pose in the camera frame
        frames.append(f)
    meta = {"frames": frames, "meta": {"fx": FX, "fy": FX, "cx": W / 2, "cy": H / 2}}
    obj = np.zeros((H, W), np.int32)
    obj[:, :10] = 1000
    obj[:, 10:] = 2000
    Z = np.where(obj == 1000, 2000, 1500).astype(np.uint16)

    class _Npz(dict):
        @property
        def files(self):
            return list(self)
    depth, masks = _Npz({gt_key("depth", 0): Z}), _Npz({gt_key("mask", 0): obj})
    g = gt_flow_between(meta, depth, masks, 0, 1)
    b = gt_static_flow_between(meta, depth, 0, 1)
    assert np.isfinite(b).all()
    assert np.allclose(g[:, :, :10], b[:, :, :10], atol=1e-6)       # static object
    assert np.abs(g[0, :, 10:] - b[0, :, 10:]).min() > 1.0          # the mover's own motion


def test_to_working_averages_known_values_per_block():
    g = np.full((2, 4, 4), np.nan)
    g[:, 0, 0] = 2.0                                  # block (0, 0): one known value
    g[:, 2:4, 2:4] = [[[4.0, 8.0], [4.0, 8.0]]]       # block (1, 1): mean 6
    w = to_working(g, 2, 2, 2)
    assert w.shape == (2, 2, 2)
    assert np.allclose(w[:, 0, 0], 1.0) and np.allclose(w[:, 1, 1], 3.0)   # / ratio
    assert np.isnan(w[:, 0, 1]).all()


def test_build_requests_the_oracle_fields_only_when_asked():
    assert oracle_inputs({"motion": {}}) == ()
    assert oracle_inputs({"motion": {"flow_source": "gt", "background_source": "gt"}}) == \
        ("gt_flow", "gt_background")


def _surfaces(seed=1):
    return (torch.rand(1, 4, 2, 16, 20, generator=torch.Generator().manual_seed(seed)) > 0.7).float()


def test_model_uses_ground_truth_where_known_and_its_own_estimate_elsewhere():
    s = _surfaces()
    gt = torch.full((1, 2, 16, 20), float("nan"))
    gt[:, 0, :8] = 2.0
    gt[:, 1, :8] = -1.0
    plain = HDEMS(_new_cfg(rot_source="none")).eval()
    oracle = HDEMS(_new_cfg(rot_source="none", flow_source="gt", background_source="gt")).eval()
    oracle.load_state_dict(plain.state_dict())
    with torch.no_grad():
        a = plain.motion_features(s)
        b = oracle.motion_features(s, gt_flow=gt, gt_background=gt * 0.5)
        assert torch.allclose(b["flow"][:, :, :8], gt[:, :, :8])
        assert torch.equal(b["flow"][:, :, 8:], a["flow"][:, :, 8:])
        # background = the true static-world flow where known -> residual = flow - it
        assert torch.allclose(b["residual"][:, :, :8], gt[:, :, :8] * 0.5)
        assert torch.isfinite(b["residual"]).all() and torch.isfinite(b["fine_residual"]).all()
        out = oracle(s, task="segmentation", gt_flow=gt, gt_background=gt)
        assert torch.isfinite(out["seg_logits"]).all()
        with pytest.raises(ValueError):
            oracle.motion_features(s)                    # oracle asked for, no field given


def test_cue_baseline_scores_through_the_benchmark(tmp_path):
    _fake_sequence(tmp_path)
    ds = _ds(tmp_path)
    for cue in ("residual", "derotated", "foe"):
        base = CueBaseline(HDEMS(_new_cfg(rot_source="none")).eval(), cue, threshold=0.5)
        out, _surf, _aux = bm.forward_sample(base, ds[5], torch.device("cpu"))
        assert out["moving_score"].shape == (1, 1, H, W)
        assert torch.equal(bm.moving_prob(out), out["moving_score"][0, 0])
        scores, stats = bm.run_benchmark(base, ds, bm.HUA2025, torch.device("cpu"),
                                         windows_ms=[12.5], max_per_sequence=3, log_every=0)
        assert stats["run"] == 3 and "13-00" in scores[0].per_sequence()
    got, n = bm.validate_event_iou(base, ds, torch.device("cpu"), thresholds=[0.5, 2.0], max_frames=3)
    assert n == 3 and set(got) == {0.5, 2.0}
