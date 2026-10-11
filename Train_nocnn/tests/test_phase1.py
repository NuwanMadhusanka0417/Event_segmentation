"""Phase-1 accuracy fixes: Phi window/borders, fixed-unit velocity, motion-first head,
legacy checkpoints, time-surface decay and the recent-event scoring mask."""

import numpy as np
import torch

from hdems.config import apply_resolution_ratio, frontend_settings, restore_frontend
from hdems.data.evimo2_reader import events_window_to_surface, recent_event_mask
from hdems.models.hdems import HDEMS
from hdems.models.segmentation import MotionFirstHead
from hdems.vsa.field import bundled_field, bundled_field_explicit
from hdems.vsa.fpe import make_base_phases
from hdems.vsa.velocity import encode_velocity, motion_scalars


def _field(d=32, h=12, w=14, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(1, d, h, w, generator=g), torch.randn(1, d, h, w, generator=g))


def test_zero_pad_matches_explicit_and_wrap_in_interior():
    F = _field()
    phx, phy = make_base_phases(32, seed=0), make_base_phases(32, seed=1)
    zero = bundled_field(F, phx, phy, M=5, pad="zero")
    assert torch.allclose(zero, bundled_field_explicit(F, phx, phy, M=5, pad="zero"), atol=1e-4)
    wrap = bundled_field(F, phx, phy, M=5, pad="wrap")
    m = 2
    assert torch.allclose(zero[..., m:-m, m:-m], wrap[..., m:-m, m:-m], atol=1e-4)
    assert not torch.allclose(zero[..., :m, :], wrap[..., :m, :], atol=1e-3)


def test_zero_pad_ignores_the_opposite_border():
    F = torch.zeros(1, 8, 10, 10, dtype=torch.complex64)
    F[..., 9, 9] = 1.0                                     # one event, bottom-right corner
    phx, phy = make_base_phases(8, seed=0), make_base_phases(8, seed=1)
    out = bundled_field(F, phx, phy, M=5, pad="zero")
    assert out[..., 0, 0].abs().max() == 0                 # wrap would put it top-left
    assert out[..., 8, 8].abs().max() > 0


def test_fixed_velocity_code_does_not_depend_on_the_rest_of_the_frame():
    phx, phy = make_base_phases(64, seed=7), make_base_phases(64, seed=8)
    still = torch.full((1, 8, 8), 0.1)                     # flow noise only
    fast = still.clone()
    fast[0, :2, :2] = 9.0                                  # a fast mover elsewhere
    kw = dict(axis_combine="bind")
    a = encode_velocity(still, still, phx, phy, norm="fixed", unit=0.5, **kw)
    b = encode_velocity(fast, fast, phx, phy, norm="fixed", unit=0.5, **kw)
    assert torch.allclose(a[..., 5, 5], b[..., 5, 5])      # same speed -> same code
    a = encode_velocity(still, still, phx, phy, norm="frame", **kw)
    b = encode_velocity(fast, fast, phx, phy, norm="frame", **kw)
    assert not torch.allclose(a[..., 5, 5], b[..., 5, 5])  # old: meaning changes per frame


def test_motion_scalars():
    res = torch.zeros(1, 2, 6, 6)
    res[0, 0, 0, 0] = 5.0
    ev = torch.ones(1, 6, 6, dtype=torch.bool)
    s = motion_scalars(res, ev, res.clone(), unit=0.5)     # still camera: flow = residual
    assert s.shape == (1, 6, 6, 6)
    assert s[0, :, 3, 3].abs().max() == 0                  # still pixel -> all zero
    assert s[0, 2, 0, 0] > 2 and s[0, 3, 0, 0] > s[0, 2, 0, 0]   # noise floor 0.1 < unit
    assert torch.allclose(s[0, 4], s[0, 2]) and s[0, 5].abs().max() == 0
    cam = motion_scalars(res, ev, res + 2.0, unit=0.5)     # camera moves 2 px on both axes
    assert torch.allclose(cam[0, 5], torch.full((6, 6), float(np.log1p(8 ** 0.5 / 0.5))))


def test_motion_first_head_drops_appearance_only_in_training():
    torch.manual_seed(0)
    head = MotionFirstHead(16, 2, app_dim=4, app_dropout=1.0)
    mv, x = _field(16, 8, 8, 1), _field(16, 8, 8, 2)
    sc = torch.randn(1, 6, 8, 8)
    head.train()
    assert torch.allclose(head(mv, x, sc), head(mv, torch.zeros_like(x), sc))
    head.eval()
    assert head(mv, x, sc).shape == (1, 2, 8, 8)
    assert not torch.allclose(head(mv, x, sc), head(mv, torch.zeros_like(x), sc))


def _tiny_cfg(head="mfcnn"):
    return {
        "d": 32,
        "encoder": {"patch_size": 5, "sigma_k": 1.0, "kernel": "conv",
                    "polarity_binding": True, "scales": 1},
        "matching": {"M": 5, "scales": [0, 1, 2], "alpha": 0.3, "smooth": 3},
        "dataset": {"height": 16, "width": 20, "time_frames": [1.0, 0.75, 0.5, 0.0]},
        "segmentation": {"head": head, "num_classes": 2, "embedding_dim": 8},
        "velocity": {"event_feature": "phi", "event_combine": "concat"},
        "flow_cache": {"enabled": False},
    }


def _stack():
    g = torch.Generator().manual_seed(3)
    s = (torch.rand(1, 4, 2, 16, 20, generator=g) > 0.7).float()
    return s


def test_mfcnn_forward_and_ablation():
    model = HDEMS(_tiny_cfg()).eval()
    s = _stack()
    with torch.no_grad():
        full = model(s, task="segmentation")["seg_logits"]
        assert full.shape == (1, 2, 16, 20)
        for what in ("motion", "appearance"):
            model.ablate = what
            model(s, task="segmentation")
        model.ablate = None


def test_legacy_frontend_rebuilds_the_old_phi():
    cfg = apply_resolution_ratio(_tiny_cfg("cnn"))
    saved = frontend_settings(cfg)
    assert saved["matching"]["phi_window"] == 7 and saved["velocity"]["vel_norm"] == "fixed"
    # a checkpoint from before the fix records none of the new keys
    old = {k: {kk: vv for kk, vv in v.items()
               if kk not in ("phi_window", "phi_pad", "vel_norm", "vel_unit_px", "x_norm", "ego_fit")}
           if isinstance(v, dict) else v for k, v in saved.items()}
    del old["velocity"]
    legacy = restore_frontend(_tiny_cfg("cnn"), old)
    assert legacy["matching"]["phi_window"] == 0 and legacy["matching"]["phi_pad"] == "wrap"
    assert legacy["velocity"]["vel_norm"] == "frame" and legacy["velocity"]["ego_fit"] == "stack"
    model = HDEMS(legacy).eval()
    f0 = model.encoder(_stack()[:, 0])
    assert torch.allclose(model.event_hv(f0), model.matcher([f0])[0])   # == old Phi


def _write_events(tmp_path, t, x, y, p):
    np.save(tmp_path / "dataset_events_t.npy", np.asarray(t, dtype=np.float64))
    np.save(tmp_path / "dataset_events_xy.npy", np.stack([x, y], 1).astype(np.int64))
    np.save(tmp_path / "dataset_events_p.npy", np.asarray(p, dtype=np.int64))


def test_single_time_surface_weights_the_newest_event_most(tmp_path):
    _write_events(tmp_path, [1.00, 1.04], [1, 2], [0, 0], [1, 1])
    s = events_window_to_surface(tmp_path, 0.99, 1.05, 3, 4, decay=np.exp(-1000 / 35))
    assert s[1, 0, 2] > s[1, 0, 1]                         # newer event (x=2) is brighter
    assert abs(float(s[1, 0, 2]) - 1.0) < 1e-6


def test_recent_event_mask(tmp_path):
    _write_events(tmp_path, [0.90, 0.995, 1.0], [0, 1, 2], [0, 0, 1], [1, 0, 1])
    m = recent_event_mask(tmp_path, ts=1.0, recent_s=0.0125, height=3, width=4)
    assert m.tolist() == [[False, True, False, False],
                          [False, False, True, False],
                          [False, False, False, False]]
