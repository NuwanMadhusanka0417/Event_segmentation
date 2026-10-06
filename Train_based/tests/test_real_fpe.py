"""Real-valued FPE (complex -> real conversion) and the per-pixel hypervector of the heads."""

import math

import numpy as np
import torch

from hdems.config import apply_resolution_ratio, frontend_settings, restore_frontend
from hdems.models.hdems import HDEMS
from hdems.vsa.real_fpe import (
    RealFPE2D,
    bind,
    check_real,
    hermitian_frequencies,
    similarity,
    to_real,
    unbind,
)


def _code(fpe, x, y=0.0):
    return fpe(torch.tensor([float(x)]), torch.tensor([float(y)]), dim=1)[0]


def test_hermitian_frequencies_give_real_vectors():
    om = hermitian_frequencies(500, seed=0)
    assert om[0] == 0 and om[250] == 0
    assert torch.allclose(om[1:250], -om[251:].flip(0))
    assert float(om.abs().max()) <= math.pi
    spec = torch.exp(1j * 0.0025 * 123.4 * om)
    assert check_real(spec) < 1e-5
    v = to_real(spec)
    assert v.dtype == torch.float32 and abs(float(v.norm()) - math.sqrt(500)) < 1e-3


def test_similarity_follows_the_sinc_kernel():
    """Uniform omega: E[cos(beta*D*omega)] = sin(pi*beta*D) / (pi*beta*D)."""
    fpe = RealFPE2D(8000, 0.0025, seed=1)
    p0 = _code(fpe, 0.0)
    for dx in (100, 200, 320, 640):
        z = math.pi * 0.0025 * dx
        assert abs(float(similarity(p0, _code(fpe, dx))) - math.sin(z) / z) < 0.04


def test_d500_matches_the_2d_fpe_experiment():
    """beta 0.0025, d 500: ~0.9 at 100 px, ~0.6 at 200 px, ~0 by 400 px, slightly < 0 at 640."""
    fpe = RealFPE2D(500, 0.0025, seed=42)
    p0 = _code(fpe, 0.0, 0.0)
    s = {dx: float(similarity(p0, _code(fpe, dx, 0.0))) for dx in (21, 100, 200, 400, 640)}
    assert s[21] > 0.98 and 0.8 < s[100] < 0.97 and 0.45 < s[200] < 0.75
    assert abs(s[400]) < 0.15 and s[640] < 0.05


def test_binding_is_circular_convolution_and_adds_coordinates():
    d = 16
    fpe = RealFPE2D(d, 0.3, seed=3)
    a, b = _code(fpe, 2.0, 1.0), _code(fpe, -0.5, 3.0)
    an, bn = a.numpy().astype(np.float64), b.numpy().astype(np.float64)
    textbook = np.array([sum(an[k] * bn[(n - k) % d] for k in range(d)) for n in range(d)])
    assert np.allclose(bind(a, b).numpy(), textbook / math.sqrt(d), atol=1e-4)
    assert torch.allclose(bind(a, b), _code(fpe, 1.5, 4.0), atol=1e-4)    # X^x ⊛ X^x' = X^(x+x')
    assert torch.allclose(unbind(bind(a, b), b), a, atol=1e-4)


def test_x_and_y_bases_are_independent_and_fixed():
    fpe = RealFPE2D(500, 0.05, seed=5)
    assert abs(float(similarity(_code(fpe, 20.0, 0.0), _code(fpe, 0.0, 20.0)))) < 0.2
    again = RealFPE2D(500, 0.05, seed=5)
    assert torch.equal(fpe.omega_x, again.omega_x) and torch.equal(fpe.omega_y, again.omega_y)


def _tiny(head="mfunet", real=True, hv_input="pv", axis="bind"):
    return {
        "d": 32,
        "encoder": {"patch_size": 5, "sigma_k": 1.0, "kernel": "conv",
                    "polarity_binding": True, "scales": 1},
        "matching": {"M": 5, "scales": [0, 1, 2], "alpha": 0.3, "smooth": 3},
        "dataset": {"height": 16, "width": 20, "time_frames": [1.0, 0.75, 0.5, 0.0],
                    "resolution_ratio": 2},
        "segmentation": {"head": head, "num_classes": 2, "mf_motion_dim": 8,
                         "mf_widths": [8, 16, 16, 16]},
        "velocity": {"event_feature": "f", "axis_combine": axis},
        "real_fpe": {"enabled": real, "d": 64, "beta_pos": 0.0025, "beta_vel": 0.05,
                     "vel_range_px": 15.0, "input": hv_input},
        "flow_cache": {"enabled": False},
    }


def test_velocity_mapping_minus15_to_0_and_plus15_to_30():
    m = HDEMS(_tiny(hv_input="v")).eval()
    r = m.res_ratio                                      # residual is in WORKING px
    res = torch.zeros(1, 2, 16, 20)
    res[0, :, 0, 0] = -15.0 / r                          # -15 sensor px -> u = 0
    res[0, :, 0, 1] = 15.0 / r                           # +15 sensor px -> u = 30
    res[0, :, 0, 2] = -40.0 / r                          # beyond the range: clamped to u = 0
    hv = m.pixel_hv(res)
    assert hv.shape == (1, 64, 16, 20) and not hv.is_complex()
    delta = torch.zeros(64)
    delta[0] = math.sqrt(64)                             # FPE(0) = identity of binding
    assert torch.allclose(hv[0, :, 0, 0], delta, atol=1e-4)
    assert torch.allclose(hv[0, :, 0, 1], _code(m.hv_vel, 30.0, 30.0), atol=1e-4)
    assert torch.allclose(hv[0, :, 0, 2], hv[0, :, 0, 0])


def test_pixel_hv_is_position_bound_to_velocity():
    m = HDEMS(_tiny(hv_input="pv")).eval()
    res = torch.zeros(1, 2, 16, 20)
    res[0, 0, 5, 7], res[0, 1, 5, 7] = 0.8, -1.1
    hv = m.pixel_hv(res)
    r = m.res_ratio
    u = (res[0, :, 5, 7] * r).clamp(-15, 15) + 15
    want = bind(_code(m.hv_pos, 7 * r, 5 * r), _code(m.hv_vel, float(u[0]), float(u[1])))
    assert torch.allclose(hv[0, :, 5, 7], want, atol=1e-4)


def test_combine_options():
    res = torch.zeros(1, 2, 16, 20)
    res[0, 0, 5, 7], res[0, 1, 5, 7] = 0.8, -1.1
    for axis in ("bind", "bundle"):
        for hv_input in ("pv_bind", "pv_bundle", "pv_concat", "v"):
            m = HDEMS(_tiny(hv_input=hv_input, axis=axis)).eval()
            r = m.res_ratio
            ux, uy = [float(u) for u in (res[0, :, 5, 7] * r).clamp(-15, 15) + 15]
            V = (_code(m.hv_vel, ux, uy) if axis == "bind"
                 else (_code(m.hv_vel, ux, 0.0) + _code(m.hv_vel, 0.0, uy)) / math.sqrt(2))
            P = _code(m.hv_pos, 7 * r, 5 * r)
            want = {"pv_bind": bind(P, V), "pv_bundle": (P + V) / math.sqrt(2),
                    "pv_concat": torch.cat([P, V]), "v": V}[hv_input]
            hv = m.pixel_hv(res)
            assert hv.shape[1] == m.seg_head.motion_proj.in_channels == want.numel()
            assert torch.allclose(hv[0, :, 5, 7], want, atol=1e-4), (axis, hv_input)


def test_mfunet_with_real_fpe_and_legacy_checkpoints():
    s = (torch.rand(1, 4, 2, 16, 20, generator=torch.Generator().manual_seed(3)) > 0.7).float()
    m = HDEMS(_tiny()).eval()
    assert m.seg_head.motion_proj.in_channels == 64          # real code: d_hv channels
    with torch.no_grad():
        assert m(s, task="segmentation")["seg_logits"].shape == (1, 2, 16, 20)
    # a checkpoint from before real_fpe records no such section -> complex code
    saved = frontend_settings(apply_resolution_ratio(_tiny()))
    del saved["real_fpe"]
    old = HDEMS(restore_frontend(_tiny(), saved))
    assert not old.real_fpe and old.seg_head.motion_proj.in_channels == 2 * 32
