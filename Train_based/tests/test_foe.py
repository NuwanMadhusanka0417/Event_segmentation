"""Depth-free FOE background test (hdems/foe.py) on synthetic translational flow."""

import torch

from hdems.foe import FoeParams, estimate_foe, foe_cue_maps, foe_cues

H, W = 60, 80


def _scene(T, mover=(0.0, 0.0), seed=0):
    """Backward displacement of static points with random depth under camera
    translation T (pixel-unit FOE geometry, f = 50 px, centre (40, 30)), plus a 10x10
    mover with its own image motion. -> d (2, H, W), mover mask (H, W)."""
    g = torch.Generator().manual_seed(seed)
    f, cx, cy = 50.0, 40.0, 30.0
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    Z = 1.0 + 4.0 * torch.rand(H, W, generator=g)                  # depth 1..5: parallax
    Tx, Ty, Tz = T
    xn, yn = (xx - cx) / f, (yy - cy) / f
    # image velocity of a static point, then displacement toward EARLIER times
    u = -(f / Z) * (Tx - xn * Tz)
    v = -(f / Z) * (Ty - yn * Tz)
    d = -torch.stack([u, v])
    mov = torch.zeros(H, W, dtype=torch.bool)
    mov[20:30, 50:60] = True
    d[0][mov] = mover[0]
    d[1][mov] = mover[1]
    return d, mov


def _separation(cue, mov):
    return float(cue[mov].min()), float(cue[~mov].quantile(0.95))


def test_forward_translation_flags_a_crossing_mover_despite_parallax():
    d, mov = _scene((0.0, 0.0, 0.1), mover=(0.0, 3.0))
    valid = torch.ones(H, W, dtype=torch.bool)
    foe = estimate_foe(d, valid)
    assert foe.e is not None and abs(float(foe.e[2])) > 1e-3
    e = foe.e[:2] / foe.e[2]
    assert torch.allclose(e, torch.tensor([40.0, 30.0]), atol=0.5)    # the image centre
    cues = foe_cues(d, foe)
    lo_mover, hi_static = _separation(cues["foe"], mov)
    assert lo_mover > 1.0 and hi_static < 0.05
    # magnitude alone cannot tell them apart: static points move up to several px
    assert float(cues["derotated"][~mov].max()) > 3.0


def test_sideways_translation_puts_the_foe_at_infinity():
    d, mov = _scene((0.1, 0.0, 0.0), mover=(0.0, 2.0))
    foe = estimate_foe(d, torch.ones(H, W, dtype=torch.bool))
    assert foe.e is not None
    assert abs(float(foe.e[2])) < 1e-3 * float(foe.e[:2].norm())     # direction only
    lo_mover, hi_static = _separation(foe_cues(d, foe)["foe"], mov)
    assert lo_mover > 1.0 and hi_static < 0.05


def test_moving_against_the_expansion_is_caught_by_the_sign():
    # forward motion: static points' backward displacement points TOWARD the FOE;
    # a mover going radially AWAY has no perpendicular part, only the wrong sign
    d, mov = _scene((0.0, 0.0, 0.1))
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    away = torch.stack([xx - 40.0, yy - 30.0])
    away = 2.0 * away / away.norm(dim=0).clamp_min(1e-6)
    d[:, mov] = away[:, mov]
    foe = estimate_foe(d, torch.ones(H, W, dtype=torch.bool))
    cues = foe_cues(d, foe)
    assert float(cues["perp"][mov].max()) < 0.3              # along the line: perp misses it
    assert float(cues["foe"][mov].min()) > 1.0               # the sign does not


def test_no_translation_falls_back_to_the_magnitude():
    d = torch.zeros(2, H, W)
    mov = torch.zeros(H, W, dtype=torch.bool)
    mov[20:30, 50:60] = True
    d[0][mov] = 3.0                      # only the mover moves: it must not define the FOE
    foe = estimate_foe(d, torch.ones(H, W, dtype=torch.bool))
    assert foe.e is None
    cues = foe_cues(d, foe)
    assert torch.equal(cues["foe"], d.norm(dim=0))


def test_batch_maps_remove_the_rotation_first():
    d, mov = _scene((0.0, 0.0, 0.1), mover=(0.0, 3.0))
    rot = torch.full((1, 2, H, W), 0.7)
    maps, foes = foe_cue_maps((d[None] + rot), rot, torch.ones(1, H, W, dtype=torch.bool),
                              FoeParams(seed=3))
    assert maps["foe"].shape == (1, 1, H, W) and len(foes) == 1
    lo_mover, hi_static = _separation(maps["foe"][0, 0], mov)
    assert lo_mover > 1.0 and hi_static < 0.05
