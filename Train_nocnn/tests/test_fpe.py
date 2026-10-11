"""Gate test 1: FPE binding, unbinding, and translation invariance."""

from __future__ import annotations

import torch

from hdems.vsa.fpe import bind, fpe, make_base_phases, similarity, unbind


def test_bind_unbind_roundtrip() -> None:
    d = 256
    phases = make_base_phases(d, seed=0)
    x = torch.tensor([1.5, -2.3, 0.7])
    hv = fpe(phases, x[0])
    role = fpe(phases, 42.0)

    bound = bind(hv, role)
    recovered = unbind(bound, role)
    sim = similarity(hv, recovered).item()
    assert sim > 0.99, f"roundtrip similarity {sim} <= 0.99"


def test_fpe_additivity() -> None:
    """fpe(p, a+b) == bind(fpe(p,a), fpe(p,b))."""
    d = 512
    phases = make_base_phases(d, seed=1)
    a, b = 1.7, -0.9
    lhs = fpe(phases, a + b)
    rhs = bind(fpe(phases, a), fpe(phases, b))
    err = (lhs - rhs).abs().max().item()
    assert err < 1e-6, f"additivity error {err}"


def test_translation_invariance() -> None:
    """Similarity depends only on x - x'."""
    d = 512
    phases = make_base_phases(d, seed=2)
    x1, x2 = 3.0, 7.0
    delta = x2 - x1

    hv1 = fpe(phases, x1)
    hv2 = fpe(phases, x2)
    sim_direct = similarity(hv1, hv2)

    # Shift both by same offset — delta unchanged
    offset = 100.0
    hv1s = fpe(phases, x1 + offset)
    hv2s = fpe(phases, x2 + offset)
    sim_shifted = similarity(hv1s, hv2s)

    err = abs(sim_direct - sim_shifted).item()
    assert err < 1e-6, f"translation invariance violated: {err}"

    # Similarity should depend on delta only
    hv_ref = fpe(phases, delta)
    hv_unit = fpe(phases, 0.0)
    sim_delta = similarity(hv_unit, hv_ref)
    assert abs(sim_direct - sim_delta).item() < 1e-4
