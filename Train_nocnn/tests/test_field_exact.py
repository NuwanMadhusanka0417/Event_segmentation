"""Gate test 3: separable bundled field equals explicit M^2 sum."""

from __future__ import annotations

import torch

from hdems.vsa.field import bundled_field, bundled_field_explicit
from hdems.vsa.fpe import make_base_phases


def test_separable_equals_explicit() -> None:
    torch.manual_seed(0)
    B, d, H, W = 1, 128, 16, 20
    M = 7

    phx = make_base_phases(d, seed=0)
    phy = make_base_phases(d, seed=1)

    F_real = torch.randn(B, d, H, W, dtype=torch.float64)
    F = F_real.to(torch.complex128)

    sep = bundled_field(F, phx.double(), phy.double(), M=M)
    exp = bundled_field_explicit(F, phx.double(), phy.double(), M=M)

    rel_err = (sep - exp).abs().max() / (exp.abs().max() + 1e-12)
    assert rel_err.item() < 1e-10, f"relative error {rel_err.item()} >= 1e-10"


def test_separable_multiple_M() -> None:
    """Separability holds for M=5 and M=9 as well."""
    torch.manual_seed(1)
    B, d, H, W = 1, 64, 8, 10
    phx = make_base_phases(d, seed=0).double()
    phy = make_base_phases(d, seed=1).double()
    F = torch.randn(B, d, H, W, dtype=torch.float64).to(torch.complex128)

    for M in (5, 7, 9):
        sep = bundled_field(F, phx, phy, M=M)
        exp = bundled_field_explicit(F, phx, phy, M=M)
        rel_err = (sep - exp).abs().max() / (exp.abs().max() + 1e-12)
        assert rel_err.item() < 1e-10, f"M={M}: relative error {rel_err.item()}"
