"""Gate test 5: translation equivariance of descriptor field."""

from __future__ import annotations

import torch

from hdems.vsa.fpe import fpe, make_base_phases


def _encode_positions(
    phases_x: torch.Tensor,
    phases_y: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
) -> torch.Tensor:
    """Encode 2-D positions as bound FPE codes. Returns (N, d) complex."""
    return fpe(phases_x, xs) * fpe(phases_y, ys)


def test_shifting_input_shifts_descriptor() -> None:
    """Shifting spatial coordinates shifts the descriptor exactly (equivariance)."""
    d = 512
    phx = make_base_phases(d, seed=0)
    phy = make_base_phases(d, seed=1)

    xs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ys = torch.tensor([0.5, 1.5, 2.5, 3.5])
    shift_x, shift_y = 5.0, -3.0

    hv = _encode_positions(phx, phy, xs, ys)
    hv_shifted = _encode_positions(phx, phy, xs + shift_x, ys + shift_y)

    # Expected: bind each original with position shift code
    shift_code = fpe(phx, shift_x) * fpe(phy, shift_y)
    expected = hv * shift_code.unsqueeze(0)

    err = (hv_shifted - expected).abs().max().item()
    assert err < 1e-5, f"equivariance error {err} >= 1e-5"
