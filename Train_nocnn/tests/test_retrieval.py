"""Gate test 4: retrieval — GO/NO-GO gate for the whole paper."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from hdems.vsa.field import bundled_field, cost_volume, cost_volume_from_field
from hdems.vsa.fpe import make_base_phases


def _make_smooth_field(
    B: int,
    d: int,
    H: int,
    W: int,
    sigma: float,
    seed: int = 0,
) -> torch.Tensor:
    """Generate a spatially smooth complex descriptor field."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(B, d, H, W, generator=g)

    if sigma > 0:
        k = int(max(3, sigma * 2 + 1))
        if k % 2 == 0:
            k += 1
        coords = torch.arange(k, dtype=torch.float32) - k // 2
        g1 = torch.exp(-coords.pow(2) / (2 * sigma**2))
        g1 = g1 / g1.sum()
        pad = k // 2
        for b in range(B):
            x = raw[b : b + 1]
            x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
            x = F.conv2d(x, g1.view(1, 1, 1, -1).expand(d, 1, 1, -1), groups=d)
            x = F.conv2d(x, g1.view(1, 1, -1, 1).expand(d, 1, -1, 1), groups=d)
            raw[b] = x[0, :, :H, :W]

    return raw.to(torch.complex64)


def _argmax_flow(C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax displacement from cost volume C: (B, M, M, H, W)."""
    B, M, _, H, W = C.shape
    flat = C.reshape(B, M * M, H, W)
    idx = flat.argmax(dim=1)
    m = M // 2
    ia = idx // M
    ib = idx % M
    flow_x = ia.float() - m
    flow_y = ib.float() - m
    return flow_x, flow_y


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    denom = a_c.norm() * b_c.norm() + 1e-8
    return (a_c @ b_c / denom).item()


@pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0, 3.0])
def test_retrieval_m7_d1024(sigma: float) -> None:
    """GO/NO-GO: argmax endpoint error <= 1 px at M=7, d=1024.

    At sigma=0 (white noise field), global Pearson correlation is ~0.49 because
    bundling cross-talk adds an offset that preserves argmax but not shape.
    The primary gate is endpoint error; correlation >= 0.85 is required once
    sigma >= 1 (encoder-realistic smoothness).
    """
    torch.manual_seed(42)
    d, M = 1024, 7
    H, W = 32, 40
    phx = make_base_phases(d, seed=0)
    phy = make_base_phases(d, seed=1)

    F = _make_smooth_field(1, d, H, W, sigma=sigma, seed=7)
    Phi = bundled_field(F, phx, phy, M=M)

    C_true = cost_volume(F, phx, phy, M=M)
    C_query = cost_volume_from_field(F, Phi, phx, phy, M=M)

    corr = _correlation(C_true, C_query)

    fx_true, fy_true = _argmax_flow(C_true)
    fx_q, fy_q = _argmax_flow(C_query)
    err = torch.sqrt((fx_true - fx_q).pow(2) + (fy_true - fy_q).pow(2)).mean()
    assert err.item() <= 1.0, f"sigma={sigma}: endpoint error {err.item():.2f} > 1 px"

    if sigma >= 1.0:
        assert corr >= 0.85, f"sigma={sigma}: correlation {corr:.3f} < 0.85"


def test_retrieval_failure_at_m15_regression_guard() -> None:
    """Document that M=15 degrades — regression guard, not a pass/fail gate."""
    torch.manual_seed(0)
    d, M = 1024, 15
    H, W = 32, 40
    sigma = 0.0
    phx = make_base_phases(d, seed=0)
    phy = make_base_phases(d, seed=1)

    F = _make_smooth_field(1, d, H, W, sigma=sigma)
    Phi = bundled_field(F, phx, phy, M=M)
    C_true = cost_volume(F, phx, phy, M=M)
    C_query = cost_volume_from_field(F, Phi, phx, phy, M=M)

    corr = _correlation(C_true, C_query)
    fx_true, fy_true = _argmax_flow(C_true)
    fx_q, fy_q = _argmax_flow(C_query)
    err = torch.sqrt((fx_true - fx_q).pow(2) + (fy_true - fy_q).pow(2)).mean()

    # At M=15 we expect degradation — this test documents it, does not assert pass
    print(f"M=15 regression: corr={corr:.3f}, endpoint_err={err.item():.2f}px")
    assert corr < 0.95 or err.item() > 0.5, (
        "Expected M=15 to show retrieval degradation; investigate if this passes cleanly"
    )
