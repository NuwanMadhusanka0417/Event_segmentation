"""Gate test 2: VFA eigen basis rank and energy retention."""

from __future__ import annotations

import torch

from hdems.vsa.kernel import eigen_basis, vfa_gram


def test_eigen_basis_99pct_energy_at_r86() -> None:
    _, frac = eigen_basis(N=21, sigma_k=1.5, r=86)
    assert frac >= 0.99, f"energy fraction {frac} < 0.99 at r=86"


def test_eigen_beats_random_projection() -> None:
    """Eigen basis retains more energy than random projection at equal r."""
    N, r = 21, 64
    filters, eigen_frac = eigen_basis(N, 1.5, r)

    G = vfa_gram(N, 1.5)
    evals, _ = torch.linalg.eigh(G)
    evals = evals.flip(0).clamp_min(0)
    total = evals.sum()

    # Random rank-r projection: expected captured energy ~ r / (N*N) for uniform random
    # but proper comparison: project G onto random r-dim subspace
    torch.manual_seed(0)
    rand = torch.randn(N * N, r, dtype=torch.float64)
    rand, _ = torch.linalg.qr(rand)
    captured = (rand.T @ G @ rand).trace()
    random_frac = (captured / total).item()

    assert eigen_frac > random_frac, (
        f"eigen {eigen_frac:.4f} did not beat random {random_frac:.4f} at r={r}"
    )


def test_filter_shape() -> None:
    filters, _ = eigen_basis(N=21, sigma_k=1.5, r=32)
    assert filters.shape == (32, 21, 21)
