"""FPE codebook encoding for GVFA event segmentation (Frady et al. VFA / HRR).

Kernel shape is set by the base-phase distribution:
  - phase ~ Uniform(-pi, pi)  -> sinc kernel  (oscillating side-lobes; far values
                                  get spurious positive/negative similarity)
  - phase ~ Normal(0, sigma^2) -> Gaussian kernel exp(-0.5*(bw*delta)^2)
                                  (monotonic decay to 0, NO side-lobes)  [default]
Both keep codes exactly real (Hermitian phases) and keep the radix binding
identity z(a+b) = bind(z(a), z(b)) exact (only the exponent adds).
"""

from __future__ import annotations

import numpy as np
import torch


def bind_hv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular convolution (VSA bind) on last dimension; returns real tensor."""
    fa = torch.fft.fft(a, dim=-1)
    fb = torch.fft.fft(b, dim=-1)
    return torch.fft.ifft(fa * fb, dim=-1).real


def _make_hermitian_phases(
    D: int,
    rng: np.random.Generator,
    dist: str = "gaussian",
    sigma: float = 1.0,
) -> np.ndarray:
    """Conjugate-symmetric phases -> ifft(exp(i*v*phase)) is exactly real.

    dist='gaussian' -> Gaussian similarity kernel (monotonic, no side-lobes).
    dist='uniform'  -> sinc kernel (legacy; has side-lobes).
    """
    def draw() -> float:
        if dist == "gaussian":
            return float(rng.normal(0.0, sigma))
        if dist == "uniform":
            return float(rng.uniform(-np.pi, np.pi))
        raise ValueError(f"unknown phase dist: {dist}")

    phase = np.zeros(D, dtype=np.float64)
    half = D // 2 if D % 2 == 0 else (D + 1) // 2
    for k in range(1, half):
        p = draw()
        phase[k] = p
        phase[D - k] = -p
    # phase[0] stays 0; for even D phase[D//2] stays 0 (self-conjugate bins)
    return phase.astype(np.float32)


def _z_from_phase(phase: torch.Tensor, bandwidth: float, v: float) -> torch.Tensor:
    ang = float(v) * bandwidth * phase
    return torch.fft.ifft(torch.exp(1j * ang)).real.float()


class FPECodebook:
    """One feature codebook with its own Hermitian base and bandwidth (length-scale)."""

    def __init__(
        self,
        name: str,
        D: int,
        bandwidth: float,
        kind: str,
        *,
        value_grid_step: float | None = None,
        radix_S: int | None = None,
        signed_log_v0: float | None = None,
        vmin: int | None = None,
        vmax: int | None = None,
        phase_dist: str = "gaussian",
        phase_sigma: float = 1.0,
        seed: int = 0,
    ):
        self.name = name
        self.D = int(D)
        self.bandwidth = float(bandwidth)
        self.kind = kind
        self.value_grid_step = value_grid_step
        self.radix_S = int(radix_S) if radix_S is not None else None
        self.signed_log_v0 = float(signed_log_v0) if signed_log_v0 is not None else None
        self.phase_dist = phase_dist
        self.phase_sigma = float(phase_sigma)
        self.seed = int(seed)
        self.vmin = 0 if vmin is None else int(vmin)
        self.vmax = 0 if vmax is None else int(vmax)

        rng = np.random.default_rng(seed)
        self.phase = torch.from_numpy(
            _make_hermitian_phases(self.D, rng, self.phase_dist, self.phase_sigma)
        )

        self._table: torch.Tensor | None = None
        self._fine: torch.Tensor | None = None
        self._coarse: torch.Tensor | None = None

        if kind == "integer":
            self._build_integer_table()
        elif kind in ("radix", "signed_log_radix"):
            if self.radix_S is None:
                raise ValueError(f"radix_S required for '{name}' ({kind})")
            if kind == "signed_log_radix" and self.signed_log_v0 is None:
                raise ValueError(f"signed_log_v0 required for '{name}'")
            self._build_radix_tables()
        else:
            raise ValueError(f"Unknown codebook kind: {kind}")

    def _build_integer_table(self) -> None:
        rows = [
            _z_from_phase(self.phase, self.bandwidth, float(v))
            for v in range(self.vmin, self.vmax + 1)
        ]
        self._table = torch.stack(rows, dim=0)

    def _build_radix_tables(self) -> None:
        S = self.radix_S
        span = max(1, self.vmax - self.vmin + 1)
        A = max(1, (span + S - 1) // S)
        self._fine = torch.stack(
            [_z_from_phase(self.phase, self.bandwidth, float(lo)) for lo in range(S)],
            dim=0,
        )
        self._coarse = torch.stack(
            [_z_from_phase(self.phase, self.bandwidth, float(hi * S)) for hi in range(A)],
            dim=0,
        )

    def z(self, v: float) -> torch.Tensor:
        if self.kind == "integer":
            idx = int(np.clip(round(v), self.vmin, self.vmax)) - self.vmin
            return self._table[idx].clone()
        V = int(np.clip(round(v), self.vmin, self.vmax)) - self.vmin
        lo = V % self.radix_S
        hi = min(V // self.radix_S, self._coarse.shape[0] - 1)
        return bind_hv(self._coarse[hi : hi + 1], self._fine[lo : lo + 1]).squeeze(0)

    def _to_grid_index(self, values: np.ndarray) -> np.ndarray:
        if self.value_grid_step is not None and self.value_grid_step > 0:
            return np.round(values / self.value_grid_step)
        return np.round(values)

    def _precondition(self, values: np.ndarray) -> np.ndarray:
        if self.kind != "signed_log_radix":
            return values
        v0 = self.signed_log_v0
        return np.sign(values) * np.log1p(np.abs(values) / v0)

    def _radix_codes(self, V: np.ndarray) -> torch.Tensor:
        """Vectorized: z(V) = bind(coarse[hi], fine[lo]) for all V at once."""
        V = np.clip(V.astype(np.int64), self.vmin, self.vmax)
        V_off = V - self.vmin
        lo = torch.from_numpy((V_off % self.radix_S)).long()
        hi = torch.from_numpy(
            np.clip(V_off // self.radix_S, 0, self._coarse.shape[0] - 1)
        ).long()
        return bind_hv(self._coarse[hi], self._fine[lo])  # [N, D]

    def encode(
        self,
        values: np.ndarray | torch.Tensor,
        *,
        interpolate: bool = False,
    ) -> torch.Tensor:
        """Vectorized encode -> [N, D] float32 real hypervectors."""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = self._precondition(np.asarray(values, dtype=np.float64).ravel())

        if self.kind == "integer":
            idx = np.round(values).astype(np.int64)
            idx = np.clip(idx, self.vmin, self.vmax) - self.vmin
            return self._table[idx].clone()

        grid = self._to_grid_index(values)
        if interpolate and self.value_grid_step is not None and self.value_grid_step > 0:
            cont = values / self.value_grid_step
            V0 = np.floor(cont).astype(np.int64)
            frac = (cont - V0).astype(np.float32)
            V1 = np.clip(V0 + 1, self.vmin, self.vmax)
            V0 = np.clip(V0, self.vmin, self.vmax)
            c0 = self._radix_codes(V0)
            c1 = self._radix_codes(V1)
            f = torch.from_numpy(frac[:, None])
            return (1.0 - f) * c0 + f * c1

        return self._radix_codes(np.clip(grid, self.vmin, self.vmax).astype(np.int64))

    def self_test(self, tol_real: float = 1e-6, tol_bind: float = 1e-4) -> bool:
        ok = True
        for v in (0.0, 2.0, 7.0):
            zv = self.z(v)
            if not torch.isreal(zv).all() and zv.imag.abs().max().item() > tol_real:
                ok = False
                print(f"[{self.name}] non-real code at v={v}")

        if self.kind in ("radix", "integer"):
            a, b = 3.0, 5.0
            za, zb, zsum = self.z(a), self.z(b), self.z(a + b)
            err = (bind_hv(za.unsqueeze(0), zb.unsqueeze(0)).squeeze(0) - zsum).abs().max().item()
            if err > tol_bind:
                ok = False
                print(f"[{self.name}] bind additivity err={err:.2e}")

            base = self.z(20.0)
            sims = [
                torch.nn.functional.cosine_similarity(
                    base.unsqueeze(0), self.z(20.0 + d).unsqueeze(0)
                ).item()
                for d in (0.0, 1.0, 3.0, 8.0)
            ]
            if any(sims[i] < sims[i + 1] - 1e-5 for i in range(len(sims) - 1)):
                ok = False
                print(f"[{self.name}] kernel not decreasing: {sims}")

        return ok


def bundle_weighted(terms: list[tuple[torch.Tensor, float]]) -> torch.Tensor:
    """Weighted sum of [N,D] tensors then L2-normalize per row."""
    out = sum(w * t.float() for t, w in terms)
    return torch.nn.functional.normalize(out, p=2, dim=1)


if __name__ == "__main__":
    cb = FPECodebook("test", 512, 0.5, "radix", radix_S=32, vmin=0, vmax=1023, seed=0)
    ok = cb.self_test()
    print("self_test:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)