"""
FPE codebook — precomputed lookup hypervectors (replaces on-the-fly graph.fpe_encode).

Two-level time (memory): n=round(t_us/Q_T), n=c*F_T+f, Phi(t)=PC[c]*PF[f] (complex product,
EXACT binding). Store C_T+F_T complex phasors (~1100 vectors) not 100k flat z(t) — ~90x saving
at 10us resolution (Q_T=10, F_T=1000, C_T=100). One batched ifft per lookup batch for z_t.

Delta reuse: Δx,Δy,Δt,Δp reuse x/y/t/p books over signed ranges — same phi/scale, no delta books.
Velocity is the ONLY new parameter: vx, vy get dedicated signed-log books (ratio, not reachable
from binding). Node HV = x,y,t,p ONLY (no velocity in nodes).

Time uses window-relative absolute µs (t - t_window_start), not per-window normalized t.
Old TIME_BW folded into S_T.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F

SEED = 0
Q_T, F_T, C_T = 10, 1000, 100          # µs quant, fine 0..10ms, coarse 0..1s
S_X = S_Y = S_T = S_P = 1.0             # per-parameter FPE scales (TIME_BW -> S_T)
V_MAX, V0, N_V = 5.0, 0.05, 256         # velocity signed-log bins (px/ms)


def _pseed(name: str, seed: int) -> int:
    h = int(hashlib.md5(f"{seed}:{name}".encode()).hexdigest(), 16)
    return seed + (h % 100_000)


def _phase(name: str, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(_pseed(name, seed))
    return rng.uniform(0, 2 * np.pi, size=dim).astype(np.float32)


def _z_real(phi: np.ndarray, v: float, scale: float) -> np.ndarray:
    ang = (v * scale) * phi
    return np.real(np.fft.ifft(np.exp(1j * ang))).astype(np.float32)


class FPECodebook:
    """Precomputed FPE lookup tables for nodes and edges."""

    def __init__(self, dim=4000, sensor=(304, 240), seed=SEED,
                 q_t=Q_T, f_t=F_T, c_t=C_T, n_v=N_V):
        self.dim = dim
        self.W, self.H = int(sensor[0]), int(sensor[1])
        self.seed = seed
        self.q_t, self.f_t, self.c_t = q_t, f_t, c_t
        self.n_v = n_v
        self.n_t_max = c_t * f_t

        self.phi_x = _phase("x", dim, seed)
        self.phi_y = _phase("y", dim, seed)
        self.phi_t = _phase("t", dim, seed)
        self.phi_p = _phase("p_pol", dim, seed)
        self.phi_vx = _phase("vx", dim, seed)
        self.phi_vy = _phase("vy", dim, seed)

        self.z_x = self._build_signed_book(self.phi_x, self.W, S_X)
        self.z_y = self._build_signed_book(self.phi_y, self.H, S_Y)
        self.z_p = np.stack([_z_real(self.phi_p, v, S_P) for v in (-1.0, 0.0, 1.0)])
        self.z_vx = self._build_velocity_book(self.phi_vx)
        self.z_vy = self._build_velocity_book(self.phi_vy)

        self.PC = np.zeros((c_t, dim), np.complex64)
        self.PF = np.zeros((f_t, dim), np.complex64)
        for c in range(c_t):
            v = (c * f_t * q_t) * S_T
            self.PC[c] = np.exp(1j * v * self.phi_t)
        for f in range(f_t):
            v = (f * q_t) * S_T
            self.PF[f] = np.exp(1j * v * self.phi_t)

        self._size_mb = self._estimate_mb()
        print(f"[FPECodebook] D={dim}  sensor={self.W}x{self.H}  "
              f"time=({c_t}+{f_t}) phasors  total≈{self._size_mb:.1f} MB")

    def _estimate_mb(self) -> float:
        n = (self.z_x.nbytes + self.z_y.nbytes + self.z_p.nbytes +
             self.z_vx.nbytes + self.z_vy.nbytes +
             self.PC.nbytes + self.PF.nbytes)
        return n / (1024 ** 2)

    @property
    def size_mb(self) -> float:
        return self._size_mb

    def _build_signed_book(self, phi, half_bins: int, scale: float) -> np.ndarray:
        n = 2 * half_bins + 1
        book = np.empty((n, self.dim), np.float32)
        for i in range(n):
            v = (i - half_bins) / max(half_bins, 1)
            book[i] = _z_real(phi, float(np.clip(v, -1, 1)), scale)
        return book

    def _build_velocity_book(self, phi) -> np.ndarray:
        book = np.empty((self.n_v, self.dim), np.float32)
        log_den = np.log1p(V_MAX / V0)
        for i in range(self.n_v):
            s = 1.0 if i >= self.n_v // 2 else -1.0
            k = (i if i < self.n_v // 2 else i - self.n_v // 2)
            k = k / max(self.n_v // 2 - 1, 1)
            mag = V0 * (np.expm1(k * log_den))
            v = float(np.clip(s * mag, -V_MAX, V_MAX))
            book[i] = _z_real(phi, v, 1.0)
        return book

    def _bin_x(self, v_px) -> np.ndarray:
        vn = np.asarray(v_px, np.float64) / self.W
        idx = np.round(vn * self.W).astype(np.int64) + self.W
        return np.clip(idx, 0, 2 * self.W)

    def _bin_y(self, v_px) -> np.ndarray:
        vn = np.asarray(v_px, np.float64) / self.H
        idx = np.round(vn * self.H).astype(np.int64) + self.H
        return np.clip(idx, 0, 2 * self.H)

    def _bin_p(self, p, delta=False) -> np.ndarray:
        p = np.asarray(p, np.float64)
        if delta:
            idx = np.round(p).astype(np.int64) + 1
        else:
            idx = np.round(p).astype(np.int64) + 1
        return np.clip(idx, 0, 2)

    def _bin_v(self, v) -> np.ndarray:
        v = np.clip(np.asarray(v, np.float64), -V_MAX, V_MAX)
        s = np.sign(v)
        s[s == 0] = 1.0
        mag = np.abs(v)
        k = np.floor((self.n_v / 2) * np.log1p(mag / V0) / np.log1p(V_MAX / V0))
        k = np.clip(k, 0, self.n_v // 2 - 1).astype(np.int64)
        idx = np.where(s > 0, self.n_v // 2 + k, self.n_v // 2 - 1 - k)
        return np.clip(idx, 0, self.n_v - 1)

    def _time_indices(self, t_us) -> np.ndarray:
        n = np.round(np.asarray(t_us, np.float64) / self.q_t).astype(np.int64)
        return np.clip(n, 0, self.n_t_max - 1)

    def _z_time_batch(self, t_us) -> np.ndarray:
        """One batched ifft for all time lookups."""
        n = self._time_indices(t_us)
        if n.size == 0:
            return np.zeros((0, self.dim), np.float32)
        c, f = n // self.f_t, n % self.f_t
        Phi = self.PC[c] * self.PF[f]
        return np.real(np.fft.ifft(Phi, axis=1)).astype(np.float32)

    @staticmethod
    def _bundle(*parts) -> torch.Tensor:
        z = sum(parts)
        return F.normalize(z, p=2, dim=-1)

    def encode_nodes(self, x, y, t_us, p, t_window_start_us=0.0) -> torch.Tensor:
        """Node HV from x,y,t,p only — [N,D]."""
        t_rel = np.asarray(t_us, np.float64) - float(t_window_start_us)
        zx = self.z_x[self._bin_x(x)]
        zy = self.z_y[self._bin_y(y)]
        zp = self.z_p[self._bin_p(p, delta=False)]
        zt = self._z_time_batch(t_rel)
        z = torch.from_numpy(zx + zy + zp + zt)
        return self._bundle(z)

    def encode_edges_spatial(self, dx, dy, dt_sec) -> torch.Tensor:
        """Spatial edge HV — reuse x/y/t books; dt_sec -> µs for time book."""
        dt_us = np.asarray(dt_sec, np.float64) * 1e6
        zx = self.z_x[self._bin_x(dx)]
        zy = self.z_y[self._bin_y(dy)]
        zt = self._z_time_batch(dt_us)
        z = torch.from_numpy(zx + zy + zt)
        return self._bundle(z)

    def encode_edges_temporal(self, dx, dy, dt_sec, vx, vy, dp) -> torch.Tensor:
        """Temporal edge HV — adds vx/vy books (px/ms)."""
        dt_us = np.asarray(dt_sec, np.float64) * 1e6
        zx = self.z_x[self._bin_x(dx)]
        zy = self.z_y[self._bin_y(dy)]
        zt = self._z_time_batch(dt_us)
        zvx = self.z_vx[self._bin_v(vx)]
        zvy = self.z_vy[self._bin_v(vy)]
        zp = self.z_p[self._bin_p(dp, delta=True)]
        z = torch.from_numpy(zx + zy + zt + zvx + zvy + zp)
        return self._bundle(z)
