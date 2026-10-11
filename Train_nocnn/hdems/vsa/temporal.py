"""Temporal binding and VSA prototype bundling for trajectory IDs."""

from __future__ import annotations

import torch

from .fpe import bind, bundle, fpe, make_base_phases, similarity, unbind


def make_time_phases(d: int, seed: int = 1) -> torch.Tensor:
    """FPE phases for temporal role binding."""
    return make_base_phases(d, seed=seed, kernel="gaussian", bandwidth=1.0)


def bind_trajectory(
    fields: torch.Tensor,
    time_phases: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Bind a sequence of descriptor fields with temporal role codes.

    Traj(x) = sum_t F^t(x) o T^t

    Parameters
    ----------
    fields      : (T, B, d, H, W) complex
    time_phases : (d,)
    times       : (T,) float, time indices or stamps

    Returns
    -------
    traj : (B, d, H, W) complex
    """
    T = fields.shape[0]
    bound = []
    for t in range(T):
        T_code = fpe(time_phases, times[t])
        bound.append(bind(fields[t], T_code.view(1, -1, 1, 1)))
    return bundle(torch.stack(bound, dim=0), dim=0)


def unbind_time(
    traj: torch.Tensor,
    time_phases: torch.Tensor,
    t: float | torch.Tensor,
) -> torch.Tensor:
    """Query descriptor at time t from trajectory hypervector.

    Returns
    -------
    F_t : (B, d, H, W) complex
    """
    T_code = fpe(time_phases, t)
    return unbind(traj, T_code.view(1, -1, 1, 1))


class PrototypeBank:
    """VSA prototype bundling for temporal object IDs (zero-parameter)."""

    def __init__(self, d: int, threshold: float = 0.6) -> None:
        self.d = d
        self.threshold = threshold
        self.prototypes: list[torch.Tensor] = []
        self.counts: list[int] = []

    def assign(self, hv: torch.Tensor) -> int:
        """Assign hypervector to nearest prototype or create new one.

        Parameters
        ----------
        hv : (d,) complex, single location trajectory HV

        Returns
        -------
        prototype_id : int
        """
        if not self.prototypes:
            self.prototypes.append(hv.clone())
            self.counts.append(1)
            return 0

        sims = torch.stack([similarity(hv, p) for p in self.prototypes])
        best = int(sims.argmax())
        if sims[best].item() >= self.threshold:
            # Running bundle (no normalisation until readout)
            self.prototypes[best] = self.prototypes[best] + hv
            self.counts[best] += 1
            return best

        self.prototypes.append(hv.clone())
        self.counts.append(1)
        return len(self.prototypes) - 1

    def reset(self) -> None:
        """Clear all prototypes."""
        self.prototypes.clear()
        self.counts.clear()
