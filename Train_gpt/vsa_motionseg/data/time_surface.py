"""Accumulative time surfaces with physical tau or legacy decay."""

from __future__ import annotations

import torch


class TimeSurfaceBuilder:
    """Incremental-friendly batch builder for event windows."""

    def __init__(
        self,
        height: int,
        width: int,
        *,
        tau_ms: float | None = None,
        legacy_decay: float = 0.8,
        use_pos: bool = True,
        use_neg: bool = True,
        normalize: bool = True,
    ) -> None:
        self.height = height
        self.width = width
        self.tau_ms = tau_ms
        self.legacy_decay = legacy_decay
        self.use_pos = use_pos
        self.use_neg = use_neg
        self.normalize = normalize

    def from_events(self, events: torch.Tensor, t_end: float | None = None) -> dict[str, torch.Tensor]:
        """
        events: (N, 4) [t, x, y, p] — t in same units as tau_ms (typically seconds or ms; consistent).
        """
        C = int(self.use_pos) + int(self.use_neg)
        if C == 0:
            C = 2
        surface = torch.zeros(C, self.height, self.width, dtype=torch.float32)
        count = torch.zeros(self.height, self.width, dtype=torch.float32)
        if events.numel() == 0:
            return self._pack(surface, count)

        t = events[:, 0].to(torch.float64)
        x = events[:, 1].long()
        y = events[:, 2].long()
        p = events[:, 3].long().clamp(0, 1)
        t_ref = float(t_end if t_end is not None else t.max().item())

        if self.tau_ms is not None and self.tau_ms > 0:
            val = torch.exp(-(t_ref - t) / self.tau_ms).to(torch.float32)
        else:
            t0 = t.min()
            val = (self.legacy_decay ** (t - t0)).to(torch.float32)

        inb = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        ch = []
        if self.use_pos:
            m = inb & (p == 1)
            s = torch.zeros(self.height, self.width)
            flat = (y[m] * self.width + x[m]).long()
            s.view(-1).index_add_(0, flat, val[m])
            ch.append(s)
        if self.use_neg:
            m = inb & (p == 0)
            s = torch.zeros(self.height, self.width)
            flat = (y[m] * self.width + x[m]).long()
            s.view(-1).index_add_(0, flat, val[m])
            ch.append(s)
        if len(ch) == 1:
            ch.append(torch.zeros_like(ch[0]))
        surface = torch.stack(ch, dim=0)

        flat_all = (y[inb] * self.width + x[inb]).long()
        count.view(-1).index_add_(0, flat_all, torch.ones(flat_all.numel()))
        return self._pack(surface, count)

    def _pack(self, surface: torch.Tensor, count: torch.Tensor) -> dict[str, torch.Tensor]:
        active = count > 0
        if self.normalize and surface.abs().max() > 0:
            surface = surface / (surface.abs().max() + 1e-8)
        return {
            "surface": surface,
            "event_count": count,
            "active_mask": active,
        }


def events_window(events: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
    t = events[:, 0]
    m = (t >= t0) & (t < t1)
    return events[m]
