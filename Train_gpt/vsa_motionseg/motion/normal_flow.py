"""Optional local plane-fit normal flow (not primary motion source)."""

from __future__ import annotations

import torch


def normal_flow_from_events(
    events: torch.Tensor,
    height: int,
    width: int,
    *,
    window_radius: int = 2,
) -> torch.Tensor:
    """
    Crude normal flow: per-pixel median event time gradient proxy.
    events: (N,4) [t,x,y,p]. Returns (2,H,W) — often ambiguous (aperture problem).
    """
    flow = torch.zeros(2, height, width)
    if events.numel() == 0:
        return flow
    # Placeholder: spread timestamps into vx, vy via local finite differences
    t = events[:, 0]
    x = events[:, 1].long().clamp(0, width - 1)
    y = events[:, 2].long().clamp(0, height - 1)
    acc = torch.zeros(height, width)
    acc.index_put_((y, x), t.to(acc.dtype), accumulate=True)
    count = torch.zeros(height, width)
    count.index_put_((y, x), torch.ones_like(t), accumulate=True)
    count = count.clamp_min(1)
    tm = acc / count
    flow[0, 1:, :] = (tm[1:, :] - tm[:-1, :])[: height - 1]
    flow[1, :, 1:] = (tm[:, 1:] - tm[:, :-1])[:, : width - 1]
    return flow
