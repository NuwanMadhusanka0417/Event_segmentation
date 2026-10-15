"""Training-free segmentation: threshold ONE motion cue of the front end, no head.

Why: the oracle ladder's O3 -- how much of the score is the front end and how much
the trained U-Net adds. If thresholding the residual already gets close, the head
is not where accuracy is lost; if it is far below, the head matters.

Cues (all in SENSOR px per flow interval, so one threshold means the same at every
resolution ratio):
  residual   |flow − background|: rotation (IMU) + ego fit removed -- what the head sees
  derotated  |flow − rotation|: only the IMU rotation removed (no ego fit)
  foe        depth-free FOE residual (hdems/foe.py): perpendicular + wrong-sign part
  perp       the perpendicular part alone

The threshold is a displacement, tuned on the VALIDATION takes with the benchmark
metric (hdems.eval --baseline), never on the test set.
"""

from __future__ import annotations

from typing import Any

import torch

from hdems.foe import FoeParams, foe_cue_maps

BASELINE_CUES = ("residual", "derotated", "foe", "perp")
# sensor px per 12.5 ms; a quarter-resolution smoke test chose the old lowest value (0.25)
THRESHOLD_GRID_PX = (0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0,
                     5.0, 6.0, 8.0)


class CueBaseline(torch.nn.Module):
    """Wraps an HDEMS front end; ``forward`` returns out["moving_score"] (B, 1, H, W) =
    the cue in sensor px, which hdems.protocols.moving_prob uses instead of a softmax,
    so the benchmark thresholds it at ``decision_threshold`` (px)."""

    def __init__(self, model: Any, cue: str, threshold: float | None = None,
                 foe: FoeParams | None = None) -> None:
        super().__init__()
        if cue not in BASELINE_CUES:
            raise ValueError(f"baseline cue must be one of {BASELINE_CUES}, got {cue!r}")
        self.model = model
        self.cue = cue
        self.decision_threshold = threshold
        self.temporal_beta = 0.0
        self.foe_params = foe or FoeParams()
        # the cue does not use the VSA consistency maps: skip their 3 extra encodes
        model.vsa_consistency = False

    @property
    def flow_cache(self):
        return self.model.flow_cache

    @property
    def name(self) -> str:
        return f"baseline:{self.cue}"

    def residual_from_flow(self, *args, **kwargs):
        return self.model.residual_from_flow(*args, **kwargs)

    @torch.no_grad()
    def forward(self, surface: torch.Tensor, *, task: str = "segmentation",
                rot_flow: torch.Tensor | None = None, **oracle: torch.Tensor | None) -> dict:
        m = self.model.motion_features(surface, rot_flow, **oracle)
        r = float(self.model.res_ratio)
        if self.cue == "residual":
            score = m["residual"].norm(dim=1, keepdim=True)
        elif self.cue == "derotated":
            rot = m["rot"]
            score = (m["flow"] if rot is None else m["flow"] - rot).norm(dim=1, keepdim=True)
        else:
            maps, _ = foe_cue_maps(m["flow"], m["rot"], self.model.ego_mask(surface),
                                   self.foe_params, r)
            score = maps[self.cue]
        return {"flow": m["flow"], "residual": m["residual"], "moving_score": score * r}
