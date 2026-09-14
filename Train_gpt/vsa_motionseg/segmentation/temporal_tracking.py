"""Temporal segment matching and ID assignment."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from scipy.optimize import linear_sum_assignment


@dataclass
class SegmentTrack:
    track_id: int
    hypervector: torch.Tensor
    mean_flow: torch.Tensor
    centroid: torch.Tensor
    area: int
    mask: torch.Tensor
    confidence: float
    last_seen: int = 0
    age: int = 0


@dataclass
class TemporalTracker:
    max_track_gap: int = 3
    min_iou: float = 0.2
    flow_weight: float = 1.0
    hv_weight: float = 0.5
    centroid_weight: float = 0.1
    next_id: int = 1
    tracks: list[SegmentTrack] = field(default_factory=list)

    def update(self, segments: list[dict], t_index: int) -> dict[int, int]:
        """Map local segment index -> stable track id."""
        if not segments:
            self._age_tracks(t_index)
            return {}
        cost = torch.zeros(len(self.tracks), len(segments))
        for i, tr in enumerate(self.tracks):
            for j, seg in enumerate(segments):
                cost[i, j] = self._match_cost(tr, seg)
        mapping: dict[int, int] = {}
        if self.tracks:
            row, col = linear_sum_assignment(cost.numpy())
            for r, c in zip(row, col):
                if cost[r, c] < 10.0:
                    mapping[c] = self.tracks[r].track_id
                    self.tracks[r] = self._track_from_seg(segments[c], self.tracks[r].track_id, t_index)
        for j, seg in enumerate(segments):
            if j not in mapping:
                tid = self.next_id
                self.next_id += 1
                mapping[j] = tid
                self.tracks.append(self._track_from_seg(seg, tid, t_index))
        self._age_tracks(t_index)
        return mapping

    def _match_cost(self, tr: SegmentTrack, seg: dict) -> float:
        hv_d = 1 - (tr.hypervector.conj() * seg["hypervector"]).sum().real.item()
        fd = (tr.mean_flow - seg["mean_flow"]).norm().item()
        iou = _mask_iou(tr.mask, seg["mask"])
        cd = (tr.centroid - seg["centroid"]).norm().item()
        return self.hv_weight * hv_d + self.flow_weight * fd + (1 - iou) + self.centroid_weight * cd

    def _track_from_seg(self, seg: dict, tid: int, t: int) -> SegmentTrack:
        return SegmentTrack(
            track_id=tid,
            hypervector=seg["hypervector"],
            mean_flow=seg["mean_flow"],
            centroid=seg["centroid"],
            area=seg["area"],
            mask=seg["mask"],
            confidence=seg["confidence"],
            last_seen=t,
            age=0,
        )

    def _age_tracks(self, t: int) -> None:
        alive = []
        for tr in self.tracks:
            if t - tr.last_seen <= self.max_track_gap:
                alive.append(tr)
        self.tracks = alive


def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().float()
    union = (a | b).sum().float()
    if union <= 0:
        return 0.0
    return (inter / union).item()


def segment_summary(
    label_map: torch.Tensor,
    Q: torch.Tensor,
    residual_flow: torch.Tensor,
    confidence: torch.Tensor,
) -> list[dict]:
    """Build segment dicts from integer label map."""
    H, W = label_map.shape
    segs = []
    for lid in label_map.unique().tolist():
        if lid == 0:
            continue
        m = label_map == lid
        q = Q[:, m].sum(dim=1)
        q = torch.sign(q.real)
        yz = m.nonzero(as_tuple=False)
        segs.append(
            {
                "hypervector": q,
                "mean_flow": residual_flow[:, m].mean(dim=1),
                "centroid": torch.stack([yz[:, 0].float().mean(), yz[:, 1].float().mean()]),
                "area": int(m.sum().item()),
                "mask": m,
                "confidence": float(confidence[m].mean().item()),
            }
        )
    return segs
