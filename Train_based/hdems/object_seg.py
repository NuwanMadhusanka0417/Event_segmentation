"""Object-level moving / static decisions from the instance embedding (head mfunet with
segmentation.instance_dim > 0). Head-agnostic otherwise: needs P(moving) and an embedding.

Why (2026-10-22, scripts/check_object_oracle.py, 5 test sequences): with the PERFECT motion
of each event, deciding per EVENT scores 80.2% event IoU (14-05: 57.6), deciding per OBJECT
90.1% (14-05: 80.0) -- above Hua et al. (77.4) on every sequence. The ground truth marks
whole objects as moving (pose speed over ~167 ms), so slow parts of a moving object, and
moments where it barely moves inside the 12.5 ms window, are lost by per-pixel decisions.

  objects   cluster the embeddings of the scoring pixels (seeded greedy clustering with a
            bandwidth, as in De Brabandere et al. 2017), then split every cluster into
            spatially connected parts (link_px): equal-looking objects far apart stay apart
  decision  an object is moving if the mean P(moving) over its pixels > threshold
  memory    (optional) an object that overlaps a recently moving object keeps memory x
            that object's score: the GT calls an object moving from its speed over ~167 ms,
            so an object that pauses for a frame is still "moving" (oracle: 14-05 80.0 -> 89.0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ObjectParams:
    bandwidth: float = 1.0        # embedding distance joining a pixel to a cluster (delta_d of the loss: 1.5)
    link_px: int = 2              # spatial split: one object = pixels linked within this (working px)
    min_px: int = 3               # smaller objects keep their own (noisy) decision
    threshold: float = 0.5        # moving if mean P(moving) over the object > this
    memory: float = 0.0           # 0 = off; else carried score = memory x previous object's score
    memory_dilate_px: int = 3     # previous objects are matched within this (working px)
    max_clusters: int = 64
    seed: int = 0


def cluster_embeddings(emb: torch.Tensor, mask: torch.Tensor, p: ObjectParams) -> np.ndarray:
    """emb (E, H, W), mask (H, W) bool -> (H, W) int64 cluster id + 1 (0 = not a mask pixel).

    Seeded greedy clustering: a random unassigned pixel seeds a cluster, its centre moves to
    the mean of the unassigned pixels within ``bandwidth`` (a few mean-shift steps), those
    pixels form the cluster; repeat. Pixels left after max_clusters join the nearest centre.
    """
    H, W = mask.shape
    out = np.zeros((H, W), np.int64)
    idx = torch.nonzero(mask.reshape(-1)).squeeze(1)
    if idx.numel() == 0:
        return out
    e = emb.reshape(emb.shape[0], -1)[:, idx].t().float()          # (N, E)
    N = e.shape[0]
    lab = torch.full((N,), -1, dtype=torch.long, device=e.device)
    gen = torch.Generator(device="cpu").manual_seed(p.seed)
    centres = []
    while (lab < 0).any() and len(centres) < p.max_clusters:
        free = torch.nonzero(lab < 0).squeeze(1)
        c = e[free[int(torch.randint(len(free), (1,), generator=gen))]]
        for _ in range(5):
            near = free[(e[free] - c).norm(dim=1) < p.bandwidth]
            if near.numel() == 0:
                break
            c_new = e[near].mean(0)
            if (c_new - c).norm() < 1e-3:
                c = c_new
                break
            c = c_new
        near = free[(e[free] - c).norm(dim=1) < p.bandwidth]
        if near.numel() == 0:                       # an outlier seed: a cluster of its own
            near = free[(e[free] - c).norm(dim=1).argmin()].reshape(1)
        lab[near] = len(centres)
        centres.append(c)
    rest = lab < 0
    if rest.any():
        C = torch.stack(centres)
        lab[rest] = torch.cdist(e[rest], C).argmin(1)
    flat = out.reshape(-1)
    flat[idx.cpu().numpy()] = lab.cpu().numpy() + 1
    return out


def split_spatially(clusters: np.ndarray, link_px: int) -> np.ndarray:
    """Every cluster -> its spatially connected parts (pixels linked within link_px).
    -> (H, W) int64 object id (1..K, 0 = none)."""
    from scipy.ndimage import binary_dilation, label
    out = np.zeros_like(clusters)
    k = 0
    st = np.ones((3, 3), bool)
    for c in np.unique(clusters[clusters > 0]):
        m = clusters == c
        grown = binary_dilation(m, structure=st, iterations=link_px) if link_px > 0 else m
        comp, n = label(grown, structure=st)
        out[m] = comp[m] + k
        k += n
    if k:                                            # compact ids
        _u, inv = np.unique(out, return_inverse=True)
        out = inv.reshape(out.shape) if (out == 0).any() else inv.reshape(out.shape) + 1
    return out


def object_scores(prob: np.ndarray, objects: np.ndarray) -> np.ndarray:
    """Mean P(moving) per object id -> (K + 1,) with [0] = 0 (no object)."""
    K = int(objects.max())
    m = objects > 0
    s = np.bincount(objects[m], prob[m], K + 1)
    n = np.bincount(objects[m], minlength=K + 1)
    out = s / np.maximum(n, 1)
    out[0] = 0.0
    return out


class ObjectMemory:
    """Per-sequence memory of object scores (causal: past frames only). An object takes
    max(own score, memory x the median, over its pixels, of the previous frame's object
    scores grown by memory_dilate_px) -- the previous objects it overlaps."""

    def __init__(self, p: ObjectParams) -> None:
        self.p = p
        self.prev: dict[str, np.ndarray] = {}        # sequence -> previous score map (H, W)

    def __call__(self, seq: str, objects: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if self.p.memory <= 0:
            return scores
        from scipy.ndimage import maximum_filter
        out = scores.copy()
        prev = self.prev.get(seq)
        if prev is not None and prev.shape == objects.shape:
            grown = maximum_filter(prev, size=2 * self.p.memory_dilate_px + 1)
            for i in range(1, len(scores)):
                m = objects == i
                if m.any():
                    out[i] = max(scores[i], self.p.memory * float(np.median(grown[m])))
        self.prev[seq] = np.where(objects > 0, out[objects], 0.0)
        return out


def segment_objects(prob: torch.Tensor, emb: torch.Tensor, mask: torch.Tensor, p: ObjectParams,
                    memory: ObjectMemory | None = None, seq: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(moving) (H, W), embedding (E, H, W), scoring pixels (H, W) bool ->
    (moving (H, W) bool, objects (H, W) int, object scores (K + 1,))."""
    objects = split_spatially(cluster_embeddings(emb, mask, p), p.link_px)
    pr = prob.detach().float().cpu().numpy()
    scores = object_scores(pr, objects)
    if memory is not None:
        scores = memory(seq, objects, scores)
    sizes = np.bincount(objects.reshape(-1), minlength=len(scores))
    moving_obj = scores > p.threshold
    moving = moving_obj[objects] & (objects > 0)
    small = (sizes < p.min_px)[objects] & (objects > 0)       # tiny objects: per-pixel decision
    moving = np.where(small, pr > p.threshold, moving)
    return moving, objects, scores
