"""VSA prototype (nearest-centroid) segmentation head — cosine to class prototypes.

Parameter-free: each class prototype is the normalized bundle (mean) of the
training feature vectors of that class; a pixel is labeled by the most similar
prototype (cosine). Natural readout for bound hypervectors.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class PrototypeHead(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("prototypes", torch.zeros(0), persistent=True)  # (C, D) normalized

    @property
    def is_loaded(self) -> bool:
        return self.prototypes.numel() > 0

    def set_prototypes(self, protos: torch.Tensor) -> None:
        t = protos.detach().float().cpu()
        if self.prototypes.shape == t.shape:
            self.prototypes.copy_(t)
        else:
            self.register_buffer("prototypes", t, persistent=True)
        self.num_classes = int(t.shape[0])

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self.set_prototypes(ckpt["prototypes"])

    def logits_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (B, D, H, W) -> cosine logits (B, C, H, W)."""
        if not self.is_loaded:
            raise RuntimeError("PrototypeHead: fit/load prototypes before inference")
        b, dfeat, h, w = feats.shape
        x = feats.permute(0, 2, 3, 1).reshape(-1, dfeat)
        xn = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        p = self.prototypes.to(device=xn.device, dtype=xn.dtype)      # already normalized
        logits = xn @ p.t()                                          # (Npix, C)
        return logits.reshape(b, h, w, self.num_classes).permute(0, 3, 1, 2).to(
            device=feats.device, dtype=torch.float32)

    def num_readout_params(self) -> int:
        return int(self.prototypes.numel())


def fit_prototypes(
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
) -> torch.Tensor:
    """Class-mean prototypes (normalized) from flattened (x[N,D], y[N]) batches."""
    d = batches[0][0].shape[1]
    sums = torch.zeros(num_classes, d, dtype=torch.float64)
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for x, y in batches:
        x = x.to(torch.float64)
        y = y.long()
        for c in range(num_classes):
            m = y == c
            if m.any():
                sums[c] += x[m].sum(dim=0)
                counts[c] += int(m.sum())
    protos = sums / counts.clamp(min=1.0).unsqueeze(1)
    protos = protos / (protos.norm(dim=1, keepdim=True) + 1e-8)
    return protos.float()


def save_prototypes(path: str, protos: torch.Tensor, num_classes: int, extra: dict | None = None) -> None:
    payload = {"prototypes": protos, "num_classes": num_classes}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
