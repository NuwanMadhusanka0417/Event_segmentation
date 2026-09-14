"""Static vs dynamic HDC readout."""

from __future__ import annotations

import torch

from vsa_motionseg.vsa.prototypes import load_prototypes, predict_prototypes, train_prototypes


class DynamicClassifier:
    def __init__(self, prototypes: torch.Tensor | None = None) -> None:
        self.prototypes = prototypes

    def fit(self, Q: torch.Tensor, labels: torch.Tensor, **kwargs) -> None:
        self.prototypes = train_prototypes(Q, labels, **kwargs)

    def predict(self, Q: torch.Tensor, threshold: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.prototypes is not None
        return predict_prototypes(Q, self.prototypes, threshold=threshold)

    @classmethod
    def from_file(cls, path: str) -> "DynamicClassifier":
        return cls(load_prototypes(path))
