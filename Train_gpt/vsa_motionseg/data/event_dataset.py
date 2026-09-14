"""Generic event dataset interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class EventFrame:
    events: torch.Tensor | None
    image_height: int
    image_width: int
    timestamp: float
    motion_masks: torch.Tensor | None = None
    instance_masks: torch.Tensor | None = None
    depth: torch.Tensor | None = None
    camera_motion: Any | None = None
    object_motion: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class EventDatasetAdapter(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_frame(self, index: int) -> EventFrame: ...

    @property
    @abstractmethod
    def timestamps(self) -> list[float]: ...


def missing_gt(name: str) -> None:
    """Ground truth field absent — callers must check for None."""
