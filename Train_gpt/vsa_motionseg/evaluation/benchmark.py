"""Per-stage runtime and memory reporting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchmarkReport:
    stages: dict[str, float] = field(default_factory=dict)
    peak_memory_mb: float = 0.0

    def format(self) -> str:
        lines = [f"{k}: {v:.4f}s" for k, v in self.stages.items()]
        total = sum(self.stages.values())
        lines.append(f"Total per window: {total:.4f}s")
        lines.append(f"Peak memory: {self.peak_memory_mb:.1f} MB")
        return "\n".join(lines)


class StageTimer:
    def __init__(self) -> None:
        self.report = BenchmarkReport()

    def run(self, name: str, fn: Callable):
        t0 = time.perf_counter()
        out = fn()
        self.report.stages[name] = time.perf_counter() - t0
        try:
            import torch

            if torch.cuda.is_available():
                self.report.peak_memory_mb = max(
                    self.report.peak_memory_mb, torch.cuda.max_memory_allocated() / 1e6
                )
        except Exception:
            pass
        return out
