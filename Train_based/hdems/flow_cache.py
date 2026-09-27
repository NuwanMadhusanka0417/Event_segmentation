"""Disk cache for the frozen VSA-Flow front end.

Why
---
The cost volume is 76-82% of a training step, yet it has no trainable parameters:
for a given time surface it returns the same flow every epoch. Recomputing it 30x
per sample is what made training take >24 h. This cache computes the flow ONCE
per sample and reuses it across epochs, validation, eval and later runs that
share the same front end (e.g. sweeping heads or combine modes).

Keys (content-addressed, so it can never serve stale flow)
--------------------------------------------------------
    <root>/<front-end key>/<surface key>.pt

  front-end key = hash of the encoder's actual kernel weights (so d, patch size,
                  sigma and seed are all covered) + cost-volume / estimator
                  settings (M, scales, alpha, smooth, vel_scale) + a version.
  surface key   = hash of the multi-time surface tensor itself (so window,
                  decay, time_frames and resolution are covered implicitly).

Change any of those and the lookup misses and recomputes. Flow is stored as fp16
(2 x H x W, ~1.2 MB per frame at 480x640) and ALWAYS returned via the same fp16
round trip, so a cache hit and a cache miss give bit-identical flow.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

# Bump when the flow computation itself changes (cost volume / estimator code).
FLOW_CACHE_VERSION = 2      # 2: zero-padded cost volume (was circular torch.roll)


def _sha1_tensor(t: torch.Tensor) -> str:
    a = t.detach().to("cpu").contiguous()
    h = hashlib.sha1()
    h.update(str(tuple(a.shape)).encode())
    h.update(str(a.dtype).encode())
    h.update(a.numpy().tobytes())
    return h.hexdigest()


class FlowCache:
    """Per-sample flow store for one front-end configuration."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._dir: Path | None = None
        self.hits = 0
        self.misses = 0

    def bind(self, model: Any) -> None:
        """Derive the front-end key from the model (once, after weights are loaded)."""
        if self._dir is not None:
            return
        enc = model.encoder
        settings = {
            "version": FLOW_CACHE_VERSION,
            "M": int(model.matcher.M),
            "scales": [int(s) for s in model.match_scales],
            "alpha": float(model.flow_alpha),
            "smooth": int(model.flow_smooth),
            "vel_scale": float(model.vel_scale),
            "d": int(enc.d),
            "patch_size": int(enc.patch_size),
            "kernel": getattr(enc, "kernel_mode", "window"),
            "polarity_binding": bool(getattr(enc, "polarity_binding", False)),
            "descriptor_scales": int(getattr(enc, "n_scales", 1)),
        }
        h = hashlib.sha1(json.dumps(settings, sort_keys=True).encode())
        # every encoder buffer (kernels for both polarities, role vectors) shapes the flow
        for name, buf in sorted(enc.state_dict().items()):
            h.update(name.encode())
            h.update(_sha1_tensor(buf).encode())
        self._dir = self.root / h.hexdigest()[:16]
        self._dir.mkdir(parents=True, exist_ok=True)
        info = self._dir / "front_end.json"
        if not info.exists():
            info.write_text(json.dumps(settings, indent=2))
        print(f"[flow-cache] {self._dir}  (M={settings['M']} alpha={settings['alpha']} "
              f"smooth={settings['smooth']} d={settings['d']} kernel={settings['kernel']})")

    def _path(self, surface: torch.Tensor) -> Path:
        assert self._dir is not None, "FlowCache.bind(model) must be called first"
        return self._dir / f"{_sha1_tensor(surface)}.pt"

    def load(self, surface: torch.Tensor) -> torch.Tensor | None:
        p = self._path(surface)
        if not p.exists():
            self.misses += 1
            return None
        try:
            flow = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:                      # truncated/corrupt file: recompute
            self.misses += 1
            return None
        self.hits += 1
        return flow

    def save(self, surface: torch.Tensor, flow: torch.Tensor) -> None:
        """Atomic write, so concurrent jobs never read a half-written file."""
        p = self._path(surface)
        if p.exists():
            return
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        os.close(fd)
        try:
            torch.save(flow.detach().to("cpu", torch.float16), tmp)
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def summary(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        return f"[flow-cache] hits {self.hits} / {total} ({rate:.0%})"
