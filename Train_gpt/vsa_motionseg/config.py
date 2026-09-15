"""Load and merge YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import torch


def resolve_device(spec: str | None = None) -> torch.device:
    """Map config/CLI device string to ``torch.device`` (falls back to CPU)."""
    if not spec or spec == "cpu":
        return torch.device("cpu")
    if spec.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(spec)
        return torch.device("cpu")
    return torch.device(spec)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    default_path = path.parent / "default.yaml"
    if default_path.exists() and path.name != "default.yaml":
        with default_path.open("r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        cfg = _deep_merge(base, cfg)
    return cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
