"""Shared CLI helpers for entry-point scripts."""

from __future__ import annotations

from typing import Any


def apply_device_override(cfg: dict[str, Any], device: str | None) -> None:
    if device:
        cfg.setdefault("runtime", {})["device"] = device


def resolve_max_frames(
    cli_value: int | None,
    cfg: dict[str, Any],
    config_key: str,
    default: int | None,
) -> int | None:
    if cli_value is not None:
        return cli_value
    ds = cfg.get("dataset", {})
    if config_key in ds and ds[config_key] is not None:
        return int(ds[config_key])
    return default


def resolve_max_samples(
    cli_value: int | None,
    cfg: dict[str, Any],
    config_key: str = "max_train_samples",
) -> int | None:
    if cli_value is not None:
        return cli_value if cli_value > 0 else None
    ds = cfg.get("dataset", {})
    if config_key in ds and ds[config_key] is not None:
        v = int(ds[config_key])
        return v if v > 0 else None
    return None
