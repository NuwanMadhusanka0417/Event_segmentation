"""Single place that turns a config into a dataset.

train.py, eval.py and scripts/fit_ridge_head.py all used to build the dataset
themselves, so a new dataset option had to be added in three places (and the
copies drifted). They all call this now.
"""

from __future__ import annotations

from typing import Any

from hdems.data.dsec import DSECDataset
from hdems.data.evimo import EVIMODataset
from hdems.data.evimo2_reader import scenes_in_split
from hdems.data.labels import resolve_label_mode
from hdems.data.motion_labels import params_from_config


def build_dataset(cfg: dict[str, Any], split: str | None = None):
    """Dataset for ``split`` (default: dataset.split)."""
    ds_cfg = cfg.get("dataset", {})
    name = ds_cfg.get("name", "dsec")
    split = split or ds_cfg.get("split", "train")

    if name == "dsec":
        return DSECDataset(ds_cfg["root"], split)
    if name != "evimo":
        raise ValueError(f"Unknown dataset: {name}")

    train_split = ds_cfg.get("split", "train")
    eval_split = ds_cfg.get("eval_split", "eval")
    # Scene-disjoint splits: train/ and eval/ hold different takes of the SAME
    # scenes (scene13/14/15), so training on both leaks the eval scenes. Drop the
    # shared scenes from the TRAIN split and leave the eval set intact.
    exclude: set[str] = set()
    if ds_cfg.get("scene_disjoint", False) and split == train_split:
        exclude = scenes_in_split(ds_cfg["root"], eval_split)

    is_train = split == train_split
    return EVIMODataset(
        ds_cfg["root"],
        split,
        ds_cfg.get("version"),
        height=ds_cfg.get("height", 480),
        width=ds_cfg.get("width", 640),
        window_ms=ds_cfg.get("window_ms", 50.0),
        decay=cfg.get("time_surface", {}).get("decay", 0.8),
        time_frames=ds_cfg.get("time_frames"),
        label_mode=resolve_label_mode(cfg),
        motion_params=params_from_config(cfg),
        boundary_ignore_px=int(cfg.get("motion_label", {}).get("boundary_ignore_px", 2)),
        exclude_scenes=exclude,
        # Frames with no moving object are pure background: keep only a controlled
        # share of them as negatives while training, but evaluate on everything.
        require_mover=bool(ds_cfg.get("require_mover", False)) and is_train,
        negative_ratio=float(ds_cfg.get("negative_ratio", 0.0)),
        interleave=bool(ds_cfg.get("interleave", True)),
        # A frame counts as "has a mover" only if that object is actually VISIBLE:
        # objects move while out of frame, and such frames have no moving pixels.
        min_moving_px=int(cfg.get("motion_label", {}).get("min_moving_px", 100)),
    )
