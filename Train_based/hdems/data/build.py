"""Single place that turns a config into a dataset.

train.py, eval.py and scripts/fit_ridge_head.py all used to build the dataset
themselves, so a new dataset option had to be added in three places (and the
copies drifted). They all call this now.
"""

from __future__ import annotations

from typing import Any

from hdems.config import apply_resolution_ratio, surface_decay
from hdems.data.dsec import DSECDataset
from hdems.data.evimo import EVIMODataset
from hdems.data.evimo2_reader import scenes_in_split
from hdems.data.labels import resolve_label_mode
from hdems.data.motion_labels import params_from_config


# Name of the validation split made of held-out TRAIN scenes (dataset.val_scenes).
HOLDOUT = "holdout"


def val_split_of(cfg: dict[str, Any]) -> str:
    """Split used to pick the best epoch / ridge alpha: dataset.val_split.

    ``holdout`` = the train scenes listed in dataset.val_scenes (never trained on).
    ``eval``    = the test set itself (old; the reported eval scores are then
                  optimistic, since the model was selected on them).
    """
    ds = cfg.get("dataset", {}) or {}
    return str(ds.get("val_split", (cfg.get("ridge", {}) or {}).get("val_split", "eval")))


def build_dataset(cfg: dict[str, Any], split: str | None = None):
    """Dataset for ``split`` (default: dataset.split); ``holdout`` = validation scenes."""
    cfg = apply_resolution_ratio(cfg)          # no-op if the entry point already did
    ds_cfg = cfg.get("dataset", {})
    name = ds_cfg.get("name", "dsec")
    split = split or ds_cfg.get("split", "train")

    if name == "dsec":
        return DSECDataset(ds_cfg["root"], split)
    if name != "evimo":
        raise ValueError(f"Unknown dataset: {name}")

    train_split = ds_cfg.get("split", "train")
    eval_split = ds_cfg.get("eval_split", "eval")
    val_scenes = {str(s).lower() for s in (ds_cfg.get("val_scenes") or [])}
    holdout = val_split_of(cfg) == HOLDOUT
    if (split == HOLDOUT or holdout) and not val_scenes:
        raise ValueError("dataset.val_split is 'holdout' but dataset.val_scenes is empty")

    # Scene-disjoint splits: train/ and eval/ hold different takes of the SAME
    # scenes (scene13/14/15), so training on both leaks the eval scenes. Drop the
    # shared scenes from the TRAIN split and leave the eval set intact. The held-out
    # validation scenes are dropped from training as well.
    exclude: set[str] = set()
    only: set[str] | None = None
    folder = split
    if split == train_split:
        if ds_cfg.get("scene_disjoint", False):
            exclude = scenes_in_split(ds_cfg["root"], eval_split)
        if holdout:
            exclude |= val_scenes
    elif split == HOLDOUT:                     # validation = held-out TRAIN scenes
        folder, only = train_split, val_scenes

    is_train = split == train_split
    return EVIMODataset(
        ds_cfg["root"],
        folder,
        ds_cfg.get("version"),
        height=ds_cfg.get("height", 480),
        width=ds_cfg.get("width", 640),
        window_ms=ds_cfg.get("window_ms", 50.0),
        decay=surface_decay(cfg),              # time_surface.tau_ms (paper: 35 ms)
        time_frames=ds_cfg.get("time_frames"),
        label_mode=resolve_label_mode(cfg),
        motion_params=params_from_config(cfg),
        boundary_ignore_px=int(cfg.get("motion_label", {}).get("boundary_ignore_px", 2)),
        exclude_scenes=exclude,
        only_scenes=only,
        # Frames with no moving object are pure background: keep only a controlled
        # share of them as negatives while training, but evaluate on everything.
        require_mover=bool(ds_cfg.get("require_mover", False)) and is_train,
        negative_ratio=float(ds_cfg.get("negative_ratio", 0.0)),
        interleave=bool(ds_cfg.get("interleave", True)),
        # A frame counts as "has a mover" only if that object is actually VISIBLE:
        # objects move while out of frame, and such frames have no moving pixels.
        min_moving_px=int(cfg.get("motion_label", {}).get("min_moving_px", 100)),
        # train and score only on events this close to the label time (0 = all events)
        score_window_ms=ds_cfg.get("score_window_ms"),
        use_shards=bool(ds_cfg.get("use_shards", False)),
    )
