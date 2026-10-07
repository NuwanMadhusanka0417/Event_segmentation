"""Published-protocol benchmark (hdems.protocols) and the position-keyed GT masks."""

from pathlib import Path

import numpy as np
import pytest
import torch

from hdems import protocols as bm
from hdems.data.evimo import EVIMODataset
from hdems.data.evimo2_reader import build_sample_index, gt_key, load_frame_sample, load_meta
from hdems.data.motion_labels import MotionParams

H, W = 16, 20
FX = 100.0


def _pose(x: float, z: float = 1.0) -> dict:
    return {"pos": {"q": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "t": {"x": x, "y": 0.0, "z": z}}}


def _fake_sequence(root: Path, name: str = "scene13_dyn_test_00_000000", n_frames: int = 13,
                   first_id: int = 10) -> Path:
    """Still camera; object 1 (top-left) moves at 1 m/s (5 px / 50 ms), object 2
    (bottom-right) is static. Frame ids start at ``first_id`` but the masks are keyed
    by POSITION -- as in the real EVIMO2 export."""
    seq = root / "eval" / name
    seq.mkdir(parents=True)
    frames, masks = [], {}
    for i in range(n_frames):
        ts = 0.2 + i / 60.0
        frames.append({"id": first_id + i, "ts": ts, "cam": _pose(0.0, 0.0),
                       "1": _pose(1.0 * ts), "2": _pose(0.3)})
        m = np.zeros((H, W), np.int32)
        m[0:8, 0:10] = 1000
        m[8:16, 10:20] = 2000
        masks[gt_key("mask", i)] = m
    meta = {"frames": frames, "meta": {"fx": FX, "fy": FX, "cx": W / 2, "cy": H / 2,
                                       "res_x": W, "res_y": H}}
    np.savez(seq / "dataset_info.npz", meta=np.array(meta, dtype=object))
    np.savez_compressed(seq / "dataset_mask.npz", **masks)
    # every pixel fires once every 5 ms from t = 0 to 1 s (same count everywhere)
    steps = np.arange(0.0, 1.0, 0.005)
    ys, xs = np.mgrid[0:H, 0:W]
    xy = np.stack([xs.ravel(), ys.ravel()], 1)
    np.save(seq / "dataset_events_t.npy", np.repeat(steps, H * W))
    np.save(seq / "dataset_events_xy.npy", np.tile(xy, (len(steps), 1)).astype(np.int16))
    np.save(seq / "dataset_events_p.npy", np.ones(len(steps) * H * W, np.int8))
    return seq


class _FixedModel(torch.nn.Module):
    """Predicts 'moving' on a fixed pixel set, whatever the input."""

    def __init__(self, pred: np.ndarray) -> None:
        super().__init__()
        self.pred = torch.from_numpy(pred)
        self.flow_cache = None

    def forward(self, surface, task="segmentation"):
        logits = torch.zeros(1, 2, *self.pred.shape)
        logits[0, 1][self.pred] = 1.0
        return {"seg_logits": logits}


def _dataset(root: Path) -> EVIMODataset:
    return EVIMODataset(root, "eval", height=H, width=W, window_ms=50.0,
                        motion_params=MotionParams(window_s=0.05))


def test_masks_are_keyed_by_position(tmp_path):
    seq = _fake_sequence(tmp_path)
    index = build_sample_index(tmp_path, "eval", window_s=0.05)
    assert [fi for _, fi in index] == list(range(13))        # ids 10..22 would find none
    meta = load_meta(seq)
    s = load_frame_sample(seq, meta["frames"][3], window_s=0.05)   # index looked up
    assert int(s["mask"][0, 0]) == 1000 and int(s["mask"][-1, -1]) == 2000


def test_event_iou_counts_every_event():
    pred = np.zeros((2, 2), bool)
    gt = np.zeros((2, 2), bool)
    pred[0, 0] = gt[0, 0] = True          # TP pixel
    pred[0, 1] = True                     # FP pixel
    gt[1, 0] = True                       # FN pixel
    # TP pixel fires 3x, FP 1x, FN 2x, background 5x
    x = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    inter, union, g, p = bm.event_iou_counts(pred, gt, x, y)
    assert (inter, union, g, p) == (3, 6, 5, 4)          # pixel IoU would be 1/3, events 1/2


def test_gt_modes():
    obj = np.array([[0, 1, 2, 3]])
    assert bm.gt_moving_pixels(obj, {1}, {2}, "moving").tolist() == [[False, True, False, False]]
    assert bm.gt_moving_pixels(obj, {1}, {2}, "moving_slow").tolist() == [[False, True, True, False]]
    assert bm.gt_moving_pixels(obj, {1}, {2}, "tracked").tolist() == [[False, True, True, True]]
    assert not bm.gt_moving_pixels(obj, set(), set(), "moving").any()
    with pytest.raises(ValueError):
        bm.gt_moving_pixels(obj, {1}, set(), "dynamic")


def test_settings_overrides():
    s = bm.settings({"benchmark": {"gt": "tracked", "event_windows_ms": 25}})
    assert s["gt"] == "tracked" and s["event_windows_ms"] == [25.0] and s["pred"] == "fg"
    s = bm.settings({}, gt="moving_slow", pred="objects", windows_ms=[10, 20])
    assert (s["gt"], s["pred"], s["event_windows_ms"]) == ("moving_slow", "objects", [10.0, 20.0])
    with pytest.raises(ValueError):
        bm.settings({}, gt="nope")


def test_benchmark_end_to_end(tmp_path):
    _fake_sequence(tmp_path)
    ds = _dataset(tmp_path)
    pred = np.zeros((H, W), bool)
    pred[0:8, 0:5] = True                 # half of the mover (40 px, all correct)
    pred[8:16, 10:15] = True              # 40 px on the STATIC object (false positive)
    model = _FixedModel(pred)

    scores, stats = bm.run_benchmark(model, ds, bm.HUA2025, torch.device("cpu"),
                                     windows_ms=[12.5, 50.0], gt_mode="moving")
    assert stats["run"] == 13 and stats["missing"] == ["13-05", "14-03", "14-04", "14-05"]
    for sc in scores:                     # uniform events -> event IoU = pixel IoU = 40/120
        assert sc.per_sequence()["13-00"] == pytest.approx(100 / 3)
        assert len(sc.frames["13-00"]) == 13

    scores, _ = bm.run_benchmark(model, ds, bm.HUA2025, torch.device("cpu"),
                                 windows_ms=[12.5], gt_mode="tracked")
    assert scores[0].per_sequence()["13-00"] == pytest.approx(50.0)   # 80 / 160

    report = bm.format_report(bm.HUA2025, scores, stats, gt_mode="tracked", pred_source="fg")
    assert "13-05" in report and "n/a" in report and "MISSING" in report
    assert "82.15" in report and "mean 1/5" in report


def test_benchmark_skips_static_frames_and_caps(tmp_path):
    seq = _fake_sequence(tmp_path)
    ds = _dataset(tmp_path)
    model = _FixedModel(np.ones((H, W), bool))
    # an IoU only over frames WITH a mover: with no moving object the frame is skipped
    scores, stats = bm.run_benchmark(model, ds, bm.HUA2025, torch.device("cpu"),
                                     windows_ms=[12.5], max_per_sequence=4)
    assert stats["candidates"] == 13 and stats["run"] == 4      # cap applies to movers
    meta = load_meta(seq)
    for f in meta["frames"]:
        f["1"] = _pose(0.0)               # nothing moves any more
    np.savez(seq / "dataset_info.npz", meta=np.array(meta, dtype=object))
    from hdems.data import motion_labels
    motion_labels._CACHE.clear()
    scores, stats = bm.run_benchmark(model, _dataset(tmp_path), bm.HUA2025,
                                     torch.device("cpu"), windows_ms=[12.5])
    assert stats["run"] == 0 and stats["no_gt_pixels"] == 13 and not scores[0].frames


def test_save_json(tmp_path):
    sc = bm.ProtocolScores(12.5)
    sc.add("13-00", 1, 2, frame_index=0)
    out = tmp_path / "b.json"
    bm.save_json(out, bm.HUA2025, [sc], {"run": 1}, {"gt": "moving"})
    import json
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["windows"][0]["per_sequence"]["13-00"] == pytest.approx(50.0)
    assert doc["paper"]["13-00"] == 82.15
