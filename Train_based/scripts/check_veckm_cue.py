"""Pre-test: is VecKM_flow's per-event NORMAL flow (Yuan et al., ICCV 2025 -- the flow
Hua et al. 2025 segment with) a better motion cue than our VSA flow?
Training-free, VALIDATION takes only.

Per frame, VecKM_flow (pretrained, github.com/dhyuan99/VecKM_flow, MIT) runs on the events
of one of its time slices ending at the label time; the events of the benchmark window
(12.5 ms) are scored. Cues per EVENT, all in sensor px per 12.5 ms:

  veckm |n|          raw normal-flow speed
  veckm - IMU        | s - rot . n0 |      s = normal speed, n0 = its direction, rot = IMU
                                           rotational flow (what a static point would do)
  veckm - our bg     | s - bg . n0 |       bg = our background (IMU + ego fit + fallback)
  vsa - IMU          | flow - rot |        our current cues, read at the event's pixel
  vsa residual       | flow - bg |
  vsa residual . n0  | (flow - bg) . n0 |  our residual projected on VecKM's normals:
                                           separates "normal vs full flow" from "their
                                           flow vs ours"

Also, where EVIMO2 has depth, both flows' NORMAL component against the ground truth
(gt_flow . n0): correlation and slope -- a direct flow-quality comparison.

Caveat: VecKM_flow's released model was trained on EVIMO2-imo's TRAIN split, which holds
these validation takes, so its numbers here are optimistic (the test takes are clean).

    python scripts/check_veckm_cue.py --veckm-dir <path to the unzipped VecKM_flow repo> \\
        --max-frames 60 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdems.baselines import THRESHOLD_GRID_PX  # noqa: E402
from hdems.config import apply_resolution_ratio  # noqa: E402
from hdems.data.build import HOLDOUT, build_dataset  # noqa: E402
from hdems.data.gt_flow import gt_flow_for_sample  # noqa: E402
from hdems.eval import load_config  # noqa: E402
from hdems.models.hdems import HDEMS  # noqa: E402
from hdems.protocols import HUA2025, GTFrames, _spread_cap  # noqa: E402
from hdems.seg_features import motion_kwargs  # noqa: E402

CUES = ("veckm |n|", "veckm - IMU", "veckm - our bg", "vsa - IMU", "vsa residual", "vsa residual . n0")
INTERVAL_S = 0.0125


def undistort_radtan(x: np.ndarray, y: np.ndarray, mm: dict, iters: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Pixel -> undistorted NORMALISED coordinates (focal length 1), radial-tangential model
    (the fixed-point inversion cv2.undistortPoints uses)."""
    fx, fy, cx, cy = (float(mm[k]) for k in ("fx", "fy", "cx", "cy"))
    k1, k2, k3 = (float(mm.get(k, 0.0)) for k in ("k1", "k2", "k3"))
    p1, p2 = float(mm.get("p1", 0.0)), float(mm.get("p2", 0.0))
    xd, yd = (x - cx) / fx, (y - cy) / fy
    xu, yu = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = xu * xu + yu * yu
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        dx = 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu)
        dy = p1 * (r2 + 2 * yu * yu) + 2 * p2 * xu * yu
        xu, yu = (xd - dx) / rad, (yd - dy) / rad
    return xu, yu


def load_veckm(repo: Path, training_set: str):
    """Their NormalEstimator with the released weights, loaded from the file directly
    (their package reads it through importlib.resources, which needs an __init__.py
    the models folder does not have on Python < 3.10). Same weights_only load."""
    from types import SimpleNamespace
    sys.path.insert(0, str(repo.resolve()))
    from VecKM_flow import params as vk_params                      # noqa: E402
    from VecKM_flow.estimator import NormalEstimator                # noqa: E402
    P = getattr(vk_params, f"{training_set}Params")
    net = NormalEstimator(P.d, P.alpha)
    net.load_state_dict(torch.load(repo / "VecKM_flow" / "models" / f"{training_set}.pth",
                                   map_location="cpu", weights_only=True))
    net.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        net = net.cuda()
    return SimpleNamespace(P=P, estimator=net, cuda_available=cuda, auto_scale_time=True)


def veckm_normal_flow(est, t: np.ndarray, xu: np.ndarray, yu: np.ndarray, ensemble: int):
    """VecKM_flow on ONE slice of events -> (normal flow (n, 2) normalised units / s,
    uncertainty (n,)), following VecKMNormalFlowEstimator.inference for a single step."""
    P = est.P
    tt = torch.from_numpy((t - t.min()) / P.t_radius).float()
    xy = torch.from_numpy(np.stack([xu, yu], 1) / P.pxl_radius).float()
    scale = float(np.clip(len(t) / 50000.0, 0.33, 1.33)) if est.auto_scale_time else 1.0
    txy = torch.cat([tt[:, None] * scale, xy], 1)
    if est.cuda_available:
        txy = txy.cuda()
    with torch.no_grad():
        pred, unc = est.estimator.inference(txy, ensemble=ensemble)
    return (pred * scale).numpy(), unc.numpy()


def frame_cues(k, ds, gtf, model, est, dev, slice_s: float, ensemble: int):
    """One frame -> ({cue: (n,) per event}, labels (n,) bool, (gt . n0, veckm, vsa . n0) or None,
    VecKM seconds). Events = the benchmark window (12.5 ms before the label time)."""
    r = model.res_ratio
    s = ds[k]
    seq_dir, _fi = ds.index[k]
    gt, _evs, fi, ts = gtf(k)
    meta = gtf.metas[Path(seq_dir)]
    mm = meta["meta"]
    stream = gtf.streams[Path(seq_dir)]
    # events of one VecKM step ending at the label time (time scale as in their code)
    i1 = int(np.searchsorted(stream.t, ts, side="right"))
    i0 = int(np.searchsorted(stream.t, ts - slice_s, side="left"))
    scale = float(np.clip((i1 - i0) / 50000.0, 0.33, 1.33))
    i0 = int(np.searchsorted(stream.t, ts - slice_s / scale, side="left"))
    t = np.asarray(stream.t[i0:i1], dtype=np.float64)
    xy = np.asarray(stream.xy[i0:i1]).astype(np.float64).reshape(-1, 2)
    H, W = gt.shape
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
    t, xy = t[ok], xy[ok]
    xu, yu = undistort_radtan(xy[:, 0], xy[:, 1], mm)
    t0 = time.perf_counter()
    nf, _unc = veckm_normal_flow(est, t, xu, yu, ensemble)
    secs = time.perf_counter() - t0

    sc = t >= ts - INTERVAL_S                                   # the benchmark window
    x_i, y_i = xy[sc, 0].astype(int), xy[sc, 1].astype(int)
    f = 0.5 * (float(mm["fx"]) + float(mm["fy"]))
    nvec = -nf[sc] * f * INTERVAL_S                             # sensor px per interval, toward EARLIER
    spd = np.linalg.norm(nvec, axis=1)
    n0 = nvec / np.maximum(spd[:, None], 1e-9)

    surf = s["surface"].unsqueeze(0).to(dev)
    flow, _conf = model.compute_motion(surf)
    rot = model._rot(motion_kwargs(s, dev).get("rot_flow"), flow)
    _res, bg = model.background_fit(flow, surf, rot)

    def up(a):
        return F.interpolate(a.float(), size=(H, W), mode="nearest")[0].cpu().numpy() * r

    flow_f, bg_f = up(flow), up(bg)
    rot_f = up(rot) if rot is not None else np.zeros_like(flow_f)
    fl, bgv, rv = flow_f[:, y_i, x_i].T, bg_f[:, y_i, x_i].T, rot_f[:, y_i, x_i].T   # (n, 2)
    cue = {
        "veckm |n|": spd,
        "veckm - IMU": np.abs(spd - (rv * n0).sum(1)),
        "veckm - our bg": np.abs(spd - (bgv * n0).sum(1)),
        "vsa - IMU": np.linalg.norm(fl - rv, axis=1),
        "vsa residual": np.linalg.norm(fl - bgv, axis=1),
        "vsa residual . n0": np.abs(((fl - bgv) * n0).sum(1)),
    }
    y = gt[y_i, x_i]
    cmp = None
    g = gt_flow_for_sample(Path(seq_dir), fi, ds.window_s, ds.time_frames, meta)
    if g is not None:
        gv = g[:, y_i, x_i].T
        okg = np.isfinite(gv).all(1) & (spd > 1e-6)
        cmp = ((gv[okg] * n0[okg]).sum(1), spd[okg], (fl[okg] * n0[okg]).sum(1))
    return cue, y, cmp, secs


def frame_iou(p: np.ndarray, g: np.ndarray, thr: float) -> float:
    pr = p > thr
    u = int((pr | g).sum())
    return int((pr & g).sum()) / u if u else float("nan")


def build(cfg_path: str, ratio: int, val_seqs: str, dev):
    cfg = load_config(cfg_path)
    cfg["dataset"] = {**cfg.get("dataset", {}), "val_split": HOLDOUT,
                      "val_sequences": [s for s in val_seqs.split(",") if s]}
    cfg = apply_resolution_ratio(cfg, ratio, verbose=True)
    cfg["segmentation"] = {**cfg.get("segmentation", {}), "head": "mfunet", "num_classes": 2}
    return cfg, HDEMS(cfg).to(dev).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--veckm-dir", required=True, help="unzipped github.com/dhyuan99/VecKM_flow")
    ap.add_argument("--training-set", default="UNION", choices=["UNION", "EVIMO", "DSEC", "MVSEC"])
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--config", default="configs/evimo_seg.yaml")
    ap.add_argument("--val-sequences", default="scene13_dyn_test_03,scene14_dyn_test_02")
    ap.add_argument("--resolution-ratio", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=60, help="validation frames")
    ap.add_argument("--benchmark", action="store_true",
                    help="also score the 5 Hua et al. test sequences (all frames with a mover) at "
                         "the thresholds chosen on validation -- test data VecKM was not trained on")
    ap.add_argument("--max-per-sequence", type=int, default=None, help="--benchmark smoke test cap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    est = load_veckm(Path(args.veckm_dir), args.training_set)
    slice_s = 2.0 * est.P.t_radius                                  # one inference step (20 ms)
    torch.set_grad_enabled(False)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg, model = build(args.config, args.resolution_ratio, args.val_sequences, dev)

    # ---- validation: AUC, best threshold per cue, flow vs ground truth
    ds = build_dataset(cfg, HOLDOUT)
    gtf = GTFrames(ds, [12.5])
    keep = [k for k in range(len(ds)) if not isinstance(gtf(k), str)]
    ks = _spread_cap(keep, args.max_frames)
    print(f"[veckm-check] validation {args.val_sequences}: {len(ks)} frames; VecKM {args.training_set} "
          f"ensemble {args.ensemble}; device {dev}", flush=True)
    vals = {c: [] for c in CUES}
    labels, per_frame_auc = [], []
    gtcmp = {"gt": [], "veckm": [], "vsa": []}
    t_veckm = 0.0
    for n, k in enumerate(ks):
        cue, y, cmp, secs = frame_cues(k, ds, gtf, model, est, dev, slice_s, args.ensemble)
        t_veckm += secs
        for c in CUES:
            vals[c].append(cue[c])
        labels.append(y)
        if 0 < y.sum() < y.size:
            per_frame_auc.append({c: float(roc_auc_score(y, cue[c])) for c in CUES})
        if cmp is not None:
            for key, v in zip(("gt", "veckm", "vsa"), cmp):
                gtcmp[key].append(v)
        if (n + 1) % 10 == 0 or n == 0:
            print(f"  val {n + 1}/{len(ks)}  ({t_veckm / (n + 1):.1f} s VecKM / frame)", flush=True)
    gtf.close()

    y_all = np.concatenate(labels)
    print(f"\nVALIDATION: {len(ks)} frames, {y_all.size} scored events ({100 * y_all.mean():.1f}% moving); "
          f"VecKM {t_veckm / max(len(ks), 1):.1f} s per frame on {('GPU' if est.cuda_available else 'CPU')}")
    print(f"{'cue':<20s}{'AUC':>7s}{'frame AUC':>11s}{'best IoU':>10s}{'@ px':>6s}")
    report = {"validation": {"frames": len(ks), "cues": {}}}
    thr_of = {}
    for c in CUES:
        auc = roc_auc_score(y_all, np.concatenate(vals[c]))
        fa = float(np.median([p[c] for p in per_frame_auc])) if per_frame_auc else float("nan")
        ious = {thr: float(np.nanmean([frame_iou(p, g, thr) for p, g in zip(vals[c], labels)]))
                for thr in THRESHOLD_GRID_PX}
        tb = max(ious, key=lambda q: ious[q] if ious[q] == ious[q] else -1)
        thr_of[c] = tb
        report["validation"]["cues"][c] = {"auc": auc, "frame_auc": fa, "best_iou": ious[tb], "threshold": tb}
        print(f"{c:<20s}{auc:7.3f}{fa:11.3f}{100 * ious[tb]:9.1f}%{tb:6g}")
    if gtcmp["gt"]:
        gtn = np.concatenate(gtcmp["gt"])
        print(f"\nnormal component vs ground truth ({gtn.size} events with depth):")
        for name in ("veckm", "vsa"):
            v = np.concatenate(gtcmp[name])
            slope = float((v * gtn).sum() / max((gtn * gtn).sum(), 1e-9))
            corr = float(np.corrcoef(v, gtn)[0, 1])
            err = float(np.median(np.abs(v - gtn)))
            report["validation"][f"gt_{name}"] = {"corr": corr, "slope": slope, "median_abs_err_px": err}
            print(f"  {name:<6s} corr {corr:.3f}  slope {slope:.3f}  median |error| {err:.2f} px / 12.5 ms")
    print("(validation takes; VecKM_flow was trained on EVIMO2-imo's train split, which holds them)")

    # ---- benchmark: the 5 test sequences, thresholds from validation
    if args.benchmark:
        dse = build_dataset(cfg, cfg.get("dataset", {}).get("eval_split", "eval"))
        gte = GTFrames(dse, [12.5])
        by_seq: dict[str, list[int]] = {}
        for k, (seq_dir, _fi) in enumerate(dse.index):
            lab = HUA2025.label_of(Path(seq_dir).name)
            if lab is not None and not isinstance(gte(k), str):
                by_seq.setdefault(lab, []).append(k)
        frames = [(lab, k) for lab in HUA2025.reported for k in _spread_cap(by_seq.get(lab, []),
                                                                          args.max_per_sequence)]
        print(f"\n[veckm-check] BENCHMARK: {len(frames)} test frames with a mover", flush=True)
        per = {c: {} for c in CUES}
        for n, (lab, k) in enumerate(frames):
            cue, y, _cmp, _secs = frame_cues(k, dse, gte, model, est, dev, slice_s, args.ensemble)
            for c in CUES:
                per[c].setdefault(lab, []).append(frame_iou(cue[c], y, thr_of[c]))
            if (n + 1) % 50 == 0:
                print(f"  test {n + 1}/{len(frames)}", flush=True)
        gte.close()
        seqs = list(HUA2025.reported)
        print(f"\nBENCHMARK (training-free cue, threshold from validation): event IoU % at 12.5 ms")
        print(f"{'cue':<20s}{'mean':>7s}" + "".join(f"{s:>8s}" for s in seqs))
        report["benchmark"] = {}
        for c in CUES:
            ps = {s: 100 * float(np.nanmean(per[c][s])) for s in seqs if per[c].get(s)}
            mean = float(np.mean(list(ps.values()))) if ps else float("nan")
            report["benchmark"][c] = {"mean": mean, "per_sequence": ps, "threshold": thr_of[c]}
            print(f"{c:<20s}{mean:7.2f}" + "".join(f"{ps.get(s, float('nan')):8.1f}" for s in seqs))
        print(f"{'paper (Hua 2025)':<20s}{77.40:7.2f}" + "".join(f"{v:8.1f}" for v in HUA2025.reported.values()))
        print("(no training on these sequences, by VecKM_flow or by us; thresholds from validation)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
