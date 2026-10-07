"""Evaluation entry point."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from hdems import protocols as bm
from hdems.config import apply_resolution_ratio, restore_frontend
from hdems.data.build import build_dataset as _build_dataset
from hdems.losses.flow import epe_loss
from hdems.losses.seg import seg_loss
from hdems.metrics import mean_iou
from hdems.data.labels import LABEL_MODES, num_classes_for, resolve_label_mode
from hdems.data.motion_labels import IGNORE_LABEL
from hdems.grouping import GroupingParams, group_objects
from hdems.grouping import params_from_config as grouping_params
from hdems.visualize import save_event_colour_figure
from hdems.detection import detection_metrics, masks_to_boxes
from hdems.instances import binary_iou, connected_components, instance_metrics
from hdems.models.hdems import ABLATIONS, HDEMS
from hdems.vsa.velocity import EVENT_COMBINES
from hdems.seg_features import score_pixel_mask

HEADS = ("cnn", "mfcnn", "mfunet", "ridge", "prototype", "motion")


def load_config(path: str | Path) -> dict:
    # utf-8 explicitly: the config comments use α, τ, ± and Windows would
    # otherwise decode it with its locale code page
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, split: str | None = None):
    """Eval defaults to the eval split; otherwise the shared builder."""
    if split is None:
        split = cfg.get("eval", {}).get("split",
                                        cfg.get("dataset", {}).get("eval_split", "eval"))
    return _build_dataset(cfg, split)


@torch.no_grad()
def evaluate_flow(model: HDEMS, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_epe = 0.0
    n = 0
    for batch in loader:
        surface = batch["surface"].to(device)
        target = batch["flow"].to(device)
        out = model(surface, task="flow")
        total_epe += epe_loss(out["flow"], target).item()
        n += 1
    return total_epe / max(n, 1)


def _annotate_box(ax, lines) -> None:
    from matplotlib.offsetbox import AnchoredText
    text = "\n".join(lines)
    at = AnchoredText(text, loc="lower left", prop=dict(family="monospace", size=7), frameon=True)
    at.patch.set_alpha(0.85)
    ax.add_artist(at)


def _save_seg_panel(
    surface,
    mask,
    pred,
    out_path: Path,
    num_classes: int,
    *,
    title: str = "prediction",
    box_lines: list[str] | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hdems.visualize import event_image

    gt = mask.detach().cpu().numpy()
    pr = pred.detach().cpu().numpy()
    ignore = gt == IGNORE_LABEL

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(event_image(surface), cmap="gray")        # contrast-stretched
    ax[0].set_title("events")
    # 'nearest': smooth resampling blends the qualitative class colours into
    # rainbow speckle. Ignore (255) is drawn in its own colour -- it used to be
    # clipped to the same colour as "moving".
    ax[1].imshow(np.where(ignore, 0, gt), cmap="tab20", vmin=0, vmax=num_classes - 1,
                 interpolation="nearest")
    if ignore.any():
        ax[1].imshow(np.ma.masked_where(~ignore, np.ones_like(gt, dtype=float)),
                     cmap="autumn_r", vmin=0, vmax=1, alpha=0.9, interpolation="nearest")
    ax[1].set_title("ground truth (yellow = ignored)")
    ax[2].imshow(pr, cmap="tab20", vmin=0, vmax=num_classes - 1, interpolation="nearest")
    ax[2].set_title(title)
    for a in ax:
        a.axis("off")
    if box_lines:
        _annotate_box(ax[2], box_lines)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def choose_panels(dataset, mode: str, k: int) -> set[int]:
    """Which eval frames to draw figures for (the metrics always use every frame).

    first  : the first k frames -- the old behaviour. The index cycles through the
             eval sequences, so these are the first ~0.1-0.3 s of every recording,
             usually BEFORE any object starts moving: almost all static frames.
    spread : k frames evenly spaced over the whole eval set (default).
    movers : k frames, evenly spaced, among frames where a moving object is
             actually visible -- best for inspecting object colouring.
    """
    n = len(dataset)
    if k <= 0 or n == 0:
        return set()
    if mode == "first":
        return set(range(min(k, n)))
    candidates = list(range(n))
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    if mode == "movers" and getattr(base, "index", None):
        from hdems.data.evimo2_reader import _visible_moving_frames, load_meta
        by_seq: dict = {}
        for i in range(n):                       # Subset(range(N)) keeps the prefix order
            seq, fi = base.index[i]
            by_seq.setdefault(seq, []).append((i, fi))
        movers = []
        for seq, items in by_seq.items():
            visible = _visible_moving_frames(seq, load_meta(seq), [fi for _, fi in items],
                                             base.motion_params, getattr(base, "min_moving_px", 100))
            movers += [i for i, fi in items if fi in visible]
        if movers:
            candidates = sorted(movers)
            print(f"[eval] panels: {min(k, len(movers))} of {len(movers)} frames with a visible mover")
        else:
            print("[eval] panels: no frame has a visible mover -- spreading over all frames")
    pick = np.linspace(0, len(candidates) - 1, num=min(k, len(candidates))).round().astype(int)
    return {candidates[j] for j in pick}


@torch.no_grad()
def evaluate_segmentation(
    model: HDEMS,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    save_dir: Path | None = None,
    max_images: int = 50,
    *,
    panel_title: str = "prediction",
    box_lines: list[str] | None = None,
    label_mode: str = "motion",
    min_instance: int = 50,
    detect: bool = False,
    color_dir: Path | None = None,
    events_only_baseline: bool = False,
    grouping: GroupingParams | None = None,
    panel_indices: set[int] | None = None,
    ablation: bool = False,
) -> dict[str, float]:
    """``grouping``: split the CNN's moving pixels into OBJECTS by motion model
    (hdems.grouping) -- one colour per object -- instead of connected components.

    ``ablation``: also predict every frame with the motion input removed and with
    the appearance input removed (HDEMS.ablate) and report how much that changes
    -- a head whose output barely changes without motion is not segmenting motion.
    """
    model.eval()
    ious: list[float] = []
    total_loss = 0.0
    n = 0
    saved = 0
    fg_ious: list[float] = []          # motion mode: foreground IoU (the headline)
    inst_scores: list[dict] = []       # motion mode: instance matching
    oracle_scores: list[dict] = []     # grouping on the GT moving pixels (best case)
    det_scores: list[dict] = []        # object detection: box matching
    ablation = ablation and not events_only_baseline
    abl = {w: {"fg": [], "changed": 0, "px": 0} for w in ABLATIONS}
    # index of the surface that ends at the label time (time_frames entry 1.0)
    base = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
    tf = getattr(base, "time_frames", None)
    label_t = int(np.argmax(tf)) if tf else 0

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    if color_dir is not None:
        color_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        surface = batch["surface"].to(device)
        mask = batch["mask"].to(device).long()
        # Score on EVENT pixels only (pixels without events carry no evidence), only
        # on events near the label time (older ones are the trail a mover leaves
        # behind, which the label calls static), and never on the ignore label
        # (ambiguous speed / mask boundary band).
        events_t = score_pixel_mask(batch, surface)
        valid = events_t & (mask >= 0) & (mask != IGNORE_LABEL)
        flow = None
        if events_only_baseline:
            # Control: call EVERY event pixel "moving". It never looks at the model,
            # so the front end is skipped entirely (no loss is reported).
            pred = events_t.long()
        else:
            out_m = model(surface, task="segmentation")
            logits, flow = out_m["seg_logits"], out_m.get("flow")
            if valid.any():
                total_loss += seg_loss(logits, mask.masked_fill(~valid, IGNORE_LABEL)).item()
            pred = logits.argmax(dim=1)
        ious.append(mean_iou(pred[0], mask[0], num_classes, valid=valid[0]))

        objects = objects_gt_mask = gt_inst = None
        if label_mode in ("motion", "tracked"):
            # Option A: foreground IoU + class-agnostic instance matching.
            v = valid[0].cpu().numpy()
            ev = events_t[0].cpu().numpy()
            pred_fg = (pred[0] > 0).cpu().numpy()
            gt_fg = (mask[0] > 0).cpu().numpy()
            fg_ious.append(binary_iou(pred_fg, gt_fg, valid=v))
            # Ground-truth OBJECTS: independently moving objects, with parts that move
            # rigidly together merged (gt_instances). Older data: raw moving ids.
            gt_src = batch.get("gt_instances")
            if gt_src is not None:
                gt_inst = gt_src[0].cpu().numpy().astype(np.int64)
            elif batch.get("gt_moving", batch.get("gt_raw")) is not None:
                gt_inst = batch.get("gt_moving", batch.get("gt_raw"))[0].cpu().numpy().astype(np.int64) // 1000
            if grouping is not None and flow is not None:
                # objects = the CNN's moving pixels grouped by motion model (the same
                # ego-compensated velocity the head saw)
                res, _ = model.residual_from_flow(flow, surface)
                res = res[0].cpu().numpy()
                objects = group_objects(res, pred_fg & ev, grouping)
                if gt_inst is not None and ((gt_inst > 0) & ev).any():
                    objects_gt_mask = group_objects(res, (gt_inst > 0) & ev, grouping)
                    oracle_scores.append(instance_metrics(objects_gt_mask, gt_inst, valid=v))
                pred_inst = objects
            else:
                pred_inst = connected_components(np.logical_and(pred_fg, v),
                                                 min_size=min_instance)
            if gt_inst is not None:
                inst_scores.append(instance_metrics(pred_inst, gt_inst, valid=v))
                if detect:                       # boxes = extent of each instance
                    det_scores.append(detection_metrics(
                        masks_to_boxes(pred_inst, min_area=min_instance),
                        masks_to_boxes(gt_inst, min_area=min_instance)))

        if ablation:
            gt_fg_t = (mask[0] > 0)
            for what in ABLATIONS:
                model.ablate = what
                try:
                    p2 = model(surface, task="segmentation")["seg_logits"].argmax(dim=1)
                finally:
                    model.ablate = None
                a = abl[what]
                a["changed"] += int(((p2 != pred) & valid).sum())
                a["px"] += int(valid.sum())
                a["fg"].append(binary_iou((p2[0] > 0).cpu().numpy(), gt_fg_t.cpu().numpy(),
                                          valid=valid[0].cpu().numpy()))

        draw = (n in panel_indices) if panel_indices is not None else saved < max_images
        if (save_dir is not None or color_dir is not None) and draw:
            # background image: the surface that ends at the label time, not the whole
            # 100 ms stack, which smears every mover into a long trail
            ref = surface[0, label_t] if surface.dim() == 5 else surface[0]
            if save_dir is not None:
                # show the prediction where it is scored; elsewhere = background
                pred_vis = pred.masked_fill(~valid, 0)
                _save_seg_panel(
                    ref, mask[0], pred_vis[0],
                    save_dir / f"eval_{n:05d}.png", num_classes,
                    title=panel_title, box_lines=box_lines,
                )
            if color_dir is not None:
                save_event_colour_figure(
                    ref, pred[0], events_t[0],
                    color_dir / f"events_{n:05d}.png",
                    gt_moving=(batch["gt_moving"][0] if "gt_moving" in batch else None),
                    objects=objects, objects_gt_mask=objects_gt_mask, gt_objects=gt_inst,
                    gt_slow=(batch["gt_slow"][0] if "gt_slow" in batch else None),
                    min_instance=min_instance,
                )
            saved += 1
        n += 1

    if save_dir is not None:
        print(f"Saved {saved} panels to {save_dir.resolve()}")
    if color_dir is not None:
        print(f"Saved {saved} event-colour figures to {color_dir.resolve()}")

    out: dict[str, float] = {
        "loss": total_loss / max(n, 1),
        "miou": sum(ious) / max(len(ious), 1),
    }
    if label_mode in ("motion", "tracked"):
        fin = [v for v in fg_ious if v == v]                      # drop NaN frames
        out["fg_iou"] = sum(fin) / max(len(fin), 1)
        if inst_scores:
            # Averaged over frames that CONTAIN a moving object: frames without one
            # give NaN precision/recall, and a plain mean turned the whole result NaN.
            with_gt = [s for s in inst_scores if s["n_gt"] > 0]
            out["instance_miou"] = float(np.nanmean([s["instance_miou"] for s in with_gt])) if with_gt else float("nan")
            out["precision"] = float(np.nanmean([s["precision"] for s in with_gt])) if with_gt else float("nan")
            out["recall"] = float(np.nanmean([s["recall"] for s in with_gt])) if with_gt else float("nan")
            out["mean_pred_instances"] = float(np.mean([s["n_pred"] for s in with_gt])) if with_gt else 0.0
            out["mean_gt_instances"] = float(np.mean([s["n_gt"] for s in with_gt])) if with_gt else 0.0
            out["frames_with_objects"] = len(with_gt)
            # static frames: how often the model invents an object where nothing moves
            static = [s for s in inst_scores if s["n_gt"] == 0]
            out["static_frames"] = len(static)
            out["static_false_object_rate"] = (float(np.mean([s["n_pred"] > 0 for s in static]))
                                               if static else float("nan"))
        if oracle_scores:
            out["oracle_instance_miou"] = float(np.nanmean([s["instance_miou"] for s in oracle_scores]))
            out["oracle_mean_pred_instances"] = float(np.mean([s["n_pred"] for s in oracle_scores]))
    if ablation:
        for what, a in abl.items():
            fin = [v for v in a["fg"] if v == v]
            out[f"no_{what}_fg_iou"] = sum(fin) / max(len(fin), 1)
            out[f"no_{what}_changed"] = a["changed"] / max(a["px"], 1)
    if det_scores:
        k = len(det_scores)
        out["det_precision"] = sum(s["det_precision"] for s in det_scores) / k
        out["det_recall"] = sum(s["det_recall"] for s in det_scores) / k
        out["det_miou"] = sum(s["det_miou"] for s in det_scores) / k
        out["det_tp"] = sum(s["tp"] for s in det_scores)
        out["det_n_gt"] = sum(s["n_gt"] for s in det_scores)
        out["det_n_pred"] = sum(s["n_pred"] for s in det_scores)
    return out


@torch.no_grad()
def measure_seg_latency(model: HDEMS, surface: torch.Tensor, device: torch.device, repeats: int = 30) -> float:
    """Real per-frame latency -- the flow cache is bypassed, otherwise this would
    report the speed of a disk read instead of the VSA front end."""
    model.eval()
    surface = surface.to(device)
    cache, model.flow_cache = model.flow_cache, None
    try:
        for _ in range(5):
            model(surface, task="segmentation")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(surface, task="segmentation")
        if device.type == "cuda":
            torch.cuda.synchronize()
    finally:
        model.flow_cache = cache
    return (time.perf_counter() - t0) / repeats * 1000.0


def run_published_benchmark(args, cfg: dict, model: HDEMS, dataset, eval_split: str,
                            device: torch.device) -> None:
    """--benchmark: score every frame of the paper's sequences with the paper's metric.

    --max-samples caps the frames PER SEQUENCE (evenly spaced) for smoke tests.
    """
    bench = bm.BENCHMARKS[args.benchmark]
    s = bm.settings(cfg, gt=args.benchmark_gt, pred=args.benchmark_pred,
                    windows_ms=args.benchmark_window_ms)
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    if eval_split != bench.split:
        base = build_dataset(cfg, split=bench.split)
    scores, stats = bm.run_benchmark(
        model, base, bench, device,
        windows_ms=s["event_windows_ms"], gt_mode=s["gt"], pred_source=s["pred"],
        grouping=grouping_params(cfg) if s["pred"] == "objects" else None,
        min_gt_events=s["min_gt_events"],
        max_per_sequence=args.max_samples,
    )
    print(bm.format_report(bench, scores, stats, gt_mode=s["gt"], pred_source=s["pred"]))
    if args.max_samples:
        print(f"(smoke test: at most {args.max_samples} frames per sequence -- not comparable)")
    if args.benchmark_out:
        bm.save_json(Path(args.benchmark_out), bench, scores, stats,
                     {**s, "head": model.seg_head.__class__.__name__,
                      "checkpoint": args.checkpoint or args.ridge_checkpoint or args.prototype_checkpoint,
                      "max_per_sequence": args.max_samples})
        print(f"Per-frame scores -> {Path(args.benchmark_out).resolve()}")
    if model.flow_cache is not None:
        print(model.flow_cache.summary())


def _apply_head_config(cfg: dict, head: str | None) -> dict:
    cfg = dict(cfg)
    seg = dict(cfg.get("segmentation", {}))
    if head:
        seg["head"] = head
    cfg["segmentation"] = seg
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HD-EMS")
    parser.add_argument("--config", type=str, default="configs/evimo_seg.yaml")
    # Any ONE of these three may be given for any head; the loader matches it to
    # --head (so a single run script can pass the same flag for every model).
    parser.add_argument("--checkpoint", type=str, required=False,
                        help="Head checkpoint .pt (any head)")
    parser.add_argument("--ridge-checkpoint", type=str, default=None,
                        help="Head checkpoint .pt (any head; alias)")
    parser.add_argument("--prototype-checkpoint", type=str, default=None,
                        help="Head checkpoint .pt (any head; alias)")
    parser.add_argument("--head", type=str, choices=list(HEADS), default=None,
                        help="Default: the head stored in the checkpoint, else segmentation.head.")
    parser.add_argument("--axis-combine", type=str, choices=["bind", "bundle"], default=None,
                        help="Override velocity.axis_combine (else taken from the checkpoint).")
    parser.add_argument("--event-combine", type=str, choices=list(EVENT_COMBINES), default=None,
                        help="Override velocity.event_combine (else taken from the checkpoint).")
    parser.add_argument("--event-feature", type=str, choices=["phi", "f"], default=None,
                        help="Override velocity.event_feature (else taken from the checkpoint).")
    parser.add_argument("--label-mode", type=str, choices=list(LABEL_MODES), default=None,
                        help="Override dataset.label_mode (else taken from the checkpoint). "
                             "motion = pose-derived moving; tracked = legacy mask>0 baseline.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-images", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate on only the first N dataset frames (smoke test).")
    parser.add_argument("--detect", action="store_true",
                        help="Also report object detection (boxes from predicted instances).")
    parser.add_argument("--color-events", type=str, default=None,
                        help="Directory for event-colour figures: moving events red, "
                             "static grey, plus per-instance colours.")
    parser.add_argument("--events-only-baseline", action="store_true",
                        help="CONTROL: ignore the model and call every event pixel "
                             "'moving'. Foreground IoU must be poor once labels are motion.")
    parser.add_argument("--compare-heads", action="store_true",
                        help="Run CNN and Ridge on same loader; write compare panels")
    parser.add_argument("--resolution-ratio", type=int, default=None,
                        help="Override the resolution ratio (default: the checkpoint's). "
                             "A head trained at one ratio will not load at another.")
    parser.add_argument("--panels", choices=["spread", "movers", "first"], default="spread",
                        help="Which frames get figures (metrics always use all frames): "
                             "spread = evenly over the eval set (default); movers = only frames "
                             "with a visible moving object; first = the first N (old behaviour: "
                             "mostly the static start of each recording).")
    parser.add_argument("--skip-latency", action="store_true",
                        help="Skip the latency measurement (35 uncached forward passes) "
                             "-- for quick checks, especially on CPU.")
    parser.add_argument("--ablation-check", action="store_true",
                        help="Also predict every frame without the motion input and without "
                             "the appearance input, and report how much each changes the "
                             "result. A motion segmenter must collapse without motion.")
    parser.add_argument("--benchmark", choices=list(bm.BENCHMARKS), default=None,
                        help="Score with a PUBLISHED protocol instead of ours and print the "
                             "paper's table next to our numbers (hdems/protocols.py). "
                             "hua2025 = IoU over events, full resolution, 5 EVIMO2 sequences.")
    parser.add_argument("--benchmark-gt", choices=list(bm.GT_MODES), default=None,
                        help="Which objects count as moving (default: config benchmark.gt).")
    parser.add_argument("--benchmark-pred", choices=list(bm.PRED_SOURCES), default=None,
                        help="fg = the head's moving pixels | objects = only pixels the motion "
                             "grouping keeps (default: config benchmark.pred).")
    parser.add_argument("--benchmark-window-ms", type=float, nargs="+", default=None,
                        help="Event window(s) per frame, ms before the label time "
                             "(default: config benchmark.event_windows_ms).")
    parser.add_argument("--benchmark-out", type=str, default=None,
                        help="Write the per-frame benchmark scores to this .json file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.label_mode:
        cfg["dataset"] = {**cfg.get("dataset", {}), "label_mode": args.label_mode}
    if args.axis_combine or args.event_combine or args.event_feature:   # CLI overrides config
        vel = dict(cfg.get("velocity", {}))
        if args.axis_combine:
            vel["axis_combine"] = args.axis_combine
        if args.event_combine:
            vel["event_combine"] = args.event_combine
        if args.event_feature:
            vel["event_feature"] = args.event_feature
        cfg["velocity"] = vel
    head = args.head or cfg.get("segmentation", {}).get("head", "cnn")
    # Whichever checkpoint flag was given (compare mode keeps them separate).
    any_ckpt = None if args.compare_heads else (
        args.checkpoint or args.prototype_checkpoint or args.ridge_checkpoint)
    # A checkpoint records the combine AND the label mode it was built with; both
    # decide tensor shapes (feature dim, class count), so read them back for EVERY
    # head before anything is constructed. CLI flags still win.
    peek_ckpt = args.checkpoint or args.prototype_checkpoint or args.ridge_checkpoint
    if peek_ckpt:
        try:
            _meta = torch.load(peek_ckpt, map_location="cpu", weights_only=False)
            if isinstance(_meta, dict):
                vel = dict(cfg.get("velocity", {}))
                if not args.axis_combine and _meta.get("axis_combine"):
                    vel["axis_combine"] = _meta["axis_combine"]
                if not args.event_combine and _meta.get("event_combine"):
                    vel["event_combine"] = _meta["event_combine"]
                if not args.event_feature:          # checkpoints older than the option = phi
                    vel["event_feature"] = _meta.get("event_feature", "phi")
                cfg["velocity"] = vel
                if not args.label_mode and _meta.get("label_mode"):
                    cfg["dataset"] = {**cfg.get("dataset", {}),
                                      "label_mode": _meta["label_mode"]}
                if not args.head and _meta.get("head") in HEADS:
                    head = _meta["head"]
                # head sizes decide the tensor shapes: rebuild what was trained
                if _meta.get("head_config"):
                    cfg["segmentation"] = {**cfg.get("segmentation", {}), **_meta["head_config"]}
                # A trained head is only valid for the front end it was trained on
                # (kernel, M, alpha, smooth, tau, resolution...). Rebuild exactly
                # that, even if the YAML has changed since training.
                if _meta.get("frontend"):
                    cfg = restore_frontend(cfg, _meta["frontend"])
                    print("[eval] front end restored from checkpoint")
        except Exception as e:
            print(f"[eval] could not read checkpoint metadata ({e}); using the YAML")
    # --resolution-ratio overrides; otherwise the checkpoint's (via frontend) or YAML's.
    cfg = apply_resolution_ratio(cfg, args.resolution_ratio, verbose=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Label mode fixes the class count -> must match what the head was trained with.
    label_mode = resolve_label_mode(cfg)
    num_classes = num_classes_for(label_mode, cfg.get("segmentation", {}).get("num_classes", 32))
    cfg["segmentation"] = {**cfg.get("segmentation", {}), "num_classes": num_classes}
    seg_cfg = cfg.get("segmentation", {})
    print(f"label_mode={label_mode}  num_classes={num_classes}")
    task = cfg.get("train", {}).get("task", "flow")
    eval_split = cfg.get("eval", {}).get("split", cfg.get("dataset", {}).get("eval_split", "eval"))

    def build_model(head_name: str) -> HDEMS:
        mcfg = _apply_head_config(cfg, head_name)
        model = HDEMS(mcfg).to(device)
        def _match_combine(path: str) -> None:
            # Auto-match the combine stored in the checkpoint unless CLI overrode it.
            meta = torch.load(path, map_location="cpu", weights_only=False)
            if not args.axis_combine and meta.get("axis_combine"):
                model.axis_combine = meta["axis_combine"]
            if not args.event_combine and meta.get("event_combine"):
                model.event_combine = meta["event_combine"]
            if not args.event_feature:
                model.event_feature = meta.get("event_feature", "phi")

        if head_name == "ridge":
            rpath = args.ridge_checkpoint or any_ckpt or seg_cfg.get("ridge_weights")
            if not rpath:
                raise SystemExit("a checkpoint flag or segmentation.ridge_weights is required")
            model.seg_head.load(rpath)
            _match_combine(rpath)
        elif head_name == "prototype":
            ppath = args.prototype_checkpoint or any_ckpt or seg_cfg.get("prototype_weights")
            if not ppath:
                raise SystemExit("a checkpoint flag or segmentation.prototype_weights is required")
            model.seg_head.load(ppath)
            _match_combine(ppath)
        else:                                               # cnn / motion (trained heads)
            cpath = args.checkpoint or any_ckpt
            if cpath:
                ckpt = torch.load(cpath, map_location=device, weights_only=True)
                state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                model.load_state_dict(state, strict=False)
            elif task == "segmentation":
                raise SystemExit(f"a checkpoint is required for head={head_name!r} "
                                 "(an untrained head would give meaningless results)")
        return model

    try:
        dataset = build_dataset(cfg, split=eval_split)
        if len(dataset) == 0:
            raise FileNotFoundError("Eval dataset is empty")
        n_full = len(dataset)
        max_n = args.max_samples or cfg.get("eval", {}).get("max_samples")
        if max_n is not None and 0 < max_n < n_full:
            dataset = Subset(dataset, list(range(max_n)))
        print(f"Eval samples ({eval_split}): {len(dataset)}"
              + (f" (of {n_full})" if len(dataset) != n_full else ""))
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        if task != "segmentation":
            model = build_model("cnn")
            epe = evaluate_flow(model, loader, device)
            print(f"Mean EPE: {epe:.4f}")
            return

        if args.compare_heads:
            save_root = Path(args.save_images or "output/ridge_compare")
            cnn = build_model("cnn")
            ridge = build_model("ridge")
            cnn_metrics = evaluate_segmentation(
                cnn, loader, device, num_classes,
                save_dir=save_root / "cnn", max_images=args.max_images,
                panel_title="CNN head",
                box_lines=[f"head: cnn", f"params: {cnn.seg_head_param_count():,}"],
            )
            ridge_ck = args.ridge_checkpoint or seg_cfg.get("ridge_weights", "")
            rk = torch.load(ridge_ck, map_location="cpu", weights_only=False) if ridge_ck else {}
            ridge_metrics = evaluate_segmentation(
                ridge, loader, device, num_classes,
                save_dir=save_root / "ridge", max_images=args.max_images,
                panel_title="Ridge head",
                box_lines=[
                    f"head: ridge",
                    f"lambda: {rk.get('alpha', '?')}",
                    f"imbalance: {rk.get('imbalance', '?')}",
                    f"D: {rk.get('feature_dim', '?')}",
                    f"mean_center: {rk.get('mean_center', '?')}",
                    f"params: {ridge.seg_head_param_count():,}",
                ],
            )
            sample = dataset[0]["surface"].unsqueeze(0).to(device)
            cnn_ms = measure_seg_latency(cnn, sample, device)
            ridge_ms = measure_seg_latency(ridge, sample, device)
            print("\n=== CNN vs Ridge ===")
            print(f"{'':12s} {'mIoU':>8s} {'loss':>8s} {'latency_ms':>12s} {'head_params':>12s}")
            print(f"{'CNN':12s} {cnn_metrics['miou']:8.4f} {cnn_metrics['loss']:8.4f} "
                  f"{cnn_ms:12.2f} {cnn.seg_head_param_count():12,}")
            print(f"{'Ridge':12s} {ridge_metrics['miou']:8.4f} {ridge_metrics['loss']:8.4f} "
                  f"{ridge_ms:12.2f} {ridge.seg_head_param_count():12,}")
            return

        model = build_model(head)
        if args.benchmark:
            run_published_benchmark(args, cfg, model, dataset, eval_split, device)
            return
        rk = {}
        if head == "ridge" and (args.ridge_checkpoint or seg_cfg.get("ridge_weights")):
            rpath = args.ridge_checkpoint or seg_cfg.get("ridge_weights")
            rk = torch.load(rpath, map_location="cpu", weights_only=False)
        box = None
        if head == "ridge":
            box = [
                f"lambda: {rk.get('alpha', '?')}",
                f"imbalance: {rk.get('imbalance', '?')}",
                f"D: {rk.get('feature_dim', '?')}",
                f"mean_center: {rk.get('mean_center', '?')}",
                f"motion: {rk.get('motion_features', '?')}",
            ]
        save_dir = Path(args.save_images) if args.save_images else None
        metrics = evaluate_segmentation(
            model, loader, device, num_classes,
            save_dir=save_dir, max_images=args.max_images,
            panel_title=f"{head} head", box_lines=box,
            label_mode=label_mode, detect=args.detect,
            color_dir=Path(args.color_events) if args.color_events else None,
            events_only_baseline=args.events_only_baseline,
            # one colour per object: group moving pixels by motion model
            grouping=(grouping_params(cfg)
                      if (cfg.get("grouping", {}) or {}).get("enabled", True)
                      and not args.events_only_baseline else None),
            panel_indices=(choose_panels(dataset, args.panels, args.max_images)
                           if (save_dir is not None or args.color_events) else None),
            ablation=args.ablation_check,
        )
        if model.flow_cache is not None:
            print(model.flow_cache.summary())
        # The control never runs the model, so its latency would be meaningless.
        ms = float("nan") if (args.events_only_baseline or args.skip_latency) else measure_seg_latency(
            model, dataset[0]["surface"].unsqueeze(0), device)
        title = "EVENTS-ONLY BASELINE (control)" if args.events_only_baseline else f"Head: {head}"
        print(f"{title}  label_mode: {label_mode}  event_feature: {model.event_feature}  "
              f"axis_combine: {model.axis_combine}  event_combine: {model.event_combine}")
        if label_mode in ("motion", "tracked"):
            # Headline: foreground IoU. mIoU averages in the easy static class and
            # stays near 0.5 even for a model that predicts "static" everywhere.
            print(f"FG IoU (moving vs rest) [HEADLINE]: {metrics.get('fg_iou', float('nan')):.4f}")
        if not args.events_only_baseline:
            print(f"Seg loss: {metrics['loss']:.4f}")
        print(f"mIoU:     {metrics['miou']:.4f}   (secondary — see FG IoU)")
        if label_mode in ("motion", "tracked"):
            if "instance_miou" in metrics:
                src = "motion grouping" if not args.events_only_baseline else "connected pieces"
                print(f"--- OBJECTS ({src}), over {metrics.get('frames_with_objects', 0)} frames "
                      f"that contain a moving object ---")
                print(f"Object mIoU (matched):        {metrics['instance_miou']:.4f}")
                print(f"Object P / R @0.5:            "
                      f"{metrics['precision']:.3f} / {metrics['recall']:.3f}")
                print(f"Objects pred / gt (avg):      "
                      f"{metrics['mean_pred_instances']:.2f} / {metrics['mean_gt_instances']:.2f}")
                if "oracle_instance_miou" in metrics:
                    print(f"Object mIoU, grouping on GT moving px [best case]: "
                          f"{metrics['oracle_instance_miou']:.4f}  "
                          f"(objects/frame {metrics['oracle_mean_pred_instances']:.2f})")
                if metrics.get("static_frames"):
                    print(f"Static frames with a false object: "
                          f"{metrics['static_false_object_rate']:.1%} of {metrics['static_frames']}")
        if "det_precision" in metrics:
            print("--- object detection (boxes from instances) ---")
            print(f"Box P / R @0.5:  {metrics['det_precision']:.3f} / {metrics['det_recall']:.3f}")
            print(f"Box mIoU:        {metrics['det_miou']:.4f}")
            print(f"TP / pred / gt:  {metrics['det_tp']} / {metrics['det_n_pred']} / {metrics['det_n_gt']}")
        if "no_motion_fg_iou" in metrics:
            full = metrics.get("fg_iou", float("nan"))
            nomo, noap = metrics["no_motion_fg_iou"], metrics["no_appearance_fg_iou"]
            drop = 1.0 - nomo / full if full > 0 else float("nan")
            print("--- ABLATION CHECK: does the head segment MOTION? ---")
            print(f"FG IoU   full {full:.4f} | without motion {nomo:.4f} | "
                  f"without appearance {noap:.4f}")
            print(f"Predictions changed:  without motion {metrics['no_motion_changed']:.1%} | "
                  f"without appearance {metrics['no_appearance_changed']:.1%}")
            if drop != drop:
                verdict = "n/a (FG IoU is 0)"
            elif drop >= 0.5:
                verdict = f"USES MOTION -- removing it costs {drop:.0%} of the FG IoU"
            elif drop < 0.1:
                verdict = (f"IGNORES MOTION -- removing it costs only {drop:.0%} of the FG IoU; "
                           "the head decides from appearance")
            else:
                verdict = f"PARTLY uses motion -- removing it costs {drop:.0%} of the FG IoU"
            print(f"Verdict: {verdict}")
        print(f"Latency:  {ms:.2f} ms/frame ({device})")
        print(f"Head params: {model.seg_head_param_count():,}  "
              f"(trainable total: {model.num_trainable_params:,})")
    except FileNotFoundError as e:
        print(f"Dataset not ready: {e}")


if __name__ == "__main__":
    main()
