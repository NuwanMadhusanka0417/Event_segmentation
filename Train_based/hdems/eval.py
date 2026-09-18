"""Evaluation entry point."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from hdems.data.dsec import DSECDataset
from hdems.data.evimo import EVIMODataset
from hdems.losses.flow import epe_loss
from hdems.losses.seg import seg_loss
from hdems.metrics import mean_iou
from hdems.models.hdems import HDEMS


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, split: str | None = None):
    ds_cfg = cfg.get("dataset", {})
    name = ds_cfg.get("name", "dsec")
    if split is None:
        split = cfg.get("eval", {}).get("split", ds_cfg.get("eval_split", "eval"))

    if name == "dsec":
        return DSECDataset(ds_cfg["root"], split)
    return EVIMODataset(
        ds_cfg["root"],
        split,
        ds_cfg.get("version"),
        height=ds_cfg.get("height", 480),
        width=ds_cfg.get("width", 640),
        window_ms=ds_cfg.get("window_ms", 50.0),
        decay=cfg.get("time_surface", {}).get("decay", 0.8),
        remap_mask=ds_cfg.get("remap_mask", True),
        use_classical_fallback=ds_cfg.get("use_classical_fallback", True),
        time_frames=ds_cfg.get("time_frames"),
    )


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

    surf = surface.detach().cpu().numpy()
    ev = surf.reshape(-1, surf.shape[-2], surf.shape[-1]).sum(0)  # handles (2,H,W) and (T,2,H,W)
    gt = mask.detach().cpu().numpy()
    pr = pred.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(ev, cmap="gray")
    ax[0].set_title("events")
    ax[1].imshow(gt, cmap="tab20", vmin=0, vmax=num_classes - 1)
    ax[1].set_title("ground truth")
    ax[2].imshow(pr, cmap="tab20", vmin=0, vmax=num_classes - 1)
    ax[2].set_title(title)
    for a in ax:
        a.axis("off")
    if box_lines:
        _annotate_box(ax[2], box_lines)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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
) -> dict[str, float]:
    model.eval()
    ious: list[float] = []
    total_loss = 0.0
    n = 0
    saved = 0

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        surface = batch["surface"].to(device)
        mask = batch["mask"].to(device).long()
        out = model(surface, task="segmentation")
        logits = out["seg_logits"]
        total_loss += seg_loss(logits, mask).item()
        pred = logits.argmax(dim=1)
        ious.append(mean_iou(pred[0], mask[0], num_classes))

        if save_dir is not None and saved < max_images:
            _save_seg_panel(
                surface[0], mask[0], pred[0],
                save_dir / f"eval_{n:05d}.png", num_classes,
                title=panel_title, box_lines=box_lines,
            )
            saved += 1
        n += 1

    if save_dir is not None:
        print(f"Saved {saved} panels to {save_dir.resolve()}")

    return {
        "loss": total_loss / max(n, 1),
        "miou": sum(ious) / max(len(ious), 1),
    }


@torch.no_grad()
def measure_seg_latency(model: HDEMS, surface: torch.Tensor, device: torch.device, repeats: int = 30) -> float:
    model.eval()
    surface = surface.to(device)
    for _ in range(5):
        model(surface, task="segmentation")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(surface, task="segmentation")
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0


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
    parser.add_argument("--checkpoint", type=str, required=False,
                        help="CNN checkpoint (seg_head weights when head=cnn)")
    parser.add_argument("--ridge-checkpoint", type=str, default=None,
                        help="Ridge weights .pt (required when head=ridge)")
    parser.add_argument("--prototype-checkpoint", type=str, default=None,
                        help="Prototype weights .pt (required when head=prototype)")
    parser.add_argument("--head", type=str, choices=["cnn", "ridge", "prototype", "motion"], default=None)
    parser.add_argument("--axis-combine", type=str, choices=["bind", "bundle"], default=None,
                        help="Override velocity.axis_combine (else taken from the checkpoint).")
    parser.add_argument("--event-combine", type=str, choices=["bind", "bundle", "concat"], default=None,
                        help="Override velocity.event_combine (else taken from the checkpoint).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-images", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate on only the first N dataset frames (smoke test).")
    parser.add_argument("--compare-heads", action="store_true",
                        help="Run CNN and Ridge on same loader; write compare panels")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.axis_combine or args.event_combine:                 # CLI overrides config
        vel = dict(cfg.get("velocity", {}))
        if args.axis_combine:
            vel["axis_combine"] = args.axis_combine
        if args.event_combine:
            vel["event_combine"] = args.event_combine
        cfg["velocity"] = vel
    head = args.head or cfg.get("segmentation", {}).get("head", "cnn")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seg_cfg = cfg.get("segmentation", {})
    num_classes = seg_cfg.get("num_classes", 16)
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

        if head_name == "ridge":
            rpath = args.ridge_checkpoint or seg_cfg.get("ridge_weights")
            if not rpath:
                raise SystemExit("--ridge-checkpoint or segmentation.ridge_weights required")
            model.seg_head.load(rpath)
            _match_combine(rpath)
        elif head_name == "prototype":
            ppath = args.prototype_checkpoint or seg_cfg.get("prototype_weights")
            if not ppath:
                raise SystemExit("--prototype-checkpoint or segmentation.prototype_weights required")
            model.seg_head.load(ppath)
            _match_combine(ppath)
        elif args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state, strict=False)
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
        )
        sample = dataset[0]["surface"].unsqueeze(0)
        ms = measure_seg_latency(model, sample, device)
        print(f"Head: {head}")
        print(f"Seg loss: {metrics['loss']:.4f}")
        print(f"mIoU:     {metrics['miou']:.4f}")
        print(f"Latency:  {ms:.2f} ms/frame ({device})")
        print(f"Head params: {model.seg_head_param_count():,}  "
              f"(trainable total: {model.num_trainable_params:,})")
    except FileNotFoundError as e:
        print(f"Dataset not ready: {e}")


if __name__ == "__main__":
    main()
