"""Evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from hdems.data.dsec import DSECDataset
from hdems.data.evimo import EVIMODataset
from hdems.losses.flow import epe_loss
from hdems.losses.seg import seg_loss
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


def _save_seg_panel(surface, mask, pred, out_path: Path, num_classes: int) -> None:
    """Write an events | ground-truth | prediction panel for one sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = surface.detach().cpu().numpy().sum(0)     # (H, W) aggregated polarity
    gt = mask.detach().cpu().numpy()
    pr = pred.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(ev, cmap="gray"); ax[0].set_title("events")
    ax[1].imshow(gt, cmap="tab20", vmin=0, vmax=num_classes - 1); ax[1].set_title("ground truth")
    ax[2].imshow(pr, cmap="tab20", vmin=0, vmax=num_classes - 1); ax[2].set_title("prediction")
    for a in ax:
        a.axis("off")
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
        valid = mask >= 0
        for cls in range(1, num_classes):
            pred_c = pred == cls
            mask_c = mask == cls
            union = (pred_c | mask_c) & valid
            if union.any():
                inter = (pred_c & mask_c) & valid
                ious.append(inter.sum().float() / union.sum().float())

        if save_dir is not None and saved < max_images:
            _save_seg_panel(surface[0], mask[0], pred[0],
                            save_dir / f"eval_{n:05d}.png", num_classes)
            saved += 1
        n += 1

    if save_dir is not None:
        print(f"Saved {saved} panels to {save_dir.resolve()}")

    return {
        "loss": total_loss / max(n, 1),
        "miou": sum(ious) / max(len(ious), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HD-EMS")
    parser.add_argument("--config", type=str, default="configs/evimo_seg.yaml")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-images", type=str, default=None,
                        help="directory to write events|GT|prediction panels (segmentation only)")
    parser.add_argument("--max-images", type=int, default=50,
                        help="max number of eval panels to save")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = HDEMS(cfg).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)

    task = cfg.get("train", {}).get("task", "flow")
    eval_split = cfg.get("eval", {}).get("split", cfg.get("dataset", {}).get("eval_split", "eval"))

    try:
        dataset = build_dataset(cfg, split=eval_split)
        if len(dataset) == 0:
            raise FileNotFoundError("Eval dataset is empty")
        print(f"Eval samples ({eval_split}): {len(dataset)}")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        if task == "segmentation":
            num_classes = cfg.get("segmentation", {}).get("num_classes", 16)
            save_dir = Path(args.save_images) if args.save_images else None
            metrics = evaluate_segmentation(
                model, loader, device, num_classes,
                save_dir=save_dir, max_images=args.max_images,
            )
            print(f"Seg loss: {metrics['loss']:.4f}")
            print(f"mIoU:     {metrics['miou']:.4f}")
        else:
            epe = evaluate_flow(model, loader, device)
            print(f"Mean EPE: {epe:.4f}")
    except FileNotFoundError as e:
        print(f"Dataset not ready: {e}")


if __name__ == "__main__":
    main()
