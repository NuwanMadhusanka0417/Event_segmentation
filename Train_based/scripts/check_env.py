"""Pre-flight check of the Python environment, GPU, code and data (NCI Gadi).

Why: on Gadi /scratch, files nobody READS for 100 days are quarantined. Python
packages ship files that are only ever checked for existence (torch/bin/
torch_shm_manager) or rarely used (pip's __main__.py, the venv's python link), so
a venv decays piece by piece and every job then dies with a different error. This
script finds all of it at once, in ~1 minute, before a long job is wasted:

  1. interpreter   python comes from the venv, the venv's python link is not dangling
  2. files         every file each installed package lists in its RECORD exists
                   (stat only: it does not refresh the files' access time)
  3. imports       torch, numpy, scipy, yaml, sklearn, matplotlib, typing_extensions
  4. torch         torch_shm_manager present; with --require-cuda: GPU, matmul, conv,
                   cuDNN, and a 2-worker DataLoader (shared-memory tensors)
  5. project       hdems imports, a tiny model runs forward on the device
  6. data          dataset.root has train/eval sequences, IMU calibration present

    python scripts/check_env.py --config configs/evimo_seg.yaml [--require-cuda]

Exit code 0 = everything OK, 1 = something failed (listed at the end).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILS: list[str] = []
WARNS: list[str] = []


def ok(msg: str) -> None:
    print(f"  OK    {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    FAILS.append(msg)


def warn(msg: str) -> None:
    print(f"  WARN  {msg}")
    WARNS.append(msg)


def check_interpreter() -> None:
    print("[1] interpreter")
    venv = os.environ.get("VIRTUAL_ENV")
    print(f"      executable {sys.executable}  prefix {sys.prefix}")
    if sys.prefix == sys.base_prefix:
        fail("not running inside a virtual environment (sys.prefix == base_prefix)")
    else:
        ok(f"virtual environment {sys.prefix}")
    if venv and Path(venv).resolve() != Path(sys.prefix).resolve():
        fail(f"VIRTUAL_ENV={venv} but python runs from {sys.prefix}")
    bindir = Path(sys.prefix) / "bin"
    for name in ("python", "python3"):
        p = bindir / name
        if p.is_symlink() and not p.exists():
            fail(f"{p} is a broken link -> {os.readlink(p)}")
        elif not p.exists():
            fail(f"{p} missing")
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if cfg.exists():
        home = next((ln.split("=", 1)[1].strip() for ln in cfg.read_text().splitlines()
                     if ln.strip().startswith("home")), None)
        if home and not (Path(home) / "python3").exists():
            fail(f"pyvenv.cfg home = {home} has no python3")
        elif home:
            ok(f"pyvenv.cfg home = {home}")


def check_files() -> None:
    print("[2] package files (RECORD vs disk)")
    venv = Path(sys.prefix).resolve()
    n_dist = n_files = 0
    for dist in md.distributions():
        try:
            loc = Path(dist.locate_file("")).resolve()
        except Exception:
            continue
        if venv not in loc.parents and loc != venv:
            continue                                   # system / module packages
        files = dist.files or []
        missing = []
        for f in files:
            s = str(f)
            if s.endswith(".pyc") or "__pycache__" in s:
                continue                               # regenerated on import
            n_files += 1
            if not Path(dist.locate_file(f)).exists():
                missing.append(s)
        n_dist += 1
        name = dist.metadata.get("Name", "?")
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            fail(f"{name} {dist.version}: {len(missing)} file(s) missing: {shown}")
    if n_dist == 0:
        fail("no packages found inside the venv")
    else:
        print(f"      {n_dist} packages, {n_files} files checked")


def check_imports() -> None:
    print("[3] imports")
    for mod in ("torch", "numpy", "scipy", "yaml", "sklearn", "matplotlib", "typing_extensions"):
        try:
            m = importlib.import_module(mod)
            ok(f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:                          # noqa: BLE001
            fail(f"import {mod}: {type(e).__name__}: {e}")


def _square(i):                                         # DataLoader worker payload
    import torch
    return torch.full((64, 64), float(i))


def check_torch(require_cuda: bool) -> str:
    print("[4] torch")
    try:
        import torch
    except Exception as e:                              # noqa: BLE001
        fail(f"import torch: {e}")
        return "cpu"
    shm = Path(torch.__file__).parent / "bin" / "torch_shm_manager"
    (ok if shm.exists() else fail)(f"torch_shm_manager {'present' if shm.exists() else 'MISSING: ' + str(shm)}")
    print(f"      torch {torch.__version__}  built for CUDA {torch.version.cuda}")
    dev = "cpu"
    if torch.cuda.is_available():
        dev = "cuda"
        ok(f"CUDA available: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
        try:
            a = torch.randn(1024, 1024, device=dev)
            b = (a @ a).sum().item()
            x = torch.randn(2, 4, 64, 64, device=dev)
            y = torch.nn.functional.conv2d(x, torch.randn(8, 4, 3, 3, device=dev)).sum().item()
            torch.cuda.synchronize()
            ok(f"GPU matmul + conv run (cuDNN {torch.backends.cudnn.version()})"
               if b == b and y == y else "GPU result is NaN")
        except Exception as e:                          # noqa: BLE001
            fail(f"GPU compute: {e}")
    elif require_cuda:
        fail("CUDA not available (--require-cuda): no GPU, or torch/driver mismatch")
    else:
        warn("CUDA not available (fine on a login node; use --require-cuda on a GPU node)")
    try:
        from torch.utils.data import DataLoader
        got = [t for t in DataLoader(range(4), batch_size=None, num_workers=2,
                                     collate_fn=None, worker_init_fn=None,
                                     multiprocessing_context="fork" if os.name != "nt" else None)]
        ok(f"DataLoader with 2 workers ({len(got)} batches)") if len(got) == 4 else fail("DataLoader lost batches")
    except Exception as e:                              # noqa: BLE001
        fail(f"DataLoader with workers: {e}")
    return dev


def check_project(dev: str) -> None:
    print("[5] project code")
    try:
        import torch
        from hdems.models.hdems import HDEMS
        import hdems.eval, hdems.imu, hdems.protocols  # noqa: F401,E401
        cfg = {
            "d": 32,
            "encoder": {"patch_size": 5, "sigma_k": 1.0, "kernel": "conv",
                        "polarity_binding": True, "scales": 1},
            "matching": {"M": 5, "scales": [0, 1, 2], "alpha": 0.85, "smooth": 3},
            "dataset": {"height": 16, "width": 20, "time_frames": [1.0, 0.75, 0.5, 0.0]},
            "segmentation": {"head": "mfunet", "num_classes": 2, "mf_motion_dim": 8,
                             "mf_widths": [8, 16, 16, 16]},
            "velocity": {"event_feature": "f"},
            "motion": {"fine_smooth": 3},
            "flow_cache": {"enabled": False},
        }
        model = HDEMS(cfg).to(dev).eval()
        s = (torch.rand(1, 4, 2, 16, 20, device=dev) > 0.7).float()
        with torch.no_grad():
            out = model(s, task="segmentation", rot_flow=torch.zeros(1, 2, 16, 20, device=dev))
        ok(f"hdems imports, tiny mfunet forward on {dev}: {tuple(out['seg_logits'].shape)}")
    except Exception as e:                              # noqa: BLE001
        fail(f"project code: {type(e).__name__}: {e}")


def check_data(config: str | None) -> None:
    print("[6] data")
    if not config:
        warn("no --config given: data not checked")
        return
    try:
        from hdems.config import frontend_option
        from hdems.data.evimo2_reader import find_sequence_dirs
        from hdems.eval import load_config
        cfg = load_config(config)
        root = Path(cfg["dataset"]["root"])
        for split, expect in (("train", 21), ("eval", 8)):
            seqs = find_sequence_dirs(root, split)
            missing = [s.name for s in seqs
                       if not all((s / f).exists() for f in ("dataset_info.npz", "dataset_events_t.npy",
                                                             "dataset_events_xy.npy", "dataset_events_p.npy"))]
            msg = f"{root / split}: {len(seqs)} sequences with masks (expected {expect})"
            (ok if len(seqs) >= expect and not missing else fail)(
                msg + (f", incomplete: {missing}" if missing else ""))
        if str(frontend_option(cfg, "motion", "rot_source")) == "imu":
            calib = Path(frontend_option(cfg, "motion", "imu_calib"))
            (ok if calib.exists() else warn)(
                f"IMU calibration {calib} {'present' if calib.exists() else 'missing (run_segmentation.pbs creates it)'}")
    except Exception as e:                              # noqa: BLE001
        fail(f"data check: {type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--require-cuda", action="store_true", help="fail if no usable GPU")
    ap.add_argument("--skip-files", action="store_true", help="skip the RECORD scan")
    args = ap.parse_args()
    t0 = time.time()
    check_interpreter()
    if not args.skip_files:
        check_files()
    check_imports()
    dev = check_torch(args.require_cuda)
    check_project(dev)
    check_data(args.config)
    print(f"\n=== check_env: {'OK' if not FAILS else f'{len(FAILS)} FAILED'}"
          f"{f', {len(WARNS)} warning(s)' if WARNS else ''}  ({time.time() - t0:.0f} s) ===")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
