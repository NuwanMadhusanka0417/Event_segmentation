#!/usr/bin/env python3
"""Pre-flight check: is the Python environment intact, and is the data there?

Every PBS script runs this before any real work, so a broken environment stops the
job in seconds with the fix printed, instead of a traceback hours into the run:

    python scripts/check_env.py --venv /scratch/jq77/nk8155/seg1 \
        --config configs/evimo_seg.yaml --require-cuda
    python scripts/check_env.py --pip-specs     # broken packages as pip specs (repair_env.sh)

Why: NCI Gadi quarantines /scratch files that nothing has READ for 100 days and
deletes them 14 days later. A venv on /scratch keeps every file a run opens, but
loses the ones nothing reads -- e.g. torch/bin/torch_shm_manager, which
``import torch`` only checks for and never opens:

    RuntimeError: Unable to find torch_shm_manager at .../torch/bin/torch_shm_manager

Checks
  1. --venv: ``python`` is that venv's own interpreter. A renamed or copied venv
     keeps its OLD path in bin/activate, so ``source seg1/bin/activate`` can still
     run the old venv's python (with the old venv's missing files). Stops here if not.
  2. every file pip installed for the packages this code uses (and their
     dependencies) still exists -- read from each package's RECORD
  3. each package really imports, in a subprocess (a crash is reported, not fatal)
  4. --require-cuda: a GPU is visible (the entry points silently fall back to CPU)
  5. --config: dataset.root holds the train / eval / validation sequences with all
     their files (the reader SILENTLY skips a sequence without dataset_mask.npz)

Steps 1-2 use the standard library only, so they run even when torch is broken.
Exit status 1 if anything is broken.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

# Distributions the code imports: requirements.txt + matplotlib (figures).
REQUIRED = ("torch", "numpy", "pyyaml", "scipy", "matplotlib")
OPTIONAL = ("scikit-learn",)              # only ridge.backend: sklearn

# (label, code, required) -- each runs in a fresh interpreter from the project root
IMPORT_CHECKS = (
    ("numpy", "import numpy", True),
    ("yaml", "import yaml", True),
    ("torch", "import torch", True),
    ("scipy", "import scipy.ndimage, scipy.optimize, scipy.spatial", True),
    ("matplotlib", "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot, "
                   "matplotlib.offsetbox, matplotlib.patches, matplotlib.path", True),
    ("sklearn", "import sklearn.linear_model", False),
    ("hdems", "import hdems.eval, hdems.vsa_grouping, hdems.grouping", True),
)
CUDA_PROBE = ("import torch; c = torch.cuda.is_available(); "
              "print(torch.__version__, torch.version.cuda, int(c), torch.cuda.device_count()); "
              "c and torch.ones(1, device='cuda').add_(1).item()")

# Missing files that cannot break a run: headers, stubs, tests, static libraries,
# bytecode Python rewrites, console scripts outside site-packages. Reported only.
NON_RUNTIME_SUFFIXES = {".pyc", ".pyi", ".h", ".hpp", ".hxx", ".cuh", ".inl", ".c", ".cc",
                        ".cpp", ".cu", ".pxd", ".pyx", ".a", ".cmake", ".md", ".rst"}
NON_RUNTIME_DIRS = {"__pycache__", "include", "tests"}

# Files every EVIMO2 sequence needs (hdems/data/evimo2_reader.py)
SEQ_FILES = ("dataset_mask.npz", "dataset_info.npz", "dataset_events_t.npy",
             "dataset_events_xy.npy", "dataset_events_p.npy")
DEPTH_FILE = "dataset_depth.npz"          # ground-truth flow only (scripts/flow_epe.py)

# Packages whose CUDA builds (2.3.1+cu121) live on the PyTorch index, not PyPI
TORCH_INDEX = {"torch", "torchvision", "torchaudio"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def installed() -> dict[str, metadata.Distribution]:
    """name -> distribution, the first on sys.path (the one ``import`` loads)."""
    out: dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name and _norm(name) not in out:
            out[_norm(name)] = dist
    return out


def closure(names, dists: dict[str, metadata.Distribution]) -> list[str]:
    """``names`` plus everything they depend on (extras skipped), as installed."""
    todo, seen = [_norm(n) for n in names], []
    while todo:
        name = todo.pop()
        if name in seen or name not in dists:
            continue
        seen.append(name)
        for req in dists[name].requires or []:
            if re.search(r"\bextra\s*==", req):
                continue
            m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
            if m:
                todo.append(_norm(m.group(1)))
    return seen


def _non_runtime(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (parts[0] == ".." or any(p in NON_RUNTIME_DIRS for p in parts[:-1])
            or PurePosixPath(path).suffix in NON_RUNTIME_SUFFIXES)


def missing_files(dist: metadata.Distribution) -> tuple[list[str], list[str]] | None:
    """(runtime, other) files in the package's RECORD that no longer exist; None
    when it cannot be checked: no RECORD, or installed by conda, which moves files
    after writing it (e.g. torch's DLLs to Library/bin)."""
    if (dist.read_text("RECORD") is None
            or (dist.read_text("INSTALLER") or "").strip().lower() == "conda"):
        return None
    runtime: list[str] = []
    other: list[str] = []
    for f in dist.files or []:
        if not os.path.exists(dist.locate_file(f)):
            (other if _non_runtime(str(f)) else runtime).append(str(f))
    return runtime, other


def pip_spec(dist: metadata.Distribution) -> tuple[str, str]:
    """``name==version`` and the extra index that hosts it ('' = PyPI)."""
    name, version = dist.metadata["Name"], dist.version
    local = version.partition("+")[2]                   # 2.3.1+cu121 -> cu121
    index = (f"https://download.pytorch.org/whl/{local}"
             if local and _norm(name) in TORCH_INDEX else "")
    return f"{name}=={version}", index


def _run(code: str) -> tuple[bool, str]:
    """Run ``code`` in a fresh interpreter; (ok, stdout or the error's last line)."""
    try:
        p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                           text=True, timeout=900, env={**os.environ, "MPLBACKEND": "Agg"})
    except subprocess.TimeoutExpired:
        return False, "timed out after 900 s"
    if p.returncode == 0:
        return True, p.stdout.strip()
    lines = [ln for ln in p.stderr.strip().splitlines() if ln.strip()]
    return False, lines[-1] if lines else f"exit status {p.returncode}"


def check_packages(dists: dict[str, metadata.Distribution], errors: list[str]) -> list[str]:
    """Step 2. Returns the names of packages with missing runtime files."""
    names = closure(REQUIRED + OPTIONAL, dists)
    for name in REQUIRED:
        if _norm(name) not in dists:
            errors.append(f"package {name} is not installed")
    broken: list[str] = []
    unchecked: list[str] = []
    harmless = 0
    for name in names:
        found = missing_files(dists[name])
        if found is None:
            unchecked.append(name)
            continue
        runtime, other = found
        harmless += len(other)
        if runtime:
            broken.append(name)
            d = dists[name]
            print(f"  FAIL {d.metadata['Name']} {d.version}: {len(runtime)} file(s) missing, e.g.")
            for f in runtime[:5]:
                print(f"         {f}")
            errors.append(f"{d.metadata['Name']} {d.version} has {len(runtime)} missing file(s)")
    print(f"[files]  {len(names)} packages checked ({', '.join(REQUIRED)} + dependencies): "
          f"{len(broken) or 'none'} broken")
    if harmless:
        print(f"         ({harmless} headers/tests/stubs missing as well -- harmless; "
              "reinstalling restores them)")
    if unchecked:
        print(f"         not checked (no RECORD, or installed by conda): {', '.join(unchecked)}")
    return broken


def check_imports(require_cuda: bool, errors: list[str]) -> None:
    """Steps 3-4."""
    results = []
    for label, code, required in IMPORT_CHECKS:
        ok, msg = _run(code)
        results.append(f"{label} {'ok' if ok else 'FAIL'}")
        if not ok:
            print(f"  FAIL import {label}: {msg}")
            if required:
                errors.append(f"import {label} fails: {msg}")
            else:
                print(f"       ({label} is optional: only needed for ridge.backend: sklearn)")
    print(f"[import] {' | '.join(results)}")

    ok, msg = _run(CUDA_PROBE)
    if not ok:
        if not any(e.startswith("import torch") for e in errors):
            errors.append(f"CUDA probe fails: {msg}")
        return
    version, cuda, available, count = (msg.splitlines()[0].split() + ["?"] * 4)[:4]
    print(f"[cuda]   torch {version} (CUDA {cuda}): "
          f"{count} GPU(s) visible" if available == "1" else
          f"[cuda]   torch {version} (CUDA {cuda}): NO GPU visible")
    if require_cuda and available != "1":
        errors.append("no GPU visible to torch (the scripts would silently run on the CPU) "
                      "-- check '#PBS -q gpuvolta -l ngpus=...' and 'module load cuda'")


def venv_dir(path: Path) -> Path:
    """The venv folder, given the folder itself or its bin/activate."""
    return path.parent.parent if path.name == "activate" else path


def venv_problem(expected: Path) -> str | None:
    """Why ``python`` is not ``expected``'s own interpreter; None when it is."""
    want = venv_dir(expected)
    if os.path.realpath(sys.prefix) == os.path.realpath(want):
        return None
    return (f"python is {sys.executable}, not the venv {want}'s own "
            f"(its bin/activate sets VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV', '(not set)')})")


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def check_data(config: Path, need_depth: bool, errors: list[str]) -> None:
    """Step 5: the sequences hdems/data/build.py would read for this config."""
    try:
        import yaml
        cfg = yaml.safe_load(config.read_text(encoding="utf-8-sig")) or {}
    except Exception as e:                       # yaml broken or config unreadable
        errors.append(f"cannot read {config}: {e}")
        return
    ds = cfg.get("dataset") or {}
    if ds.get("name", "dsec") != "evimo":
        print(f"[data]   dataset.name={ds.get('name')!r}: not checked")
        return
    root = Path(ds["root"])
    print(f"[data]   dataset.root = {ds['root']}  ->  {os.path.abspath(root)}")
    if not root.is_dir():
        errors.append(f"dataset.root not found: {os.path.abspath(root)} (relative paths start "
                      f"at the code dir) -- edit dataset.root in {config} or link the data there")
        return
    need = SEQ_FILES + ((DEPTH_FILE,) if need_depth else ())
    train = ds.get("split", "train")
    found: dict[str, list[Path]] = {}
    for split in dict.fromkeys([train, ds.get("eval_split", "eval")]):
        folder = root / split
        seqs = sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
        found[split] = seqs
        if not seqs:
            errors.append(f"no sequences in {folder}")
            continue
        bad = {p.name: [f for f in need if not _nonempty(p / f)] for p in seqs}
        bad = {name: files for name, files in bad.items() if files}
        print(f"         {split}/: {len(seqs) - len(bad)} of {len(seqs)} sequences complete")
        for name, files in bad.items():
            errors.append(f"{split}/{name} is missing {', '.join(files)}")
    if str(ds.get("val_split", "")) == "holdout":
        scenes = {str(s).lower() for s in ds.get("val_scenes") or []}
        have = {p.name.split("_", 1)[0].lower() for p in found.get(train, [])}
        print(f"         holdout (validation) scenes {sorted(scenes)}: "
              f"{'present' if scenes <= have else 'MISSING'} in {train}/")
        if not scenes <= have:
            errors.append(f"validation scenes {sorted(scenes - have)} not found in {root / train}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venv", type=Path, default=None,
                    help="the venv this must run in (its folder or bin/activate): stop if "
                         "python is another interpreter (renamed / copied venv)")
    ap.add_argument("--config", type=Path, default=None,
                    help="also check the dataset of this config")
    ap.add_argument("--require-cuda", action="store_true",
                    help="fail if torch sees no GPU (PBS GPU jobs)")
    ap.add_argument("--require-depth", action="store_true",
                    help=f"sequences also need {DEPTH_FILE} (scripts/flow_epe.py)")
    ap.add_argument("--pip-specs", action="store_true",
                    help="only print 'name==version<TAB>extra-index' for every package with "
                         "missing files (used by scripts/repair_env.sh)")
    args = ap.parse_args()

    wrong_venv = venv_problem(args.venv) if args.venv is not None else None
    dists = installed()
    if args.pip_specs:
        if wrong_venv:               # never "repair" another venv than the one asked for
            print(f"check_env.py: {wrong_venv}", file=sys.stderr)
            return 1
        for name in closure(REQUIRED + OPTIONAL, dists):
            found = missing_files(dists[name])
            if found and found[0]:
                print("\t".join(pip_spec(dists[name])))
        return 0

    errors: list[str] = []
    venv = os.environ.get("VIRTUAL_ENV")
    print("=" * 63)
    print(" PRE-FLIGHT CHECK  (scripts/check_env.py)")
    print("=" * 63)
    print(f"[python] {sys.version.split()[0]}  {sys.executable}")
    if wrong_venv:
        want = venv_dir(args.venv)
        print("=" * 63)
        print(f" PRE-FLIGHT FAILED:\n   - {wrong_venv}")
        print(f"""
 HOW TO FIX  (a renamed or copied venv keeps its OLD path in bin/activate)
   Point bin/activate at the venv's real folder, then open a new shell:
        sed -i 's#^VIRTUAL_ENV=.*#VIRTUAL_ENV="{want}"#' {want}/bin/activate
   Use 'python -m pip', not 'pip': pip's first line still names the old path.""")
        print("=" * 63)
        return 1
    if not venv:
        print("  WARNING: no virtualenv is activated")
    elif os.path.realpath(sys.prefix) != os.path.realpath(venv):
        print(f"  WARNING: python is not the venv's interpreter "
              f"(sys.prefix {sys.prefix}, VIRTUAL_ENV {venv})")
    broken = check_packages(dists, errors)
    check_imports(args.require_cuda, errors)
    if args.config is not None:
        check_data(args.config, args.require_depth, errors)

    print("=" * 63)
    if not errors:
        print(" PRE-FLIGHT OK")
        print("=" * 63)
        return 0
    print(" PRE-FLIGHT FAILED:")
    for e in errors:
        print(f"   - {e}")
    if broken or any(e.startswith(("import", "CUDA")) for e in errors):
        activate = (venv_dir(args.venv) if args.venv else Path(venv or sys.prefix)) / "bin" / "activate"
        print(f"""
 HOW TO FIX THE ENVIRONMENT  (on a Gadi LOGIN node: compute nodes have no internet)
   NCI quarantines /scratch files that nothing has read for 100 days and deletes
   them 14 days later -- a venv on /scratch loses the files no run ever opens.
   1. Still in quarantine (< 14 days)? Restore them as they were:
        nci-file-expiry list-quarantined
        nci-file-expiry recover <UUID>
   2. Otherwise reinstall only the broken packages, same versions:
        bash scripts/repair_env.sh {activate}
   3. Long term: keep the venv on /g/data (no expiry) -- README, "Setup (NCI Gadi)".""")
    if any(("sequences" in e or "dataset" in e or "missing dataset_" in e) for e in errors):
        print("""
 HOW TO FIX THE DATA
   Point dataset.root (configs/*.yaml; relative paths start at the code dir) at the
   EVIMO2 samsung_mono/imo folder, or link it, e.g. from the code dir:
        ln -s /path/to/Data ../Data
   The data on /scratch has the same 100-day expiry: restore missing files with
   nci-file-expiry, or copy them again.""")
    print("=" * 63)
    return 1


if __name__ == "__main__":
    sys.exit(main())
