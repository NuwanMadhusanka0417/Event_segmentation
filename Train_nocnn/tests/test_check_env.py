"""scripts/check_env.py: finds files that vanished from a pip-installed package
(the NCI /scratch expiry: "Unable to find torch_shm_manager")."""

from __future__ import annotations

import importlib.util
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_env", ROOT / "scripts" / "check_env.py")
check_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_env)


def _fake_dist(tmp_path: Path, installer: str, version: str = "2.3.1+cu121") -> metadata.Distribution:
    """A 'torch' installed by ``installer`` whose RECORD lists files that are gone."""
    files = {
        "torch/__init__.py": True,
        "torch/bin/torch_shm_manager": False,        # the file the Gadi job could not find
        "torch/lib/libc10.so": True,
        "torch/include/c10/core/Device.h": False,    # headers: never read, harmless
        "torch/__pycache__/__init__.cpython-39.pyc": False,
        "../../../bin/torchrun": False,              # console script outside site-packages
    }
    for rel, present in files.items():
        if present:
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_text("x")
    info = tmp_path / f"torch-{version.split('+')[0]}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: torch\nVersion: {version}\n")
    (info / "INSTALLER").write_text(installer + "\n")
    (info / "RECORD").write_text("".join(f"{rel},,\n" for rel in files))
    return metadata.PathDistribution(info)


def test_missing_runtime_file_is_found(tmp_path):
    runtime, other = check_env.missing_files(_fake_dist(tmp_path, "pip"))
    assert runtime == ["torch/bin/torch_shm_manager"]
    assert sorted(other) == ["../../../bin/torchrun",
                             "torch/__pycache__/__init__.cpython-39.pyc",
                             "torch/include/c10/core/Device.h"]


def test_conda_packages_are_not_checked(tmp_path):
    # conda moves files after writing RECORD (torch DLLs -> Library/bin)
    assert check_env.missing_files(_fake_dist(tmp_path, "conda")) is None


def test_pip_spec_keeps_the_cuda_build(tmp_path):
    assert check_env.pip_spec(_fake_dist(tmp_path, "pip")) == (
        "torch==2.3.1+cu121", "https://download.pytorch.org/whl/cu121")


def test_wrong_venv_is_caught(tmp_path):
    # a renamed venv: `source seg1/bin/activate` still runs the old venv's python
    import sys
    assert check_env.venv_problem(Path(sys.prefix)) is None
    assert check_env.venv_problem(Path(sys.prefix) / "bin" / "activate") is None
    assert "not the venv" in check_env.venv_problem(tmp_path / "seg1" / "bin" / "activate")


def test_pip_spec_plain_pypi(tmp_path):
    assert check_env.pip_spec(_fake_dist(tmp_path, "pip", version="2.3.1")) == ("torch==2.3.1", "")
