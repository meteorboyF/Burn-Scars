"""Helpers called by notebooks/runner.ipynb. All runner logic lives here, not in cells.

Rules (02_CODEBASE_AND_INFRA.md): HF caches and outputs on LOCAL disk, copied to
Drive afterwards; Drive is FUSE and fails under load; "Transport endpoint is not
connected" means a dead runtime, not a retry.
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .paths import REPO_ROOT

LOCAL_ROOT = Path("/content/local")
DRIVE_ROOT = Path("/content/drive/MyDrive/burnscars")


def configure_env() -> dict:
    env = {
        "HF_HOME": "/content/hf_cache",
        "BURNSCARS_DATA_DIR": str(LOCAL_ROOT / "data"),
        "BURNSCARS_OUTPUTS_DIR": str(LOCAL_ROOT / "outputs"),
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    os.environ.update(env)
    for v in ("HF_HOME", "BURNSCARS_DATA_DIR", "BURNSCARS_OUTPUTS_DIR"):
        Path(env[v]).mkdir(parents=True, exist_ok=True)
    return env


def ensure_requirements(req_file: str = "requirements.lock") -> str:
    """pip install only when the lockfile fingerprint changed since the last install."""
    req = REPO_ROOT / req_file
    fp = hashlib.sha256(req.read_bytes()).hexdigest()[:16]
    marker = Path("/content") / f".burnscars_req_{fp}"
    if marker.exists():
        print(f"requirements unchanged ({fp}); skipping install")
        return fp
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)
    marker.touch()
    print(f"installed requirements ({fp})")
    return fp


def print_sha() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    print(f"git SHA: {sha}{'-dirty' if dirty else ''}")
    return sha


def run(tool: str, config: str, *extra: str, smoke: bool = False) -> None:
    """Run tools/<tool>.py as a subprocess with PYTHONPATH=src."""
    cmd = [sys.executable, str(REPO_ROOT / "tools" / f"{tool}.py"), "--config", config, *extra]
    if smoke:
        cmd.append("--smoke")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def copy_to_drive(local_subdir: str) -> Path:
    """Copy LOCAL_ROOT/outputs/<subdir> to Drive (never write in place on Drive)."""
    src = LOCAL_ROOT / "outputs" / local_subdir
    dst = DRIVE_ROOT / "outputs" / local_subdir
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"copied {src} -> {dst}")
    return dst
