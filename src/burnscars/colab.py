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


def restore_data(config: str = "download_hls_burn_scars") -> None:
    """Local copy of the dataset: from the Drive tarball cache if present (SHA-checked by
    download_data), else downloaded once and the tarball copied to Drive for next time."""
    import yaml
    cfg = yaml.safe_load((REPO_ROOT / "configs" / f"{config}.yaml").read_text())
    local = Path(os.environ["BURNSCARS_DATA_DIR"]) / cfg["subdir"]
    cache = DRIVE_ROOT / "data_cache" / cfg["subdir"]
    local.mkdir(parents=True, exist_ok=True)
    for fname in cfg["files"]:
        if (cache / fname).exists() and not (local / fname).exists():
            (local / fname).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache / fname, local / fname)
    run("download_data", config=config)  # verifies size + SHA256, extracts
    cache.mkdir(parents=True, exist_ok=True)
    for fname in cfg["files"]:
        if not (cache / fname).exists():
            (cache / fname).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local / fname, cache / fname)


def start_allocation(alloc_id: str, gpu_hours: float) -> float:
    """Arm an APPROVED allocation (checked against docs/COMPUTE_LEDGER.md by gpu-tier tools)
    and set a hard wall-clock deadline; training stops at a checkpoint when it passes."""
    import time
    from .hardware import approved_allocations
    if alloc_id not in approved_allocations():
        raise RuntimeError(f"{alloc_id} is not APPROVED in docs/COMPUTE_LEDGER.md at this commit")
    deadline = time.time() + gpu_hours * 3600
    os.environ["BURNSCARS_ALLOCATION"] = alloc_id
    os.environ["BURNSCARS_DEADLINE_UNIX"] = str(deadline)
    print(f"{alloc_id} armed: hard stop in {gpu_hours} h")
    return deadline
