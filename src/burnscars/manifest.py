"""Run manifests: the single source every paper number is built from.

A manifest records config/dataset/split hashes, git SHA (with -dirty), seed,
environment, wall clock, the metric values, and `complete: true` as the LAST
key. It is written atomically (tmp file + rename). Loading re-validates and
REFUSES (raises) with a printed reason; there is no warn-and-continue path.
"""
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .paths import REPO_ROOT

SCHEMA_VERSION = 1
REQUIRED_KEYS = (
    "schema_version", "tool", "config_name", "config_hash", "dataset_hash",
    "split_hash", "git_sha", "seed", "smoke", "environment", "started_utc",
    "wall_clock_s", "metrics", "complete",
)
_ENV_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "rasterio", "pyproj",
                 "shapely", "matplotlib", "torch", "terratorch", "segmentation-models-pytorch")


class ManifestError(RuntimeError):
    """Raised when a manifest is missing, incomplete or fails validation."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj) -> str:
    """Hash of a canonical JSON encoding (sorted keys, no whitespace)."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def git_sha(repo=REPO_ROOT) -> str:
    """HEAD SHA, suffixed with -dirty if tracked files differ. 'nogit' outside a repo."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"
    return sha + ("-dirty" if dirty else "")


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "machine": platform.machine(), "packages": {}}
    for pkg in _ENV_PACKAGES:
        try:
            env["packages"][pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    try:
        import torch  # noqa: PLC0415
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        env["cuda"] = None
        env["gpu"] = None
    return env


class RunRecorder:
    """Collects manifest fields during a run; `finish()` writes it with complete=true LAST."""

    def __init__(self, tool: str, config_name: str, config: dict, seed, smoke: bool,
                 dataset_hash=None, split_hash=None):
        self._t0 = time.time()
        self.data = {
            "schema_version": SCHEMA_VERSION,
            "tool": tool,
            "config_name": config_name,
            "config_hash": sha256_json(config),
            "config": config,
            "dataset_hash": dataset_hash,
            "split_hash": split_hash,
            "git_sha": git_sha(),
            "seed": seed,
            "smoke": smoke,
            "environment": environment(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._t0)),
            "metrics": {},
        }

    def finish(self, out_dir, metrics: dict, extra: dict | None = None) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        d = dict(self.data)
        d["metrics"] = metrics
        if extra:
            d["extra"] = extra
        d["wall_clock_s"] = round(time.time() - self._t0, 3)
        d.pop("complete", None)
        d["complete"] = True  # insertion order: always the last key
        path = out_dir / "manifest.json"
        tmp = out_dir / ".manifest.json.tmp"
        tmp.write_text(json.dumps(d, indent=2, sort_keys=False) + "\n")
        os.replace(tmp, path)
        return path


def load_manifest(path, require_clean: bool = False, require_data_hashes: bool = False) -> dict:
    """Load and validate. Raises ManifestError (and prints the reason) on any problem."""
    def refuse(msg):
        print(f"REFUSED manifest {path}: {msg}", file=sys.stderr)
        raise ManifestError(f"{path}: {msg}")

    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        refuse("file does not exist")
    try:
        d = json.loads(path.read_text())
    except ValueError as e:
        refuse(f"invalid JSON ({e})")
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        refuse(f"missing keys {missing}")
    if d["complete"] is not True or list(d)[-1] != "complete":
        refuse("not complete (complete must be true and the last key)")
    if d["schema_version"] != SCHEMA_VERSION:
        refuse(f"schema_version {d['schema_version']} != {SCHEMA_VERSION}")
    if "config" in d and sha256_json(d["config"]) != d["config_hash"]:
        refuse("config_hash does not match embedded config")
    if require_clean and (d["git_sha"].endswith("-dirty") or d["git_sha"] == "nogit"):
        refuse(f"git_sha {d['git_sha']!r} is not a clean commit (required for cited results)")
    if require_clean and d["smoke"]:
        refuse("smoke runs cannot be cited")
    if require_data_hashes and (not d["dataset_hash"] or not d["split_hash"]):
        refuse("dataset_hash and split_hash are required")
    return d
