"""Filesystem locations. Data never lives inside the package or in git."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Root for downloaded data. Override with BURNSCARS_DATA_DIR (e.g. a Colab local disk)."""
    return Path(os.environ.get("BURNSCARS_DATA_DIR", REPO_ROOT / "data"))


def outputs_dir() -> Path:
    """Root for run outputs. Override with BURNSCARS_OUTPUTS_DIR."""
    return Path(os.environ.get("BURNSCARS_OUTPUTS_DIR", REPO_ROOT / "outputs"))


def configs_dir() -> Path:
    return REPO_ROOT / "configs"


def committed_manifests_dir() -> Path:
    """Small manifests of cited runs are committed here; figures/tables read only these."""
    return REPO_ROOT / "results" / "manifests"
