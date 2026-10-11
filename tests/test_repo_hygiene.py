import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from burnscars.manifest import load_manifest, sha256_json
from burnscars.paths import REPO_ROOT, committed_manifests_dir

needs_git = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(), reason="not a git checkout")


def _ignored(path):
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT).returncode == 0


@needs_git
def test_data_gitignore_is_root_anchored():
    assert _ignored("data/hls_burn_scars.tar.gz")
    assert _ignored("data/chips/x.tif")
    assert not _ignored("src/burnscars/data/__init__.py")
    assert not _ignored("src/burnscars/data/new_module.py")


def test_gitignore_has_no_unanchored_data_pattern():
    lines = [l.strip() for l in (REPO_ROOT / ".gitignore").read_text().splitlines()]
    assert "/data/" in lines
    assert "data/" not in lines and "data" not in lines


def test_runner_notebook_has_no_outputs():
    nb = json.loads((REPO_ROOT / "notebooks" / "runner.ipynb").read_text())
    for cell in nb["cells"]:
        assert not cell.get("outputs")
        assert cell.get("execution_count") is None


def test_committed_manifests_match_immutable_configs():
    """A config that produced a committed manifest must not have changed since."""
    for p in sorted(committed_manifests_dir().rglob("manifest.json")):
        m = load_manifest(p)
        cfg_path = REPO_ROOT / "configs" / f"{m['config_name']}.yaml"
        assert cfg_path.is_file(), f"{p}: config {cfg_path} was deleted"
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        assert sha256_json(cfg) == m["config_hash"], f"{cfg_path} changed after producing {p}"


def test_requirements_are_pinned():
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            assert "==" in line, f"unpinned requirement: {line}"
