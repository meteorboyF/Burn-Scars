"""The IBM test split may only be referenced from the allowed files."""
import re
import subprocess
import sys

import pytest

from burnscars.data import splits
from burnscars.paths import REPO_ROOT

ALLOWED = {
    "src/burnscars/data/splits.py",
    "tools/final_eval.py",
    "tools/audit_footprints.py",
    "tests/test_quarantine.py",
    "configs/download_ibm_splits.yaml",   # fetch + checksum only
}
PATTERN = re.compile(r"test\.txt|ibm_test_ids")


def test_only_allowed_files_reference_ibm_test():
    offenders = []
    for p in REPO_ROOT.rglob("*"):
        rel = p.relative_to(REPO_ROOT).as_posix()
        if not p.is_file() or rel.split("/")[0] in {".git", ".venv", "data", "outputs", "docs"}:
            continue
        if p.suffix not in {".py", ".yaml", ".yml", ".ipynb", ".sh", ".txt", ".toml"}:
            continue
        if PATTERN.search(p.read_text(errors="ignore")) and rel not in ALLOWED:
            offenders.append(rel)
    assert not offenders, f"files referencing the quarantined IBM test split: {offenders}"


def test_ibm_ids_refuses_test():
    with pytest.raises(splits.QuarantineError):
        splits.ibm_ids("test")


def test_test_ids_refuses_unknown_caller():
    with pytest.raises(splits.QuarantineError):
        splits.ibm_test_ids("train_unet", "peek")


def test_final_eval_refuses_without_flag(tmp_path):
    cfg = tmp_path / "f.yaml"
    cfg.write_text("purpose: x\n")
    r = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "final_eval.py"),
                        "--config", str(cfg)], capture_output=True, text=True)
    assert r.returncode != 0 and "REFUSED" in r.stderr
