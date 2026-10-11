"""The IBM test split may only be referenced from the allowed files."""
import json
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
    cfg.write_text("purpose: x\ntrain_manifest: none\n")
    r = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "final_eval.py"),
                        "--config", str(cfg)], capture_output=True, text=True)
    assert r.returncode != 0 and "REFUSED" in r.stderr


def test_official_trained_model_never_scored_on_ibm_test():
    official = json.loads((REPO_ROOT / "splits" / "v1" / "official.json").read_text())["folds"][0]
    with pytest.raises(splits.EvalLeakError, match="never"):
        splits.assert_eval_allowed("official", official["train"], "ibm_test", ["T00XXX.2000001.v1"])
    # even with an empty eval list (the check runs before the test ids are read)
    with pytest.raises(splits.EvalLeakError):
        splits.assert_eval_allowed("official", official["train"], "ibm_test", [])


def test_train_eval_overlap_always_refused():
    with pytest.raises(splits.EvalLeakError, match="training chips"):
        splits.assert_eval_allowed("cv_grouped_684", ["a", "b"], "fold_test", ["b", "c"])
    splits.assert_eval_allowed("cv_grouped_684", ["a"], "ibm_test", ["c"])  # allowed


def test_official_arm_is_scored_on_official_val_only():
    official = json.loads((REPO_ROOT / "splits" / "v1" / "official.json").read_text())["folds"][0]
    splits.assert_eval_allowed("official", official["train"], "official_val", official["test"])


def test_final_eval_refuses_official_model(tmp_path):
    from burnscars.manifest import RunRecorder
    rec = RunRecorder("train_unet", "x", {"a": 1}, seed=0, smoke=False)
    rec.data["git_sha"] = "0" * 40
    m = rec.finish(tmp_path / "run", {}, extra={"train_split": "official", "train_ids": ["a"]})
    cfg = tmp_path / "f.yaml"
    cfg.write_text(f"purpose: x\ntrain_manifest: {m}\n")
    r = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "final_eval.py"), "--config",
                        str(cfg), "--final"], capture_output=True, text=True,
                       env={**__import__("os").environ, "BURNSCARS_ALLOCATION": ""})
    assert r.returncode != 0 and "may never be evaluated on IBM test" in r.stderr
