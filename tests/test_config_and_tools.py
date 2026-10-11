import subprocess
import sys

import pytest

from burnscars import config as C
from burnscars.hardware import TierError, approved_allocations, check_tier
from burnscars.manifest import load_manifest
from burnscars.paths import REPO_ROOT


def _parser():
    ap = C.base_parser("t")
    C.require(ap.add_argument("--seed", type=int))
    ap.add_argument("--lr", type=float, default=0.1)
    return ap


def _cfg(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return str(p)


def test_config_merges_and_cli_overrides(tmp_path):
    path = _cfg(tmp_path, "seed: 3\nlr: 0.5\n")
    args, name, cfg = C.parse(_parser(), ["--config", path])
    assert (args.seed, args.lr, name) == (3, 0.5, "c")
    args, _, _ = C.parse(_parser(), ["--config", path, "--lr", "0.01"])
    assert args.lr == 0.01


def test_unknown_key_fails(tmp_path):
    with pytest.raises(C.ConfigError, match="unknown keys"):
        C.parse(_parser(), ["--config", _cfg(tmp_path, "seed: 1\nlearning_rate: 2\n")])


def test_reserved_key_fails(tmp_path):
    with pytest.raises(C.ConfigError, match="CLI-only"):
        C.parse(_parser(), ["--config", _cfg(tmp_path, "seed: 1\nsmoke: true\n")])


def test_missing_required_fails(tmp_path):
    with pytest.raises(C.ConfigError, match="missing required"):
        C.parse(_parser(), ["--config", _cfg(tmp_path, "lr: 1\n")])


def test_tiers(monkeypatch):
    check_tier("cpu", smoke=False)
    check_tier("gpu", smoke=True)
    monkeypatch.setattr("burnscars.hardware.cuda_available", lambda: True)
    monkeypatch.delenv("BURNSCARS_ALLOCATION", raising=False)
    with pytest.raises(TierError):  # GPU present but no approved allocation
        check_tier("gpu", smoke=False)


def test_approved_allocations_parse(tmp_path):
    led = tmp_path / "L.md"
    led.write_text("| A-001 | pilot | 3 GPU-h | APPROVED |\n| A-002 | full | 30 | REQUESTED |\n")
    assert approved_allocations(led) == {"A-001"}


def test_plumbing_check_smoke_subprocess(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "plumbing_check.py"),
                        "--config", "plumbing_check", "--smoke", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = load_manifest(out)
    assert m["smoke"] is True and m["seed"] == 0 and m["metrics"]["n"] == 10
