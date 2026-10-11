import json

import pytest

from burnscars.manifest import ManifestError, RunRecorder, load_manifest


def _write(tmp_path, **kw):
    rec = RunRecorder("t", "cfg", {"a": 1}, seed=0, smoke=False, **kw)
    return rec.finish(tmp_path, {"iou": 0.5})


def test_roundtrip_and_complete_is_last(tmp_path):
    p = _write(tmp_path)
    d = load_manifest(p)
    assert list(d)[-1] == "complete" and d["complete"] is True
    assert d["metrics"] == {"iou": 0.5}


def test_refuses_incomplete(tmp_path):
    p = _write(tmp_path)
    d = json.loads(p.read_text())
    d["complete"] = False
    p.write_text(json.dumps(d))
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_refuses_complete_not_last(tmp_path):
    p = _write(tmp_path)
    d = json.loads(p.read_text())
    d = {"complete": True, **{k: v for k, v in d.items() if k != "complete"}}
    p.write_text(json.dumps(d))
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_refuses_tampered_config(tmp_path):
    p = _write(tmp_path)
    d = json.loads(p.read_text())
    d["config"]["a"] = 2
    p.write_text(json.dumps(d))
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_refuses_missing_file(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "nope.json")


def test_require_clean_refuses_dirty_and_smoke(tmp_path):
    p = _write(tmp_path)
    d = json.loads(p.read_text())
    d["git_sha"] = "abc-dirty"
    p.write_text(json.dumps(d))
    with pytest.raises(ManifestError):
        load_manifest(p, require_clean=True)
    load_manifest(p)  # fine when not cited


def test_require_data_hashes(tmp_path):
    p = _write(tmp_path)
    with pytest.raises(ManifestError):
        load_manifest(p, require_data_hashes=True)
    p2 = _write(tmp_path / "b", dataset_hash="d", split_hash="s")
    load_manifest(p2, require_data_hashes=True)
