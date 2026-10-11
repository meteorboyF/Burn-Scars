import pytest

from burnscars import checkpoint as ck


def test_save_and_resume_latest(tmp_path):
    for s in (1, 2, 9, 10):
        ck.save(tmp_path, s, 100, f"state{s}".encode(), keep_last=10)
    step, payload = ck.latest_valid(tmp_path, 100)
    assert step == 10 and payload == b"state10"  # numeric sort: 10 > 9


def test_skips_corrupt_and_partial(tmp_path):
    ck.save(tmp_path, 1, 100, b"good", keep_last=10)
    ck.save(tmp_path, 2, 100, b"tobecorrupted", keep_last=10)
    (tmp_path / "step_2" / "payload.bin").write_bytes(b"truncated")
    (tmp_path / "step_3").mkdir()  # a directory existing is not a checkpoint
    assert ck.latest_valid(tmp_path, 100) == (1, b"good")


def test_refuses_max_steps_mismatch(tmp_path):
    ck.save(tmp_path, 5, 100, b"x")
    with pytest.raises(ck.CheckpointError):
        ck.latest_valid(tmp_path, 200)


def test_prune_keeps_last(tmp_path):
    for s in range(5):
        ck.save(tmp_path, s, 10, b"x", keep_last=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_3", "step_4"]


def test_empty_dir_returns_none(tmp_path):
    assert ck.latest_valid(tmp_path / "missing", 10) is None
