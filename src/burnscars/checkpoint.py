"""Framework-agnostic checkpoint/resume with content validation.

Layout: <ckpt_root>/step_<N>/{payload.bin, meta.json}. meta.json is written
LAST and carries the payload's SHA256, the step, and the run's max_steps. A
directory existing is NOT a checkpoint: `latest_valid` skips anything whose
meta is missing, whose hash mismatches, or whose payload is absent. Steps are
sorted numerically (step_10 > step_9). Resuming refuses when the recorded
max_steps disagrees with the current config.

Trainers supply (de)serialisers, e.g. torch.save/torch.load to bytes.
"""
import json
import os
import re
import shutil
from pathlib import Path

from .manifest import sha256_bytes

_STEP_RE = re.compile(r"^step_(\d+)$")


class CheckpointError(RuntimeError):
    pass


def save(ckpt_root, step: int, max_steps: int, payload: bytes, keep_last: int = 2,
         extra_meta: dict | None = None) -> Path:
    root = Path(ckpt_root)
    final = root / f"step_{step}"
    tmp = root / f".tmp_step_{step}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / "payload.bin").write_bytes(payload)
    meta = {"step": step, "max_steps": max_steps, "payload_sha256": sha256_bytes(payload),
            "payload_bytes": len(payload), **(extra_meta or {})}
    (tmp / "meta.json").write_text(json.dumps(meta))
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    _prune(root, keep_last)
    return final


def _steps(root: Path) -> list[tuple[int, Path]]:
    if not root.is_dir():
        return []
    out = []
    for p in root.iterdir():
        m = _STEP_RE.match(p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    return sorted(out)  # numeric, not lexicographic


def _prune(root: Path, keep_last: int):
    for _, p in _steps(root)[:-keep_last]:
        shutil.rmtree(p, ignore_errors=True)


def validate(path: Path) -> dict | None:
    """Return meta if the checkpoint at `path` is intact, else None."""
    meta_p, pay_p = path / "meta.json", path / "payload.bin"
    if not (meta_p.is_file() and pay_p.is_file()):
        return None
    try:
        meta = json.loads(meta_p.read_text())
    except ValueError:
        return None
    payload = pay_p.read_bytes()
    if sha256_bytes(payload) != meta.get("payload_sha256") or len(payload) != meta.get("payload_bytes"):
        return None
    return meta


def latest_valid(ckpt_root, max_steps: int) -> tuple[int, bytes] | None:
    """Newest intact checkpoint as (step, payload), or None. Refuses max_steps mismatch."""
    for step, p in reversed(_steps(Path(ckpt_root))):
        meta = validate(p)
        if meta is None:
            print(f"checkpoint {p} failed validation; skipping")
            continue
        if meta["max_steps"] != max_steps:
            raise CheckpointError(
                f"{p}: recorded max_steps={meta['max_steps']} != config max_steps={max_steps}; "
                "refusing to resume (start a new run dir)")
        return step, (p / "payload.bin").read_bytes()
    return None
