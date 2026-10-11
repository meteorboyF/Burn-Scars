"""Split membership: official train/val, IBM train/val/test, and the test quarantine.

IBM's 120-chip test set is QUARANTINED. Its membership list may be read only by
the callers in `_TEST_ALLOWED`, and every read is appended to
docs/QUARANTINE_LOG.md (committed, so access is auditable in git history):

  audit_footprints : geometry only (footprints / distances); no pixels, no labels,
                     no model outputs.
  final_eval       : tools/final_eval.py, which refuses to run without --final.

tests/test_quarantine.py fails if any other file references the test list.
"""
import time
from pathlib import Path

from ..manifest import git_sha
from ..paths import REPO_ROOT, data_dir

IBM_DIR = "ibm_splits/splits"
_TEST_ALLOWED = {"audit_footprints", "final_eval"}
QUARANTINE_LOG = REPO_ROOT / "docs" / "QUARANTINE_LOG.md"


class QuarantineError(PermissionError):
    pass


def _read_ids(path: Path) -> list[str]:
    ids = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ids")
    return ids


def ibm_ids(name: str) -> list[str]:
    """IBM train or val ids (form `T10SEJ.2018185.v1`). Test is refused here."""
    if name not in ("train", "val"):
        raise QuarantineError(f"ibm_ids({name!r}): only 'train'/'val'; test is quarantined")
    return _read_ids(data_dir() / IBM_DIR / f"{name}.txt")


def ibm_test_ids(caller: str, purpose: str) -> list[str]:
    """Quarantined IBM test ids. Allowed callers only; every call is logged."""
    if caller not in _TEST_ALLOWED:
        raise QuarantineError(f"caller {caller!r} may not read the IBM test split")
    if not purpose.strip():
        raise QuarantineError("a purpose string is required")
    ids = _read_ids(data_dir() / IBM_DIR / "test.txt")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"| {stamp} | {caller} | {purpose} | {git_sha()} | {len(ids)} |\n"
    if not QUARANTINE_LOG.exists():
        QUARANTINE_LOG.write_text(
            "# QUARANTINE LOG: every read of the IBM test split\n\n"
            "| UTC | Caller | Purpose | git SHA | n ids |\n|---|---|---|---|---|\n")
    with open(QUARANTINE_LOG, "a") as f:
        f.write(line)
    return ids


class EvalLeakError(PermissionError):
    pass


# Training-split families that may ever be scored on IBM test. Official-split
# models are excluded by rule: all 120 IBM-test chips lie inside official train/val.
_IBM_TEST_ELIGIBLE = {"ibm", "cv_random_684", "cv_grouped_684"}


def assert_eval_allowed(train_split: str, train_ids, eval_name: str, eval_ids) -> None:
    """Refuse an evaluation that would leak. Called by every evaluation entrypoint.

    1. name rule: eval on "ibm_test" only for models whose train_split family is
       in _IBM_TEST_ELIGIBLE (never "official").
    2. set rule: training chips and evaluation chips must be disjoint, always.
    """
    if eval_name == "ibm_test" and train_split not in _IBM_TEST_ELIGIBLE:
        raise EvalLeakError(f"model trained on {train_split!r} may never be evaluated on IBM test")
    overlap = set(train_ids) & set(eval_ids)
    if overlap:
        raise EvalLeakError(f"{len(overlap)} training chips are in the evaluation set {eval_name!r}, "
                            f"e.g. {sorted(overlap)[:3]}")
