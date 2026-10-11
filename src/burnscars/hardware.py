"""Cost-tier tags. Each entrypoint declares a TIER and calls `check_tier` first.

  cpu  : runs anywhere.
  gpu  : needs CUDA. Refuses on a CPU-only host unless --smoke.

Colab compute is shared with other projects: a gpu-tier tool additionally
refuses to start a non-smoke run unless BURNSCARS_ALLOCATION names an
allocation id that was approved and recorded in docs/COMPUTE_LEDGER.md.
"""
import os
import re

from .paths import REPO_ROOT

TIERS = ("cpu", "gpu")


class TierError(RuntimeError):
    pass


def cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def approved_allocations(ledger=None) -> set[str]:
    """Allocation ids marked APPROVED in the compute ledger, e.g. `| A-001 | ... | APPROVED |`."""
    ledger = ledger or (REPO_ROOT / "docs" / "COMPUTE_LEDGER.md")
    try:
        text = open(ledger).read()
    except OSError:
        return set()
    out = set()
    header = None
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if "Status" in cells and cells[0] == "Id":
            header = cells
        elif header and re.fullmatch(r"A-\d{3}", cells[0]) and len(cells) == len(header):
            if cells[header.index("Status")] == "APPROVED":  # exact cell, not a substring
                out.add(cells[0])
    return out


def check_tier(tier: str, smoke: bool) -> None:
    if tier not in TIERS:
        raise TierError(f"unknown tier {tier!r}")
    if tier == "cpu":
        return
    if smoke:
        return  # smoke mode is CPU by contract
    if not cuda_available():
        raise TierError("gpu-tier tool started without CUDA; use --smoke on CPU")
    alloc = os.environ.get("BURNSCARS_ALLOCATION")
    if not alloc or alloc not in approved_allocations():
        raise TierError(
            f"BURNSCARS_ALLOCATION={alloc!r} is not an APPROVED allocation in "
            "docs/COMPUTE_LEDGER.md; GPU runs need explicit approval")
