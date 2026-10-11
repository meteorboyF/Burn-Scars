"""Final evaluation on IBM's quarantined 120-chip test split. Refuses without --final.

    PYTHONPATH=src python tools/final_eval.py --config <final_eval_cfg> --final

The only tool allowed to compute predictions/metrics on the IBM test split.
Every use is logged in docs/QUARANTINE_LOG.md. Model evaluation is wired in
Phase C; until then it stops after the gate so no number on test can exist.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars import config as C  # noqa: E402
from burnscars.data.splits import ibm_test_ids  # noqa: E402
from burnscars.hardware import check_tier  # noqa: E402

TIER = "gpu"


def main(argv=None):
    ap = C.base_parser(__doc__.splitlines()[0])
    ap.add_argument("--final", action="store_true", help="required: confirms a final test evaluation")
    C.require(ap.add_argument("--purpose"))
    args, name, cfg = C.parse(ap, argv)
    if not args.final:
        raise SystemExit("REFUSED: final_eval reads the quarantined IBM test split; pass --final")
    if args.smoke:
        raise SystemExit("REFUSED: no smoke mode on the quarantined test split")
    check_tier(TIER, args.smoke)
    ids = ibm_test_ids("final_eval", f"{name}: {args.purpose}")
    raise NotImplementedError(f"model evaluation lands in Phase C ({len(ids)} test ids loaded)")


if __name__ == "__main__":
    main()
