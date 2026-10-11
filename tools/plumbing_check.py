"""Exercise the run plumbing end to end: config validation, set_seed, tier
check, manifest. No data, no science. Usage:

    PYTHONPATH=src python tools/plumbing_check.py --config plumbing_check [--smoke]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars import config as C  # noqa: E402
from burnscars.hardware import check_tier  # noqa: E402
from burnscars.manifest import RunRecorder  # noqa: E402
from burnscars.paths import outputs_dir  # noqa: E402
from burnscars.seed import set_seed  # noqa: E402

TIER = "cpu"


def main(argv=None):
    ap = C.base_parser(__doc__.splitlines()[0])
    C.require(ap.add_argument("--seed", type=int))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args, name, cfg = C.parse(ap, argv)
    check_tier(TIER, args.smoke)
    seed_info = set_seed(args.seed)
    n = 10 if args.smoke else args.n

    rec = RunRecorder("plumbing_check", name, cfg, args.seed, args.smoke)
    x = np.random.rand(n)
    metrics = {"n": n, "mean": float(x.mean())}
    if args.smoke:
        print("smoke example:", x[:3])
    out = Path(args.out) if args.out else outputs_dir() / "plumbing_check" / name
    path = rec.finish(out, metrics, extra={"seed_info": seed_info})
    print(f"manifest: {path}")
    return path


if __name__ == "__main__":
    main()
