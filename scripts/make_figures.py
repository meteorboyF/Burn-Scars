"""Build every paper figure from committed manifests. Vector PDF only.

    PYTHONPATH=src python scripts/make_figures.py [--only NAME] [--allow-dirty]

Each figure is a function registered in FIGURES; it receives the list of
validated manifests and writes paper/figures/<name>.pdf. No number is typed
by hand anywhere in the paper.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars.manifest import load_manifest  # noqa: E402
from burnscars.paths import REPO_ROOT, committed_manifests_dir  # noqa: E402

FIG_DIR = REPO_ROOT / "paper" / "figures"

# name -> callable(manifests: list[dict], out_pdf: Path) -> None
FIGURES = {}


def load_all(allow_dirty: bool) -> list[dict]:
    paths = sorted(committed_manifests_dir().rglob("manifest.json"))
    return [load_manifest(p, require_clean=not allow_dirty) for p in paths]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="draft builds only; never for submission")
    args = ap.parse_args(argv)
    manifests = load_all(args.allow_dirty)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    todo = {k: v for k, v in FIGURES.items() if args.only in (None, k)}
    if not todo:
        print(f"no figures registered yet ({len(manifests)} manifests loaded)")
    for name, fn in todo.items():
        out = FIG_DIR / f"{name}.pdf"
        fn(manifests, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
