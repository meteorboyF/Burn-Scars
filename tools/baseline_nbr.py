"""CPU trivial baselines + leakage control: all-unburned and NBR threshold (tuned on train only).

    PYTHONPATH=src python tools/baseline_nbr.py --config baseline_nbr_v1 [--smoke]

For every split file and every fold: accumulate the NBR histogram over the
fold's TRAINING chips, pick the threshold maximising dataset-level burned IoU
on those chips, then score the held-out chips (never seen during tuning).
Writes per-chip raw scores (CSV) and per-fold aggregates (both metric
conventions) plus a manifest.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars import config as C  # noqa: E402
from burnscars.data.chips import build_index, dataset_hash  # noqa: E402
from burnscars.eval.baselines import NBRHistogram, nbr, predict_all_unburned, predict_nbr  # noqa: E402
from burnscars.eval.metrics import aggregate, chip_scores  # noqa: E402
from burnscars.hardware import check_tier  # noqa: E402
from burnscars.manifest import RunRecorder, sha256_file, sha256_json  # noqa: E402
from burnscars.paths import REPO_ROOT, data_dir  # noqa: E402

TIER = "cpu"


def load(root, row):
    with rasterio.open(root / row.scene_path) as s:
        x = s.read().astype(np.float32)
    with rasterio.open(root / row.mask_path) as m:
        y = m.read(1).astype(np.int64)
    return x, y


def main(argv=None):
    ap = C.base_parser(__doc__.splitlines()[0])
    C.require(ap.add_argument("--subdir"))
    C.require(ap.add_argument("--split_files", type=list))
    C.require(ap.add_argument("--out"))
    args, name, cfg = C.parse(ap, argv)
    check_tier(TIER, args.smoke)
    root = data_dir() / args.subdir
    rec = RunRecorder("baseline_nbr", name, cfg, seed=None, smoke=args.smoke)
    idx = build_index(root).set_index("chip_id")

    # one pass: NBR index + target per chip, kept compactly (float16 index, int8 target)
    cache = {}
    needed = set()
    splits = {}
    for sf in args.split_files:
        spec = json.loads((REPO_ROOT / sf).read_text())
        folds = spec["folds"][:1] if args.smoke else spec["folds"]
        if args.smoke:
            folds = [{"fold": f["fold"], "train": f["train"][:8], "test": f["test"][:4]} for f in folds]
        splits[Path(sf).stem] = folds
        for f in folds:
            needed |= set(f["train"]) | set(f["test"])
    for cid in sorted(needed):
        x, y = load(root, idx.loc[cid].rename(cid))
        cache[cid] = (nbr(x).astype(np.float32), y.astype(np.int8))

    per_chip, per_fold = [], []
    for split_name, folds in splits.items():
        for f in folds:
            h = NBRHistogram()
            for cid in f["train"]:
                h.add(*cache[cid])
            t, train_iou = h.best_threshold()
            scores = {"nbr": [], "all_unburned": []}
            for cid in f["test"]:
                ind, y = cache[cid]
                y = y.astype(np.int64)
                for method, pred in (("nbr", predict_nbr(ind, t)), ("all_unburned", predict_all_unburned(y))):
                    s = chip_scores(y, pred)
                    scores[method].append(s)
                    per_chip.append({"split": split_name, "fold": f["fold"], "method": method,
                                     "chip_id": cid, "iou_burned": s["iou_burned"],
                                     "iou_unburned": s["iou_unburned"], "miou": s["miou"],
                                     "n_valid": s["n_valid"], "cm": json.dumps(s["cm"].tolist())})
            for method, sc in scores.items():
                agg = aggregate(sc)
                per_fold.append({"split": split_name, "fold": f["fold"], "method": method,
                                 "threshold": t if method == "nbr" else None,
                                 "train_dataset_iou_burned": train_iou if method == "nbr" else None,
                                 "n_train": len(f["train"]), "n_test": len(f["test"]),
                                 **{k: v for k, v in agg.items() if k != "confusion"},
                                 "confusion": json.dumps(agg["confusion"])})
    out = REPO_ROOT / args.out if not args.smoke else REPO_ROOT / "outputs" / "smoke" / name
    out.mkdir(parents=True, exist_ok=True)
    pc = pd.DataFrame(per_chip)
    pf = pd.DataFrame(per_fold)
    pc.to_csv(out / "per_chip.csv", index=False, float_format="%.6g")
    pf.to_csv(out / "per_fold.csv", index=False, float_format="%.6g")
    summary = {}
    for (sp, m), g in pf.groupby(["split", "method"]):
        summary[f"{sp}/{m}"] = {c: {"mean": float(g[c].mean()), "sd": float(g[c].std(ddof=1)) if len(g) > 1 else None}
                                for c in ("dataset_iou_burned", "dataset_miou", "perchip_iou_burned", "perchip_miou")}
        if m == "nbr":
            summary[f"{sp}/{m}"]["thresholds"] = g.threshold.tolist()
    rec.data["dataset_hash"] = dataset_hash(root, idx.loc[sorted(needed)].reset_index())
    rec.data["split_hash"] = sha256_json({sf: sha256_file(REPO_ROOT / sf) for sf in args.split_files})
    rec.finish(out, summary)
    print(pf[["split", "fold", "method", "threshold", "dataset_iou_burned", "dataset_miou",
              "perchip_iou_burned"]].to_string())


if __name__ == "__main__":
    main()
