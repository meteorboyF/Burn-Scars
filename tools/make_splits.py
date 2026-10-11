"""Build the versioned split files: official, IBM train/val, 5-fold random and grouped CV.

    PYTHONPATH=src python tools/make_splits.py --config make_splits_v1 [--smoke]

Inputs are committed, data-free tables (results/profile/chips_profile.csv: bounds,
CRS, burned fraction) plus, for the IBM-based pool, IBM train/val membership.
The IBM test list is never read: pool "ibm_trainval" = IBM train ∪ IBM val.
Outputs splits/<version>/*.json and SHA256SUMS; tests/test_splits.py pins them.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars import config as C  # noqa: E402
from burnscars.data import cv  # noqa: E402
from burnscars.data.footprints import (pairwise_distance, reproject_footprint,  # noqa: E402
                                       single_linkage_groups)
from burnscars.data.splits import ibm_ids  # noqa: E402
from burnscars.hardware import check_tier  # noqa: E402
from burnscars.manifest import RunRecorder, sha256_json  # noqa: E402
from burnscars.paths import REPO_ROOT  # noqa: E402

TIER = "cpu"


def dump(path: Path, obj) -> str:
    text = json.dumps(obj, indent=1, sort_keys=True) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def build_cv(pool: pd.DataFrame, k, k_inner, group_km, seed):
    ids = pool.chip_id.tolist()
    bf = pool.burned_frac.fillna(0).to_numpy()
    polys = [reproject_footprint((r.left, r.bottom, r.right, r.top), r.crs) for r in pool.itertuples()]
    D = pairwise_distance(polys, polys)
    np.fill_diagonal(D, np.inf)
    groups = single_linkage_groups(D, group_km * 1000)

    assign = {"random": cv.stratified_random_folds(ids, bf, k, seed),
              "grouped": cv.grouped_folds(ids, groups, bf, k, seed)}
    members = {r: cv.fold_members(a, k) for r, a in assign.items()}
    train_sizes = [len(ids) - len(m) for r in members for m in members[r]]
    n_train = min(train_sizes)
    pos = {c: i for i, c in enumerate(ids)}

    out = {}
    for r in ("random", "grouped"):
        folds = []
        for f in range(k):
            test = members[r][f]
            full_train = sorted(set(ids) - set(test))
            train, dropped = cv.subsample(full_train, n_train, seed * 1000 + f)
            # inner folds within the (equal-size) training set, same regime logic
            tr_bf = bf[[pos[c] for c in train]]
            if r == "random":
                ia = cv.stratified_random_folds(train, tr_bf, k_inner, seed * 1000 + 100 + f)
            else:
                tr_groups = groups[[pos[c] for c in train]]
                ia = cv.grouped_folds(train, tr_groups, tr_bf, k_inner, seed * 1000 + 100 + f)
            inner = [{"inner_fold": j, "val": m, "train": sorted(set(train) - set(m))}
                     for j, m in enumerate(cv.fold_members(ia, k_inner))]
            folds.append({"fold": f, "test": test, "train": train, "train_dropped": dropped,
                          "inner": inner})
        out[r] = folds
    meta = {"group_of": {c: int(g) for c, g in zip(ids, groups)}, "n_train_per_fold": n_train,
            "n_groups": int(groups.max() + 1)}
    return out, meta


def main(argv=None):
    ap = C.base_parser(__doc__.splitlines()[0])
    C.require(ap.add_argument("--version"))
    C.require(ap.add_argument("--profile_csv"))
    C.require(ap.add_argument("--pools", type=list))
    C.require(ap.add_argument("--k", type=int))
    C.require(ap.add_argument("--k_inner", type=int))
    C.require(ap.add_argument("--group_km", type=float))
    C.require(ap.add_argument("--seed", type=int))
    ap.add_argument("--out_dir", default=None, help="CLI only: write elsewhere (regeneration tests)")
    args, name, cfg = C.parse(ap, argv)
    check_tier(TIER, args.smoke)
    rec = RunRecorder("make_splits", name, cfg, seed=args.seed, smoke=args.smoke)

    prof = pd.read_csv(REPO_ROOT / args.profile_csv).sort_values("chip_id").reset_index(drop=True)
    out_dir = REPO_ROOT / "splits" / args.version if not args.smoke else REPO_ROOT / "outputs" / "smoke" / name
    if args.out_dir:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {"version": args.version, "seed": args.seed, "dataset": "ibm-nasa-geospatial/hls_burn_scars@1864285"}
    sums = {}

    off = {**common, "kind": "official", "folds": [{
        "fold": 0, "train": sorted(prof[prof.official_split == "train"].chip_id),
        "test": sorted(prof[prof.official_split == "val"].chip_id)}]}
    sums["official.json"] = dump(out_dir / "official.json", off)
    ibm_tr, ibm_va = sorted(ibm_ids("train")), sorted(ibm_ids("val"))
    sums["ibm_trainval.json"] = dump(out_dir / "ibm_trainval.json", {
        **common, "kind": "ibm", "source": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars@a3f2c41",
        "folds": [{"fold": 0, "train": ibm_tr, "test": ibm_va}],
        "note": "IBM test split is quarantined and deliberately absent"})

    stats = {}
    for pool_name in args.pools:
        if pool_name == "all804":
            pool = prof
        elif pool_name == "ibm_trainval":
            pool = prof[prof.chip_id.isin(set(ibm_tr) | set(ibm_va))].reset_index(drop=True)
        else:
            raise SystemExit(f"unknown pool {pool_name}")
        if args.smoke:
            pool = pool.iloc[:60]
        regimes, meta = build_cv(pool, args.k, args.k_inner, args.group_km, args.seed)
        for r, folds in regimes.items():
            fname = f"{r}_cv{args.k}_{pool_name}.json"
            sums[fname] = dump(out_dir / fname, {
                **common, "kind": f"{r}_cv", "pool": pool_name, "k": args.k, "k_inner": args.k_inner,
                "group_km": args.group_km, "n_pool": len(pool), **meta, "folds": folds})
            bfm = dict(zip(pool.chip_id, pool.burned_frac.fillna(0)))
            stats[fname] = {"test_sizes": [len(f["test"]) for f in folds],
                            "train_size": meta["n_train_per_fold"],
                            "test_burned_frac_mean": [round(float(np.mean([bfm[c] for c in f["test"]])), 4)
                                                      for f in folds],
                            "n_groups": meta["n_groups"]}
    (out_dir / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(sums.items())))
    rec.data["split_hash"] = sha256_json(sums)
    rec.finish(out_dir / "_manifest" if not args.smoke else out_dir, {"sha256": sums, "stats": stats})
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
