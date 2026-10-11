"""Footprint leakage audit of the official split and the IBM split (geometry only).

    PYTHONPATH=src python tools/audit_footprints.py --config audit_footprints [--smoke]

Reads each chip's bounds + CRS with rasterio (headers only), reprojects
footprints to EPSG:5070, and for each split computes: train/held-out area
overlap, nearest-neighbour distance from each held-out chip to the training
set, same-footprint repeats, and shared MGRS tiles. Writes a per-chip CSV, a
pairs CSV, summary.json, vector-PDF histograms, docs/AUDIT_REPORT.md (generated,
never hand-edited) and a manifest. Reads the quarantined IBM test MEMBERSHIP
list for geometry only (logged); no pixels, labels or model outputs.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars import config as C  # noqa: E402
from burnscars.data.chips import build_index, dataset_hash  # noqa: E402
from burnscars.data.footprints import (DIST_CRS, geodesic_nearest, pairwise_distance,  # noqa: E402
                                       pairwise_overlap, reproject_footprint,
                                       single_linkage_groups)
from burnscars.data.splits import ibm_ids, ibm_test_ids  # noqa: E402
from burnscars.hardware import check_tier  # noqa: E402
from burnscars.manifest import RunRecorder, sha256_json  # noqa: E402
from burnscars.paths import REPO_ROOT, data_dir  # noqa: E402

TIER = "cpu"


def read_footprints(root, idx):
    polys, crs, centroids = [], [], []
    to_ll = Transformer.from_crs(DIST_CRS, "EPSG:4326", always_xy=True)
    for r in idx.itertuples():
        with rasterio.open(root / r.scene_path) as s:
            b, c = tuple(s.bounds), s.crs.to_string()
        p = reproject_footprint(b, c)
        polys.append(p)
        crs.append(c)
        cx, cy = p.centroid.x, p.centroid.y
        centroids.append(to_ll.transform(cx, cy))
    return np.array(polys, dtype=object), crs, np.array(centroids)


def _n_pairs(group_sizes) -> int:
    k = np.asarray(group_sizes, dtype=np.int64)
    return int((k * (k - 1) // 2).sum())


def held_out_stats(name, polys, D, IOU, ids, held, train, geodesic=True):
    """Per held-out chip: nearest train chip, distance, max footprint IoU, area covered."""
    hi = np.array([i for i, x in enumerate(ids) if x in held])
    ti = np.array([i for i, x in enumerate(ids) if x in train])
    sub = D[np.ix_(hi, ti)]
    nn = sub.argmin(1)
    rows = []
    for k, i in enumerate(hi):
        j = ti[nn[k]]
        touching = ti[sub[k] == 0]
        cov = 0.0
        if len(touching):
            u = shapely.union_all(polys[touching])
            cov = shapely.area(shapely.intersection(polys[i], u)) / shapely.area(polys[i])
        rows.append({"chip_id": ids[i], f"{name}_nn_train": ids[j],
                     f"{name}_nn_dist_km": sub[k, nn[k]] / 1000,
                     f"{name}_nn_dist_geod_km": (geodesic_nearest(polys[i], polys[j]) / 1000
                                                 if geodesic else np.nan),
                     f"{name}_max_iou_train": IOU[i, ti].max(),
                     f"{name}_area_covered_by_train": cov,
                     f"{name}_n_train_touching": len(touching)})
    return pd.DataFrame(rows)


def split_summary(name, df, idx, held, train, pairs, repeat_iou, thresholds=(0, 1, 2.5, 5, 10, 20)):
    d = df[f"{name}_nn_dist_km"].to_numpy()
    g = df[f"{name}_nn_dist_geod_km"].to_numpy()
    tiles_h = set(idx[idx.chip_id.isin(held)].tile)
    tiles_t = set(idx[idx.chip_id.isin(train)].tile)
    cross = pairs[(pairs.a.isin(held) & pairs.b.isin(train)) | (pairs.b.isin(held) & pairs.a.isin(train))]
    rep = cross[cross.iou >= repeat_iou]
    s = {"n_held_out": len(df), "n_train": len(train),
         "frac_within_km": {f"{t:g}": float(np.mean(d <= t)) for t in thresholds},
         "n_within_km": {f"{t:g}": int(np.sum(d <= t)) for t in thresholds},
         "frac_within_km_geodesic": {f"{t:g}": float(np.mean(g <= t)) for t in thresholds},
         "nn_dist_km_quantiles": {str(q): float(np.quantile(d, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
         "n_overlapping_train": int((df[f"{name}_n_train_touching"] > 0).sum()),
         "mean_area_covered_by_train": float(df[f"{name}_area_covered_by_train"].mean()),
         "n_area_covered_ge_50pct": int((df[f"{name}_area_covered_by_train"] >= .5).sum()),
         "n_cross_pairs_overlapping": int(len(cross)),
         "n_same_footprint_repeats_iou_ge_0.9": int(len(rep)),
         "n_same_footprint_repeats_diff_date": int((rep.date_a != rep.date_b).sum()),
         "n_same_footprint_repeats_same_date": int((rep.date_a == rep.date_b).sum()),
         "n_held_out_chips_in_a_repeat": int(len(set(rep.a) & held | set(rep.b) & held)),
         "tiles_held_out": len(tiles_h), "tiles_train": len(tiles_t),
         "tiles_shared": len(tiles_h & tiles_t), "tiles_total": len(tiles_h | tiles_t),
         "n_held_out_chips_on_shared_tiles": int(idx[idx.chip_id.isin(held)].tile.isin(tiles_t).sum())}
    return s


def hist_pdf(path, series, labels, title):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bins = np.r_[0, np.logspace(-1, np.log10(max(1000, max(s.max() for s in series) + 1)), 40)]
    styles = ["-", "--", ":"]
    for k, (s, lab) in enumerate(zip(series, labels)):
        ax.hist(np.clip(s, 0.05, None), bins=bins, histtype="step", linestyle=styles[k % 3],
                linewidth=1.2, label=f"{lab} (n={len(s)})")
    ax.axvline(5, color="k", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("distance to nearest training chip (km; 0 plotted at 0.05)", fontsize=7)
    ax.set_ylabel("chips", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(path, s, cfg_name, manifest_rel):
    o, ib = s["official"], s["ibm"]
    k1 = s["K1"]

    def pct(x):
        return f"{100 * x:.1f}%"

    lines = [
        "# Footprint audit report (GENERATED by tools/audit_footprints.py; do not edit)", "",
        f"Config `{cfg_name}`; manifest `{manifest_rel}`. Distances: min edge-to-edge between",
        f"chip footprints in {DIST_CRS}; geodesic cross-check on the same nearest pair.", "",
        "## K1 gate (pre-registered 2026-10-11, docs/REALITY_CHECK.md)", "",
        f"- p_near = official-val chips within 5 km of an official-train chip: "
        f"**{o['n_within_km']['5']}/{o['n_held_out']} = {pct(o['frac_within_km']['5'])}** "
        f"(geodesic: {pct(o['frac_within_km_geodesic']['5'])}); threshold 5.0%",
        f"- n_rep = official train/val same-footprint repeats (footprint IoU ≥ 0.9): "
        f"**{o['n_same_footprint_repeats_iou_ge_0.9']}**",
        f"- K1 fires iff p_near < 5% AND n_rep == 0 → **{'FIRES (stop)' if k1['fires'] else 'does NOT fire'}**"
        f"{' (borderline: verdict differs under geodesic)' if k1['borderline'] else ''}", "",
        "## Official split vs IBM split", "",
        "| Quantity | Official val vs train | IBM val vs train | IBM test vs train+val |",
        "|---|---|---|---|"]
    rows = [("held-out chips", "n_held_out", str),
            ("training chips", "n_train", str),
            ("within 0 km (touching/overlap)", ("frac_within_km", "0"), pct),
            ("within 1 km", ("frac_within_km", "1"), pct),
            ("within 2.5 km", ("frac_within_km", "2.5"), pct),
            ("within 5 km", ("frac_within_km", "5"), pct),
            ("within 10 km", ("frac_within_km", "10"), pct),
            ("median NN distance (km)", ("nn_dist_km_quantiles", "0.5"), lambda v: f"{v:.2f}"),
            ("min NN distance (km)", ("nn_dist_km_quantiles", "0"), lambda v: f"{v:.2f}"),
            ("held-out chips overlapping ≥1 train chip", "n_overlapping_train", str),
            ("held-out chips ≥50% covered by train footprints", "n_area_covered_ge_50pct", str),
            ("mean held-out area covered by train", "mean_area_covered_by_train", pct),
            ("overlapping cross pairs", "n_cross_pairs_overlapping", str),
            ("same-footprint repeats (IoU≥0.9)", "n_same_footprint_repeats_iou_ge_0.9", str),
            ("  of which different date", "n_same_footprint_repeats_diff_date", str),
            ("MGRS tiles shared / total", None, None)]
    cols = [o, ib["val_vs_train"], ib["test_vs_trainval"]]
    for label, key, fmt in rows:
        if key is None:
            vals = [f"{c['tiles_shared']}/{c['tiles_total']}" for c in cols]
        elif isinstance(key, tuple):
            vals = [fmt(c[key[0]][key[1]]) for c in cols]
        else:
            vals = [fmt(c[key]) for c in cols]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += ["", "## Whole-pool structure", "",
              f"- chips: {s['n_chips']}; MGRS tiles: {s['n_tiles']}; UTM zones: {s['utm_zones']}",
              f"- all overlapping pairs (any split): {s['pool']['n_pairs_overlapping']}; "
              f"IoU ≥ 0.9: {s['pool']['n_pairs_iou_ge_0.9']} "
              f"(different date: {s['pool']['n_pairs_iou_ge_0.9_diff_date']}, same date: "
              f"{s['pool']['n_pairs_iou_ge_0.9_same_date']})",
              f"- same tile, different date pairs: {s['pool']['n_same_tile_diff_date_pairs']}",
              f"- 5 km single-linkage groups: {s['groups_5km']['n_groups']} "
              f"(singletons {s['groups_5km']['n_singletons']}, largest {s['groups_5km']['largest']} chips)",
              f"- TACO claim recount (official split shared MGRS tiles): "
              f"{o['tiles_shared']}/{o['tiles_total']} (claim: 124/365)",
              f"- IBM test chips touching IBM train+val: {ib['test_vs_trainval']['n_overlapping_train']}; "
              f"min IBM test→train+val distance {ib['test_vs_trainval']['nn_dist_km_quantiles']['0']:.2f} km",
              "", "Figures: `docs/figures/audit_nn_distance.pdf`."]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = C.base_parser(__doc__.splitlines()[0])
    C.require(ap.add_argument("--subdir"))
    C.require(ap.add_argument("--out"))
    C.require(ap.add_argument("--k1_km", type=float))
    C.require(ap.add_argument("--k1_frac", type=float))
    C.require(ap.add_argument("--repeat_iou", type=float))
    C.require(ap.add_argument("--group_km", type=float))
    args, name, cfg = C.parse(ap, argv)
    check_tier(TIER, args.smoke)
    root = data_dir() / args.subdir
    rec = RunRecorder("audit_footprints", name, cfg, seed=None, smoke=args.smoke)

    idx = build_index(root)
    if args.smoke:
        idx = idx.groupby("official_split").head(15).reset_index(drop=True)
    ids = idx.chip_id.tolist()
    polys, crs, ll = read_footprints(root, idx)
    D = pairwise_distance(polys, polys)
    np.fill_diagonal(D, np.inf)
    INTER, IOU = pairwise_overlap(polys, polys, np.where(np.isinf(D), 1, D))

    # pairs with any overlap (i<j)
    ii, jj = np.nonzero(np.triu(D == 0, 1))
    date = dict(zip(idx.chip_id, idx.date))
    pairs = pd.DataFrame({"a": [ids[i] for i in ii], "b": [ids[j] for j in jj],
                          "iou": IOU[ii, jj], "inter_km2": INTER[ii, jj] / 1e6,
                          "date_a": [date[ids[i]] for i in ii], "date_b": [date[ids[j]] for j in jj]})
    pairs["same_tile"] = [ids[i].split(".")[0] == ids[j].split(".")[0] for i, j in zip(ii, jj)]

    off_train = set(idx[idx.official_split == "train"].chip_id)
    off_val = set(idx[idx.official_split == "val"].chip_id)
    ibm_tr, ibm_va = set(ibm_ids("train")), set(ibm_ids("val"))
    ibm_te = set(ibm_test_ids("audit_footprints", "footprint geometry audit, no pixels/labels"))
    if not args.smoke:
        unknown = (ibm_tr | ibm_va | ibm_te) - set(ids)
        if unknown or len(ibm_tr | ibm_va | ibm_te) != len(ids):
            raise SystemExit(f"IBM ids do not map 1:1 onto chips: {sorted(unknown)[:5]}")
    else:
        ibm_tr, ibm_va, ibm_te = (s & set(ids) for s in (ibm_tr, ibm_va, ibm_te))

    per = idx[["chip_id", "tile", "utm_zone", "date", "official_split"]].copy()
    per["crs"] = crs
    per["lon"], per["lat"] = ll[:, 0], ll[:, 1]
    per["ibm_split"] = per.chip_id.map(lambda c: "train" if c in ibm_tr else "val" if c in ibm_va
                                       else "test" if c in ibm_te else "")
    per["pool_nn_dist_km"] = D.min(1) / 1000
    per["pool_max_iou"] = IOU.max(1)
    groups = single_linkage_groups(np.where(np.isinf(D), np.inf, D), args.group_km * 1000)
    per["group_5km"] = groups

    off = held_out_stats("official", polys, D, IOU, ids, off_val, off_train)
    iv = held_out_stats("ibm_val", polys, D, IOU, ids, ibm_va, ibm_tr) if ibm_va else None
    it = held_out_stats("ibm_test", polys, D, IOU, ids, ibm_te, ibm_tr | ibm_va) if ibm_te else None
    for t in (off, iv, it):
        if t is not None:
            per = per.merge(t, on="chip_id", how="left")

    s_off = split_summary("official", off, idx, off_val, off_train, pairs, args.repeat_iou)
    s_iv = split_summary("ibm_val", iv, idx, ibm_va, ibm_tr, pairs, args.repeat_iou)
    s_it = split_summary("ibm_test", it, idx, ibm_te, ibm_tr | ibm_va, pairs, args.repeat_iou)
    p_near = s_off["frac_within_km"][f"{args.k1_km:g}"]
    p_geo = s_off["frac_within_km_geodesic"][f"{args.k1_km:g}"]
    n_rep = s_off["n_same_footprint_repeats_iou_ge_0.9"]
    fires = p_near < args.k1_frac and n_rep == 0
    fires_geo = p_geo < args.k1_frac and n_rep == 0
    gsz = np.bincount(groups)
    summary = {
        "n_chips": len(ids), "n_tiles": int(idx.tile.nunique()),
        "utm_zones": sorted(set(idx.utm_zone.tolist())),
        "K1": {"p_near": p_near, "p_near_geodesic": p_geo, "n_rep": n_rep, "fires": bool(fires),
               "borderline": bool(fires != fires_geo), "km": args.k1_km, "frac": args.k1_frac},
        "official": s_off,
        "ibm": {"val_vs_train": s_iv, "test_vs_trainval": s_it,
                "n_train": len(ibm_tr), "n_val": len(ibm_va), "n_test": len(ibm_te)},
        "pool": {"n_pairs_overlapping": int(len(pairs)),
                 "n_pairs_iou_ge_0.9": int((pairs.iou >= args.repeat_iou).sum()),
                 "n_pairs_iou_ge_0.9_diff_date": int(((pairs.iou >= args.repeat_iou)
                                                      & (pairs.date_a != pairs.date_b)).sum()),
                 "n_pairs_iou_ge_0.9_same_date": int(((pairs.iou >= args.repeat_iou)
                                                      & (pairs.date_a == pairs.date_b)).sum()),
                 "n_same_tile_diff_date_pairs": _n_pairs(idx.groupby("tile").size())
                 - _n_pairs(idx.groupby(["tile", "date"]).size())},
        "groups_5km": {"n_groups": int(len(gsz)), "n_singletons": int((gsz == 1).sum()),
                       "largest": int(gsz.max()), "size_quantiles":
                       {str(q): float(np.quantile(gsz, q)) for q in (.5, .9, .99, 1)}},
        "distance_crs": DIST_CRS,
        "all_chips_in_conus_bbox": bool(((ll[:, 0] > -125) & (ll[:, 0] < -66)
                                         & (ll[:, 1] > 24) & (ll[:, 1] < 50)).all()),
    }

    out = REPO_ROOT / args.out if not args.smoke else REPO_ROOT / "outputs" / "smoke" / name
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "chips_audit.csv", index=False, float_format="%.6g")
    pairs.to_csv(out / "pairs_overlapping.csv", index=False, float_format="%.6g")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figdir = REPO_ROOT / "docs" / "figures" if not args.smoke else out
    figdir.mkdir(parents=True, exist_ok=True)
    series = [off.official_nn_dist_km.to_numpy()]
    labels = ["official val→train"]
    if iv is not None and len(iv):
        series.append(iv.ibm_val_nn_dist_km.to_numpy())
        labels.append("IBM val→train")
    if it is not None and len(it):
        series.append(it.ibm_test_nn_dist_km.to_numpy())
        labels.append("IBM test→train+val")
    hist_pdf(figdir / "audit_nn_distance.pdf", series, labels, "Nearest training chip")
    rec.data["dataset_hash"] = dataset_hash(root, idx)
    rec.data["split_hash"] = sha256_json({"official_train": sorted(off_train), "ibm_train": sorted(ibm_tr),
                                          "ibm_val": sorted(ibm_va), "ibm_test_n": len(ibm_te)})
    mpath = rec.finish(out, summary)
    if not args.smoke:
        write_report(REPO_ROOT / "docs" / "AUDIT_REPORT.md", summary, name,
                     mpath.relative_to(REPO_ROOT).as_posix())
    print(json.dumps(summary["K1"], indent=1))


if __name__ == "__main__":
    main()
