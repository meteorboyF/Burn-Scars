"""Profile HLS Burn Scars: shapes, dtypes, bands, value ranges, nodata, masks, parse checks.

    PYTHONPATH=src python tools/profile_data.py --config profile_hls_burn_scars [--smoke]

Writes <out>/chips_profile.csv (one row per chip), <out>/summary.json and a manifest.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars import config as C  # noqa: E402
from burnscars.data.chips import build_index, dataset_hash  # noqa: E402
from burnscars.hardware import check_tier  # noqa: E402
from burnscars.manifest import RunRecorder  # noqa: E402
from burnscars.paths import REPO_ROOT, data_dir  # noqa: E402

TIER = "cpu"
BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]


def profile_chip(root: Path, row) -> dict:
    out = {"chip_id": row.chip_id}
    with rasterio.open(root / row.scene_path) as s:
        x = s.read()
        out |= {"scene_count": s.count, "scene_dtype": s.dtypes[0], "height": s.height,
                "width": s.width, "crs": s.crs.to_string() if s.crs else None,
                "res_x": s.res[0], "res_y": s.res[1], "scene_nodata_hdr": s.nodata,
                "left": s.bounds.left, "bottom": s.bounds.bottom, "right": s.bounds.right,
                "top": s.bounds.top, "band_desc": "|".join(str(d) for d in s.descriptions)}
        scene_crs, scene_tf = s.crs, s.transform
    with rasterio.open(root / row.mask_path) as m:
        y = m.read(1)
        out |= {"mask_count": m.count, "mask_dtype": m.dtypes[0], "mask_nodata_hdr": m.nodata,
                "mask_grid_matches": bool(m.crs == scene_crs and m.transform == scene_tf
                                          and m.shape == x.shape[1:])}
    vals, cnts = np.unique(y, return_counts=True)
    vc = dict(zip(vals.tolist(), cnts.tolist()))
    n = y.size
    n_b, n_u, n_i = vc.get(1, 0), vc.get(0, 0), vc.get(-1, 0)
    out |= {"mask_values": json.dumps({str(k): v for k, v in vc.items()}),
            "mask_unexpected": int(n - n_b - n_u - n_i),
            "n_burned": n_b, "n_unburned": n_u, "n_ignore": n_i,
            "burned_frac": n_b / (n_b + n_u) if n_b + n_u else np.nan,
            "ignore_frac": n_i / n}
    xf = x.astype(np.float64)
    for b, name in enumerate(BANDS):
        v = xf[b]
        fin = np.isfinite(v)
        out |= {f"{name}_min": float(v[fin].min()) if fin.any() else np.nan,
                f"{name}_max": float(v[fin].max()) if fin.any() else np.nan,
                f"{name}_mean": float(v[fin].mean()) if fin.any() else np.nan,
                f"{name}_n_nonfinite": int((~fin).sum()),
                f"{name}_n_neg": int((v < 0).sum())}
    # what do scene pixels look like where the mask says no-data?
    ign = y == -1
    allzero = (x == 0).all(axis=0)
    out |= {"n_scene_allzero_px": int(allzero.sum()),
            "n_ignore_and_allzero": int((ign & allzero).sum()),
            "n_scene_any_neg_px": int((x < 0).any(axis=0).sum()),
            "n_ignore_and_any_neg": int((ign & (x < 0).any(axis=0)).sum())}
    if out["scene_nodata_hdr"] is not None:
        nd = (x == out["scene_nodata_hdr"]).any(axis=0)
        out |= {"n_scene_eq_nodata_hdr": int(nd.sum()), "n_ignore_and_nodata_hdr": int((ign & nd).sum())}
    return out


def summarise(idx: pd.DataFrame, prof: pd.DataFrame) -> dict:
    df = idx.merge(prof, on="chip_id")
    s = {
        "n_chips": len(df),
        "n_by_official_split": df.official_split.value_counts().to_dict(),
        "n_by_year": {str(k): int(v) for k, v in df.year.value_counts().sort_index().items()},
        "n_by_utm_zone": {str(k): int(v) for k, v in df.utm_zone.value_counts().sort_index().items()},
        "n_by_sensor": df.sensor.value_counts().to_dict(),
        "n_by_crs": df.crs.value_counts().to_dict(),
        "n_tiles": int(df.tile.nunique()),
        "shapes": Counter(f"{c}x{h}x{w}" for c, h, w in zip(df.scene_count, df.height, df.width)),
        "scene_dtypes": df.scene_dtype.value_counts().to_dict(),
        "mask_dtypes": df.mask_dtype.value_counts().to_dict(),
        "res": Counter(f"{a}x{b}" for a, b in zip(df.res_x, df.res_y)),
        "scene_nodata_hdr": {str(k): int(v) for k, v in df.scene_nodata_hdr.value_counts(dropna=False).items()},
        "mask_nodata_hdr": {str(k): int(v) for k, v in df.mask_nodata_hdr.value_counts(dropna=False).items()},
        "mask_grid_mismatch_chips": int((~df.mask_grid_matches).sum()),
        "mask_value_totals": dict(sum((Counter(json.loads(v)) for v in df.mask_values), Counter())),
        "chips_with_unexpected_mask_values": int((df.mask_unexpected > 0).sum()),
        "chips_zero_burned": int((df.n_burned == 0).sum()),
        "chips_all_ignore": int((df.n_burned + df.n_unburned == 0).sum()),
        "chips_any_ignore": int((df.n_ignore > 0).sum()),
        "pixel_frac_burned_of_all": float(df.n_burned.sum() / (df.height * df.width).sum()),
        "pixel_frac_ignore_of_all": float(df.n_ignore.sum() / (df.height * df.width).sum()),
        "burned_frac_quantiles": {q: float(df.burned_frac.quantile(q)) for q in (0, .1, .25, .5, .75, .9, 1)},
        "ignore_frac_quantiles": {q: float(df.ignore_frac.quantile(q)) for q in (0, .5, .9, .99, 1)},
        "band_ranges": {b: {"min": float(df[f"{b}_min"].min()), "max": float(df[f"{b}_max"].max()),
                            "mean_of_means": float(df[f"{b}_mean"].mean()),
                            "chips_with_nonfinite": int((df[f"{b}_n_nonfinite"] > 0).sum()),
                            "chips_with_negative": int((df[f"{b}_n_neg"] > 0).sum())} for b in BANDS},
        "px_scene_allzero": int(df.n_scene_allzero_px.sum()),
        "px_ignore_and_allzero": int(df.n_ignore_and_allzero.sum()),
        "px_scene_any_negative": int(df.n_scene_any_neg_px.sum()),
        "px_ignore_and_any_negative": int(df.n_ignore_and_any_neg.sum()),
        "band_descriptions": df.band_desc.value_counts().to_dict(),
        "date_range": [df.date.min(), df.date.max()],
        "chip_id_unique": bool(df.chip_id.is_unique),
        "tile_date_repeats": int(df.duplicated(["tile", "date"]).sum()),
    }
    for c in ("n_scene_eq_nodata_hdr", "n_ignore_and_nodata_hdr"):
        if c in df:
            s["px_" + c[2:]] = int(df[c].fillna(0).sum())
    return json.loads(json.dumps(s, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))


def main(argv=None):
    ap = C.base_parser(__doc__.splitlines()[0])
    C.require(ap.add_argument("--subdir"))
    C.require(ap.add_argument("--out"))
    args, name, cfg = C.parse(ap, argv)
    check_tier(TIER, args.smoke)
    root = data_dir() / args.subdir
    rec = RunRecorder("profile_data", name, cfg, seed=None, smoke=args.smoke)
    idx = build_index(root)
    if args.smoke:
        idx = idx.iloc[:5]
    prof = pd.DataFrame([profile_chip(root, r) for r in idx.itertuples()])
    out = REPO_ROOT / args.out if not args.smoke else REPO_ROOT / "outputs" / "smoke" / name
    out.mkdir(parents=True, exist_ok=True)
    table = idx.merge(prof, on="chip_id")
    table.to_csv(out / "chips_profile.csv", index=False, float_format="%.6g")
    summary = summarise(idx, prof)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.smoke:
        print(table.iloc[0].to_dict())
    rec.data["dataset_hash"] = dataset_hash(root, idx)
    rec.finish(out, summary)
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
