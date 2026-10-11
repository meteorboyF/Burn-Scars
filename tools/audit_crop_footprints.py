"""Header-only footprint audit of the multi-temporal crop classification dataset.

    PYTHONPATH=src python tools/audit_crop_footprints.py --config audit_crop_footprints [--smoke]

Streams each pinned .tgz over HTTPS (tar stream mode, never written to disk),
reads every chip's GeoTIFF header (CRS, bounds) from memory, discards pixels,
then computes K1-style statistics for the dataset's official train/val split:
val->train nearest distance, overlap, same-footprint repeats, shared MGRS tiles.
Chips are axis-aligned boxes in one CRS (EPSG:5070), so distances are exact
box arithmetic. Disk use: the output CSV/JSON only.
"""
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.io import MemoryFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars import config as C  # noqa: E402
from burnscars.data.footprints import single_linkage_groups  # noqa: E402
from burnscars.hardware import check_tier  # noqa: E402
from burnscars.manifest import RunRecorder, sha256_bytes, sha256_json  # noqa: E402
from burnscars.paths import REPO_ROOT  # noqa: E402

TIER = "cpu"
BASE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{f}"


class HashingReader:
    """File-like wrapper that SHA256-hashes every byte read through it."""

    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def read(self, size=-1):
        b = self.f.read(size)
        self.h.update(b)
        self.n += len(b)
        return b


def stream_headers(url, max_members=None, log_every=500):
    """Returns (rows, sha256_of_stream or None if stopped early, bytes_read)."""
    rows, n = [], 0
    with urllib.request.urlopen(url) as resp:
        hr = HashingReader(resp)
        rows, complete = _read_tar(hr, url, max_members, log_every)
        if complete:
            while hr.read(1 << 20):  # drain tar padding so the hash covers the whole file
                pass
        return rows, (hr.h.hexdigest() if complete else None), hr.n


def _read_tar(hr, url, max_members, log_every):
    rows, n = [], 0
    with tarfile.open(fileobj=hr, mode="r|gz") as tf:
        for m in tf:
            name = m.name.split("/")[-1]
            if not m.isfile() or name.startswith("._") or not name.endswith(".tif"):
                continue
            data = tf.extractfile(m).read()  # in memory only
            with MemoryFile(data) as mf, mf.open() as ds:
                b = ds.bounds
                rows.append({"member": m.name, "file": name, "crs": ds.crs.to_string() if ds.crs else None,
                             "left": b.left, "bottom": b.bottom, "right": b.right, "top": b.top,
                             "width": ds.width, "height": ds.height, "count": ds.count,
                             "res_x": ds.res[0], "res_y": ds.res[1]})
            n += 1
            if n % log_every == 0:
                print(f"  {n} tifs read from {url.rsplit('/', 1)[-1]}", flush=True)
            if max_members and n >= max_members:
                return rows, False
    return rows, True


def box_distance(B1, B2):
    """Exact min distance between axis-aligned boxes (rows: l, b, r, t)."""
    dx = np.maximum(0, np.maximum(B1[:, None, 0] - B2[None, :, 2], B2[None, :, 0] - B1[:, None, 2]))
    dy = np.maximum(0, np.maximum(B1[:, None, 1] - B2[None, :, 3], B2[None, :, 1] - B1[:, None, 3]))
    return np.hypot(dx, dy)


def box_iou(B1, B2):
    w = np.clip(np.minimum(B1[:, None, 2], B2[None, :, 2]) - np.maximum(B1[:, None, 0], B2[None, :, 0]), 0, None)
    h = np.clip(np.minimum(B1[:, None, 3], B2[None, :, 3]) - np.maximum(B1[:, None, 1], B2[None, :, 1]), 0, None)
    inter = w * h
    a1 = (B1[:, 2] - B1[:, 0]) * (B1[:, 3] - B1[:, 1])
    a2 = (B2[:, 2] - B2[:, 0]) * (B2[:, 3] - B2[:, 1])
    return inter, inter / (a1[:, None] + a2[None, :] - inter)


def main(argv=None):
    ap = C.base_parser(__doc__.splitlines()[0])
    for k in ("repo", "revision", "out"):
        C.require(ap.add_argument(f"--{k}"))
    C.require(ap.add_argument("--archives", type=dict, help="split name -> {file, size, sha256}"))
    C.require(ap.add_argument("--lists", type=dict, help="split name -> {file, sha256} id list"))
    C.require(ap.add_argument("--chips_csv", type=dict, help="{file, sha256}"))
    C.require(ap.add_argument("--k1_km", type=float))
    C.require(ap.add_argument("--repeat_iou", type=float))
    C.require(ap.add_argument("--group_km", type=float))
    args, name, cfg = C.parse(ap, argv)
    check_tier(TIER, args.smoke)
    rec = RunRecorder("audit_crop_footprints", name, cfg, seed=None, smoke=args.smoke)

    def fetch_small(spec):
        raw = urllib.request.urlopen(BASE.format(repo=args.repo, rev=args.revision, f=spec["file"])).read()
        if sha256_bytes(raw) != spec["sha256"]:
            raise SystemExit(f"{spec['file']}: SHA256 mismatch")
        return raw.decode()

    lists = {s: [x.strip() for x in fetch_small(spec).splitlines() if x.strip()] for s, spec in args.lists.items()}
    meta = pd.read_csv(io.StringIO(fetch_small(args.chips_csv)))

    rows, stream_check = [], {}
    for split, spec in args.archives.items():
        url = BASE.format(repo=args.repo, rev=args.revision, f=spec["file"])
        print(f"streaming {url}", flush=True)
        got, digest, nbytes = stream_headers(url, max_members=20 if args.smoke else None)
        if digest is not None and (digest != spec["sha256"] or nbytes != spec["size"]):
            raise SystemExit(f"{spec['file']}: stream SHA256/size mismatch ({digest}, {nbytes})")
        stream_check[spec["file"]] = {"sha256_verified": digest is not None, "bytes_streamed": nbytes}
        for r in got:
            r["archive_split"] = split
            rows.append(r)
    hdr = pd.DataFrame(rows)
    hdr["chip_id"] = hdr.file.str.replace(r"(_merged|\.mask)\.tif$", "", regex=True)
    hdr["kind"] = np.where(hdr.file.str.endswith(".mask.tif"), "mask", "scene")

    # scene and mask headers must agree; one row per chip
    g = hdr.groupby("chip_id")
    agree = g[["left", "bottom", "right", "top", "crs"]].nunique().max(axis=1) == 1
    chips = g.first()[["archive_split", "crs", "left", "bottom", "right", "top", "width", "height", "res_x", "res_y"]]
    chips["n_files"] = g.size()
    chips["scene_mask_agree"] = agree
    chips = chips.reset_index()

    list_split = {c: s for s, ids in lists.items() for c in ids}
    chips["list_split"] = chips.chip_id.map(list_split)
    chips = chips.merge(meta[["chip_id", "tile"]], on="chip_id", how="left")
    out = REPO_ROOT / args.out if not args.smoke else REPO_ROOT / "outputs" / "smoke" / name
    out.mkdir(parents=True, exist_ok=True)

    tr = chips[chips.list_split == "train"]
    va = chips[chips.list_split == "val"]
    cols = ["left", "bottom", "right", "top"]
    Bt, Bv = tr[cols].to_numpy(float), va[cols].to_numpy(float)
    summary = {
        "n_chips_streamed": int(len(chips)), "n_tif_files": int(len(hdr)),
        "crs": chips.crs.value_counts().to_dict(),
        "sizes_px": {f"{w}x{h}": int(n) for (w, h), n in chips.groupby(["width", "height"]).size().items()},
        "chip_side_m": sorted(set((chips.right - chips.left).round(3).tolist()))[:5],
        "scene_mask_disagree": int((~chips.scene_mask_agree).sum()),
        "chips_not_in_lists": int(chips.list_split.isna().sum()),
        "list_sizes": {s: len(v) for s, v in lists.items()},
        "list_ids_missing_from_archives": {s: len(set(v) - set(chips.chip_id)) for s, v in lists.items()},
        "archive_vs_list_mismatch": int((chips.archive_split != chips.list_split).sum()),
        "stream_check": stream_check,
    }
    if len(tr) and len(va):
        if chips.crs.nunique() != 1:
            raise SystemExit(f"multiple CRS {chips.crs.unique()}: box arithmetic invalid")
        D = box_distance(Bv, Bt)
        inter, iou = box_iou(Bv, Bt)
        nn = D.min(1) / 1000
        va = va.assign(nn_dist_km=nn, nn_train=tr.chip_id.to_numpy()[D.argmin(1)],
                       max_iou_train=iou.max(1), n_train_touching=(D == 0).sum(1),
                       n_train_overlap=(inter > 0).sum(1))
        tiles_t, tiles_v = set(tr.tile.dropna()), set(va.tile.dropna())
        th = (0, 1, 2.5, 5, 10, 20)
        Ball = chips[cols].to_numpy(float)
        Dall = box_distance(Ball, Ball)
        np.fill_diagonal(Dall, np.inf)
        groups = single_linkage_groups(Dall, args.group_km * 1000)
        gsz = np.bincount(groups)
        _, iou_all = box_iou(Ball, Ball)
        np.fill_diagonal(iou_all, 0)
        summary |= {
            "n_train": int(len(tr)), "n_val": int(len(va)),
            "frac_val_within_km": {f"{t:g}": float(np.mean(nn <= t)) for t in th},
            "n_val_within_km": {f"{t:g}": int(np.sum(nn <= t)) for t in th},
            "val_nn_dist_km_quantiles": {str(q): float(np.quantile(nn, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
            "n_val_overlapping_train_area": int((va.n_train_overlap > 0).sum()),
            "n_val_touching_train_edge_or_area": int((va.n_train_touching > 0).sum()),
            "n_cross_repeats_iou_ge": int((iou >= args.repeat_iou).sum()),
            "n_pool_pairs_overlapping_area": int((np.triu(iou_all, 1) > 0).sum()),
            "tiles_train": len(tiles_t), "tiles_val": len(tiles_v),
            "tiles_shared": len(tiles_t & tiles_v), "tiles_total": len(tiles_t | tiles_v),
            "n_val_chips_on_shared_tiles": int(va.tile.isin(tiles_t).sum()),
            "groups_5km": {"n_groups": int(len(gsz)), "n_singletons": int((gsz == 1).sum()),
                           "largest": int(gsz.max())},
            "K1_style": {"p_near": float(np.mean(nn <= args.k1_km)),
                         "n_rep": int((iou >= args.repeat_iou).sum()),
                         "would_fire": bool(np.mean(nn <= args.k1_km) < 0.05 and (iou >= args.repeat_iou).sum() == 0)},
        }
        chips = chips.merge(va[["chip_id", "nn_dist_km", "nn_train", "max_iou_train",
                                "n_train_touching", "n_train_overlap"]], on="chip_id", how="left")
        chips["group_5km"] = groups
    chips.to_csv(out / "chips_crop_audit.csv", index=False, float_format="%.6g")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    rec.data["dataset_hash"] = sha256_json(sorted(zip(chips.chip_id, chips.left, chips.bottom)))
    rec.data["split_hash"] = sha256_json({s: sorted(v) for s, v in lists.items()})
    rec.finish(out, summary)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
