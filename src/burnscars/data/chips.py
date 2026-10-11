"""Chip index: filenames -> (chip_id, sensor, MGRS tile, UTM zone, date, official split).

Observed naming (verified 2026-10-11 on all 1608 files, docs/DATA.md):
    training/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif   (6-band scene)
    training/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4.mask.tif     (1-band mask)
    training/training_index.csv, validation/validation_index.csv  (scene,mask pairs)
chip_id = "<tile>.<yyyyddd>.<v>" e.g. "T10SEH.2018190.v1", the form used by IBM's
split lists. Parsing is strict: anything that does not match raises.
"""
import datetime as dt
import re
from pathlib import Path

import pandas as pd

from ..manifest import sha256_file, sha256_json

from ..paths import data_dir

NAME_RE = re.compile(
    r"^(?:subsetted_512x512_)?HLS\.(?P<sensor>[SL]30)\.(?P<tile>T(?P<zone>\d{2})(?P<band>[C-X])(?P<sq>[A-Z]{2}))"
    r"\.(?P<year>\d{4})(?P<doy>\d{3})\.(?P<ver>v\d+)\.(?P<minor>\d+)"
    r"(?P<kind>_merged\.tif|\.mask\.tif)$")
OFFICIAL_DIRS = {"training": "train", "validation": "val"}


def parse_name(name: str) -> dict:
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"unparseable chip filename: {name}")
    g = m.groupdict()
    year, doy = int(g["year"]), int(g["doy"])
    if not 1 <= doy <= (366 if year % 4 == 0 else 365):
        raise ValueError(f"bad day-of-year in {name}")
    date = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)
    return {
        "chip_id": f"{g['tile']}.{g['year']}{g['doy']}.{g['ver']}",
        "sensor": g["sensor"], "tile": g["tile"], "utm_zone": int(g["zone"]),
        "lat_band": g["band"], "year": year, "doy": doy, "date": date.isoformat(),
        "hls_version": f"{g['ver']}.{g['minor']}",
        "kind": "scene" if g["kind"] == "_merged.tif" else "mask",
    }


def build_index(root: Path | None = None) -> pd.DataFrame:
    """One row per chip with scene + mask paths and the official split. Strict pairing."""
    root = Path(root or data_dir() / "hls_burn_scars")
    rows = {}
    for sdir, split in OFFICIAL_DIRS.items():
        d = root / sdir
        if not d.is_dir():
            raise FileNotFoundError(d)
        for p in sorted(d.iterdir()):
            if not p.name.endswith(".tif"):
                continue
            info = parse_name(p.name)
            key = info["chip_id"]
            r = rows.setdefault(key, {k: v for k, v in info.items() if k != "kind"}
                                | {"official_split": split})
            if r["official_split"] != split:
                raise ValueError(f"{key} appears in both official splits")
            col = f"{info['kind']}_path"
            if col in r:
                raise ValueError(f"duplicate {info['kind']} for {key}")
            r[col] = str(p.relative_to(root))
        listed = pd.read_csv(d / f"{sdir}_index.csv", header=None, names=["scene", "mask"])
        on_disk = {(r["scene_path"].split("/")[-1], r["mask_path"].split("/")[-1])
                   for r in rows.values() if r["official_split"] == split}
        if set(map(tuple, listed.to_numpy())) != on_disk:
            raise ValueError(f"{sdir}_index.csv disagrees with the files on disk")
    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["chip_id"]))
    missing = df[df[["scene_path", "mask_path"]].isna().any(axis=1)]
    if len(missing):
        raise ValueError(f"unpaired chips: {missing.chip_id.tolist()[:5]}")
    return df


def dataset_hash(root: Path, idx: pd.DataFrame) -> str:
    """Hash over (chip_id, scene sha256, mask sha256) for the chips in `idx`."""
    root = Path(root)
    return sha256_json(sorted((r.chip_id, sha256_file(root / r.scene_path), sha256_file(root / r.mask_path))
                              for r in idx.itertuples()))
