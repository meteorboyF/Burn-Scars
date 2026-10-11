"""Pin the released split files (hash) and check their structure without the raster data."""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from burnscars.data.footprints import pairwise_distance, reproject_footprint
from burnscars.paths import REPO_ROOT, data_dir

V1 = REPO_ROOT / "splits" / "v1"
PINNED = {  # changing a released split is a new version, never an edit
    "grouped_cv5_all804.json": "ec7df7dadca093ae64183348109bc2c2b7e7b82f9fc5b2960430185b38aec878",
    "grouped_cv5_ibm_trainval.json": "57b75d6b1a0a8af8f9e94106fbeea1696418c5ad7a9af5c6711a33305e1550dc",
    "ibm_trainval.json": "0e675124b9d81bb51f67be30f7e2562f901e9c8b9521b230a72a2b15caafa346",
    "official.json": "0e08ca16e4f4cf126c3270481c640d5f95d7b898834183712233b140e3982695",
    "random_cv5_all804.json": "0dd2d0f907090703140a8f1715171754138a65bd393d20737e7dfd7054538373",
    "random_cv5_ibm_trainval.json": "b193ac1e372dac2345defbad602444a1bf00a0756e616499f59ed948ec7906d1",
}
CV = [n for n in PINNED if "_cv5_" in n]


def _load(name):
    return json.loads((V1 / name).read_text())


@pytest.mark.parametrize("name", sorted(PINNED))
def test_hash_pinned(name):
    assert hashlib.sha256((V1 / name).read_bytes()).hexdigest() == PINNED[name]


def test_sha256sums_file_matches():
    listed = dict(reversed(l.split("  ")) for l in (V1 / "SHA256SUMS").read_text().splitlines())
    assert listed == PINNED


def test_official_split_counts():
    f = _load("official.json")["folds"][0]
    assert (len(f["train"]), len(f["test"])) == (540, 264) and not set(f["train"]) & set(f["test"])


@pytest.mark.parametrize("name", CV)
def test_cv_structure(name):
    s = _load(name)
    pool = set().union(*(f["test"] for f in s["folds"]))
    assert len(pool) == s["n_pool"] == sum(len(f["test"]) for f in s["folds"])  # partition
    for f in s["folds"]:
        tr, te = set(f["train"]), set(f["test"])
        assert not tr & te and tr | te | set(f["train_dropped"]) == pool
        assert len(tr) == s["n_train_per_fold"]
        for inner in f["inner"]:
            assert set(inner["val"]) | set(inner["train"]) == tr and not set(inner["val"]) & set(inner["train"])


@pytest.mark.parametrize("pool", ["all804", "ibm_trainval"])
def test_equal_train_sizes_across_regimes(pool):
    assert _load(f"random_cv5_{pool}.json")["n_train_per_fold"] == _load(f"grouped_cv5_{pool}.json")["n_train_per_fold"]


@pytest.mark.parametrize("pool", ["all804", "ibm_trainval"])
def test_grouped_folds_are_5km_separated(pool):
    s = _load(f"grouped_cv5_{pool}.json")
    prof = pd.read_csv(REPO_ROOT / "results" / "profile" / "chips_profile.csv").set_index("chip_id")
    poly = {c: reproject_footprint((r.left, r.bottom, r.right, r.top), r.crs) for c, r in prof.iterrows()}
    for f in s["folds"]:
        d = pairwise_distance([poly[c] for c in f["test"]], [poly[c] for c in f["train"]])
        assert d.min() > s["group_km"] * 1000, f"fold {f['fold']}: {d.min():.0f} m"
        groups_test = {s["group_of"][c] for c in f["test"]}
        assert not groups_test & {s["group_of"][c] for c in f["train"]}


@pytest.mark.skipif(not (data_dir() / "ibm_splits" / "splits" / "train.txt").exists(),
                    reason="IBM split lists not downloaded")
def test_regenerates_byte_identical(tmp_path):
    r = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "make_splits.py"), "--config",
                        "make_splits_v1", "--out_dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for name, h in PINNED.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == h, name
