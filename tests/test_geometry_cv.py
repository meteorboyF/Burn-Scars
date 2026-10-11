import numpy as np
import pytest
from shapely.geometry import box

from burnscars.data import cv
from burnscars.data.footprints import (geodesic_nearest, pairwise_distance, pairwise_overlap,
                                       reproject_footprint, single_linkage_groups)


def test_distance_overlap_iou_by_hand():
    a = [box(0, 0, 10, 10)]
    b = [box(5, 0, 15, 10), box(13, 0, 20, 10), box(0, 0, 10, 10)]
    d = pairwise_distance(a, b)
    assert d.tolist() == [[0.0, 3.0, 0.0]]
    inter, iou = pairwise_overlap(a, b, d)
    assert inter[0].tolist() == [50.0, 0.0, 100.0]
    assert iou[0, 0] == pytest.approx(50 / 150) and iou[0, 2] == 1.0


def test_single_linkage_chains():
    polys = [box(0, 0, 1, 1), box(4, 0, 5, 1), box(8, 0, 9, 1), box(30, 0, 31, 1)]
    d = pairwise_distance(polys, polys)
    assert single_linkage_groups(d, 3.0).tolist() == [0, 0, 0, 1]   # chain 0-1-2 at gap 3
    assert single_linkage_groups(d, 2.9).tolist() == [0, 1, 2, 3]


def test_reprojection_utm_chip_size_and_geodesic():
    # 512 px * 30 m chip in UTM 10N near 38N; in EPSG:5070 area ~ (15.36 km)^2 within 2%
    p = reproject_footprint((500000, 4200000, 515360, 4215360), "EPSG:32610")
    assert p.area == pytest.approx(15360 ** 2, rel=0.02)
    q = reproject_footprint((520360, 4200000, 535720, 4215360), "EPSG:32610")  # 5 km east
    planar = p.distance(q)
    assert planar == pytest.approx(5000, rel=0.02)
    assert geodesic_nearest(p, q) == pytest.approx(5000, rel=0.01)


def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"c{i:03d}" for i in range(n)]
    groups = np.repeat(np.arange(n // 3), 3)
    return ids, groups, rng.uniform(0, 0.4, n)


def test_random_folds_deterministic_and_balanced():
    ids, _, bf = _toy()
    a = cv.stratified_random_folds(ids, bf, 5, seed=1)
    assert a == cv.stratified_random_folds(ids, bf, 5, seed=1)
    assert a != cv.stratified_random_folds(ids, bf, 5, seed=2)
    assert sorted(len(m) for m in cv.fold_members(a, 5)) == [12] * 5


def test_grouped_folds_keep_groups_together():
    ids, groups, bf = _toy()
    a = cv.grouped_folds(ids, groups, bf, 5, seed=0)
    assert a == cv.grouped_folds(ids, groups, bf, 5, seed=0)
    for g in np.unique(groups):
        assert len({a[i] for i, gg in zip(ids, groups) if gg == g}) == 1
    sizes = [len(m) for m in cv.fold_members(a, 5)]
    assert max(sizes) - min(sizes) <= 3


def test_grouped_refuses_giant_group():
    ids = [str(i) for i in range(10)]
    with pytest.raises(ValueError, match="exceeds"):
        cv.grouped_folds(ids, [0] * 5 + [1, 2, 3, 4, 5], [0.1] * 10, 5, seed=0)


def test_subsample():
    kept, dropped = cv.subsample([f"x{i}" for i in range(10)], 7, seed=0)
    assert len(kept) == 7 and len(dropped) == 3 and not set(kept) & set(dropped)
    assert kept == cv.subsample([f"x{i}" for i in range(10)], 7, seed=0)[0]
