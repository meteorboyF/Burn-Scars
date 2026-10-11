import numpy as np
import pandas as pd
import pytest

from burnscars.eval import stats


def _cm(tp, fp, fn, tn=100):
    return np.array([[tn, fp], [fn, tp]])


def _design(n_groups=60, per=3, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"c{i}" for i in range(n_groups * per)]
    group_of = {c: i // per for i, c in enumerate(ids)}
    return ids, group_of, rng


def test_iou_from_cm_by_hand():
    assert stats.iou_burned_from_cm(_cm(2, 1, 1).ravel()) == pytest.approx(0.5)


def test_bootstrap_weights_rows_sum_to_n():
    W = stats.bootstrap_weights(7, 50, 0)
    assert W.shape == (50, 7) and (W.sum(1) == 7).all()


def test_pooled_point_equals_summed_confusion():
    ids, group_of, rng = _design()
    arm = {c: _cm(*rng.integers(1, 50, 3)) for c in ids}
    res = stats.paired_gap({"a": arm}, ids, group_of, B=200)
    total = sum(m.ravel() for m in arm.values())
    assert res["arms"]["a"]["pooled_iou_burned"] == pytest.approx(stats.iou_burned_from_cm(total))


def test_gap_detected_and_null_gap_ci_covers_zero():
    ids, group_of, rng = _design()
    base = {c: _cm(40, 10, 10) for c in ids}
    better = {c: _cm(45, 7, 5) for c in ids}            # clearly higher IoU
    noisy = {c: _cm(*(np.array([40, 10, 10]) + rng.integers(-3, 4, 3))) for c in ids}
    res = stats.paired_gap({"rand": better, "grp": base, "nbr_r": noisy, "nbr_g": noisy}, ids, group_of,
                           B=2000, seed=1)
    d = stats.h1_decision(res, "rand", "grp", "nbr_r", "nbr_g")
    assert d["G"]["point_pts"] > 2 and d["G"]["ci_excludes_0"] and d["supported"]
    null = stats.contrast(res, "nbr_r", "nbr_g")
    assert null["point_pts"] == 0 and null["ci_pts"] == [0.0, 0.0]


def test_unpaired_design_refused():
    ids, group_of, _ = _design()
    with pytest.raises(ValueError, match="paired"):
        stats.paired_gap({"a": {ids[0]: _cm(1, 1, 1)}}, ids, group_of, B=10)


def test_rank_reversal():
    ids, group_of, _ = _design()
    arms = {"u_r": {c: _cm(50, 5, 5) for c in ids}, "p_r": {c: _cm(40, 10, 10) for c in ids},
            "u_g": {c: _cm(30, 15, 15) for c in ids}, "p_g": {c: _cm(40, 10, 10) for c in ids}}
    res = stats.paired_gap(arms, ids, group_of, B=500)
    rr = stats.rank_reversal(res, "u_r", "p_r", "u_g", "p_g")
    assert rr["point_reversed"] and rr["p_reversal"] == 1.0 and rr["counts"]


def test_kendall_two_models():
    k = stats.kendall_tau_by_fold({"u": [1, 1], "p": [0, 0]}, {"u": [0, 1], "p": [1, 0]})
    assert k["tau_per_fold"] == [-1.0, 1.0]


def test_regressions_recover_known_slope():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"chip_id": range(n), "group": rng.integers(0, 150, n),
                       "dist_km": rng.exponential(20, n), "burned_frac": rng.uniform(0, .5, n),
                       "year": rng.choice([2018, 2019, 2020, 2021], n), "iou_nbr": rng.uniform(.2, .6, n)})
    df["iou_learned"] = (0.8 - 0.05 * np.log1p(df.dist_km) + 0.2 * df.burned_frac + 0.3 * df.iou_nbr
                         + rng.normal(0, .02, n))
    out = stats.distance_regressions(df, B=300)
    assert set(out) == {"R0", "R1", "R2"}
    assert out["R1"]["dist_coef"] == pytest.approx(-0.05, abs=0.01)
    assert out["R1"]["dist_negative_and_excludes_0"]
    assert out["R1"]["coef"]["iou_nbr"] == pytest.approx(0.3, abs=0.03)
