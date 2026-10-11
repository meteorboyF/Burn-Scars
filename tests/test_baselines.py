import numpy as np
import pytest

from burnscars.eval.baselines import EDGES, NBRHistogram, nbr, predict_nbr
from burnscars.eval.metrics import aggregate, chip_scores


def _bands(nir, swir2):
    b = np.zeros((6,) + np.shape(nir))
    b[3], b[5] = nir, swir2
    return b


def test_nbr_values_and_undefined():
    out = nbr(_bands(np.array([[0.3, 0.0, 0.2]]), np.array([[0.1, 0.0, 0.2]])))
    assert out[0, 0] == pytest.approx(0.5)
    assert np.isnan(out[0, 1])          # 0/0
    assert out[0, 2] == 0.0
    assert predict_nbr(out, 0.1).tolist() == [[0, 0, 1]]  # NaN -> unburned


def test_histogram_curve_matches_bruteforce():
    rng = np.random.default_rng(0)
    chips = []
    for _ in range(4):
        idx = rng.uniform(-1.2, 1.2, (16, 16))
        idx[0, :3] = np.nan
        idx[1, :2] = [EDGES[100], EDGES[250]]  # values exactly on edges
        tgt = (idx < rng.uniform(-0.3, 0.3)).astype(int)
        tgt[2, :4] = -1
        tgt[0, 0] = 1                            # burned with undefined NBR
        chips.append((idx, tgt))
    h = NBRHistogram()
    for idx, tgt in chips:
        h.add(idx, tgt)
    curve = h.iou_burned_curve()
    for j in (0, 37, 100, 200, 250, 333, 400):
        agg = aggregate(chip_scores(t, predict_nbr(i, EDGES[j])) for i, t in chips)
        assert curve[j] == pytest.approx(agg["dataset_iou_burned"]), j
    t_best, iou_best = h.best_threshold()
    assert iou_best == pytest.approx(np.nanmax(curve))
