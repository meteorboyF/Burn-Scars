"""Hand-computed toy masks. -1 is always ignored, never a negative label."""
import math

import numpy as np
import pytest

from burnscars.eval.metrics import aggregate, chip_scores, confusion, iou_from_confusion

# chip A (2x3): valid pixels = 5 (one ignored)
#   target  1  1  0        pred  1  0  0
#           0 -1  1              1  1  1
# burned:   TP=2 (0,0),(1,2); FN=1 (0,1); FP=1 (1,0)       -> IoU_1 = 2/4 = 0.5
# unburned: TP=1 (0,2); FN=1 (1,0); FP=1 (0,1)              -> IoU_0 = 1/3
# pred at the ignored pixel (1,1) is 1 and must not count.
TA = np.array([[1, 1, 0], [0, -1, 1]])
PA = np.array([[1, 0, 0], [1, 1, 1]])


def test_chip_a_by_hand():
    s = chip_scores(TA, PA)
    assert s["iou_burned"] == pytest.approx(0.5)
    assert s["iou_unburned"] == pytest.approx(1 / 3)
    assert s["miou"] == pytest.approx((0.5 + 1 / 3) / 2)
    assert s["n_valid"] == 5
    assert confusion(TA, PA).tolist() == [[1, 1], [1, 2]]


def test_ignore_pixel_prediction_is_irrelevant():
    p2 = PA.copy()
    p2[1, 1] = 0
    assert chip_scores(TA, p2)["cm"].tolist() == chip_scores(TA, PA)["cm"].tolist()


def test_minus_one_is_not_a_label():
    with pytest.raises(ValueError):
        chip_scores(np.array([[0, 2]]), np.array([[0, 1]]))
    with pytest.raises(ValueError):  # predicting -1 at a valid pixel is an error
        chip_scores(np.array([[0, 1]]), np.array([[0, -1]]))


def test_all_ignore_chip():
    s = chip_scores(np.full((2, 2), -1), np.ones((2, 2), int))
    assert s["n_valid"] == 0
    assert math.isnan(s["iou_burned"]) and math.isnan(s["iou_unburned"]) and math.isnan(s["miou"])


def test_empty_burn_chip_correct_and_with_false_positive():
    t = np.zeros((2, 2), int)
    ok = chip_scores(t, np.zeros((2, 2), int))
    assert math.isnan(ok["iou_burned"])          # empty union -> undefined, not 0 or 1
    assert ok["iou_unburned"] == 1.0
    assert ok["miou"] == 1.0                       # mean over defined classes only
    fp = chip_scores(t, np.array([[1, 0], [0, 0]]))
    assert fp["iou_burned"] == 0.0                 # union = 1 FP pixel, TP = 0
    assert fp["iou_unburned"] == pytest.approx(3 / 4)


def test_aggregate_both_conventions_by_hand():
    # chip B (1x4): target 1 1 1 1, pred 1 1 0 0 -> burned TP=2 FN=2: IoU_1=0.5; IoU_0: union=2 FN-> 0/2=0
    tb, pb = np.array([[1, 1, 1, 1]]), np.array([[1, 1, 0, 0]])
    # chip C: all ignore; chip D: empty burn, perfect
    tc, pc = np.full((1, 2), -1), np.zeros((1, 2), int)
    td, pd = np.zeros((1, 3), int), np.zeros((1, 3), int)
    agg = aggregate([chip_scores(TA, PA), chip_scores(tb, pb), chip_scores(tc, pc), chip_scores(td, pd)])
    # dataset-level: A cm [[1,1],[1,2]] + B [[0,0],[2,2]] + D [[3,0],[0,0]] = [[4,1],[3,4]]
    assert agg["confusion"] == [[4, 1], [3, 4]]
    assert agg["dataset_iou_burned"] == pytest.approx(4 / (4 + 1 + 3))      # 0.5
    assert agg["dataset_iou_unburned"] == pytest.approx(4 / (4 + 1 + 3))    # 0.5
    assert agg["dataset_miou"] == pytest.approx(0.5)
    # per-chip burned: A 0.5, B 0.5, C nan, D nan -> 0.5 over 2 defined chips
    assert agg["perchip_iou_burned"] == pytest.approx(0.5)
    assert agg["n_chips_burned_defined"] == 2
    # per-chip unburned: A 1/3, B 0, C nan, D 1 -> (1/3 + 0 + 1)/3
    assert agg["perchip_iou_unburned"] == pytest.approx((1 / 3 + 0 + 1) / 3)
    assert agg["n_chips"] == 4


def test_iou_from_confusion_empty_union_is_nan():
    iou = iou_from_confusion(np.array([[5, 0], [0, 0]]))
    assert iou[0] == 1.0 and math.isnan(iou[1])


def test_shape_mismatch():
    with pytest.raises(ValueError):
        chip_scores(np.zeros((2, 2), int), np.zeros((2, 3), int))
