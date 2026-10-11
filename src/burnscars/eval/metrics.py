"""Metric replica: burned-class IoU and two-class mIoU, label -1 always ignored.

Two aggregation conventions, both reported, never mixed silently:

  (a) per-chip   : IoU computed on each chip, then averaged over chips. A chip
                   where a class has an empty union (no GT and no prediction of
                   it) gives NaN for that class and is EXCLUDED from that
                   class's mean (nanmean). An all-ignore chip gives NaN for both.
  (b) dataset    : confusion counts (TP, FP, FN per class) accumulated over all
                   valid pixels of all chips, then one IoU per class. This is
                   the "micro" convention of torchmetrics-style accumulators.

mIoU = mean over the two classes {0 unburned, 1 burned} of the per-class IoU.
Which convention PANGAEA / TerraTorch / the Prithvi card use: docs/DATA.md
(metric conventions) — do not assume.

Inputs: `target` int array with values in {-1, 0, 1}; `pred` int/bool array of
the same shape with values in {0, 1} (pred at ignored pixels is irrelevant).
"""
import numpy as np

IGNORE = -1
CLASSES = (0, 1)


def _check(target, pred):
    target = np.asarray(target)
    pred = np.asarray(pred)
    if target.shape != pred.shape:
        raise ValueError(f"shape mismatch {target.shape} vs {pred.shape}")
    bad_t = ~np.isin(target, (IGNORE, 0, 1))
    if bad_t.any():
        raise ValueError(f"target has values outside {{-1,0,1}}: {np.unique(target[bad_t])}")
    valid = target != IGNORE
    bad_p = ~np.isin(pred[valid], (0, 1))
    if bad_p.any():
        raise ValueError("pred must be binary {0,1} at valid pixels")
    return target, pred.astype(np.int64), valid


def confusion(target, pred) -> np.ndarray:
    """2x2 confusion matrix C[t, p] over valid (non-ignored) pixels."""
    target, pred, valid = _check(target, pred)
    t = target[valid].astype(np.int64)
    p = pred[valid]
    return np.bincount(2 * t + p, minlength=4).reshape(2, 2)


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU [IoU_0, IoU_1]; NaN where a class's union is empty."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, tp / union, np.nan)


def chip_scores(target, pred) -> dict:
    """Per-chip IoUs plus the counts needed for dataset-level accumulation."""
    cm = confusion(target, pred)
    iou = iou_from_confusion(cm)
    return {"iou_unburned": iou[0], "iou_burned": iou[1],
            "miou": float(np.nanmean(iou)) if not np.isnan(iou).all() else np.nan,
            "n_valid": int(cm.sum()), "cm": cm}


def aggregate(chips) -> dict:
    """Both conventions from an iterable of chip_scores dicts."""
    chips = list(chips)
    if not chips:
        raise ValueError("no chips")
    total = sum(c["cm"] for c in chips)
    ds = iou_from_confusion(total)
    pc_b = np.array([c["iou_burned"] for c in chips], dtype=float)
    pc_u = np.array([c["iou_unburned"] for c in chips], dtype=float)
    pc_m = np.array([c["miou"] for c in chips], dtype=float)
    nan = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")  # noqa: E731
    return {
        "dataset_iou_burned": float(ds[1]),
        "dataset_iou_unburned": float(ds[0]),
        "dataset_miou": float(np.nanmean(ds)),
        "perchip_iou_burned": nan(pc_b),
        "perchip_iou_unburned": nan(pc_u),
        "perchip_miou": nan(pc_m),
        "n_chips": len(chips),
        "n_chips_burned_defined": int(np.isfinite(pc_b).sum()),
        "confusion": total.tolist(),
    }
