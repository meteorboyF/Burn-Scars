"""Trivial baselines and the NBR leakage control.

NBR = (B8A - B12) / (B8A + B12) on the post-fire chip (no pre-fire image exists
in this dataset). Rule: burned iff NBR < t. Pixels where NBR is undefined
(zero/non-finite denominator) are predicted unburned. t is tuned to maximise
dataset-level burned IoU on TRAINING chips only, by exact histogram counting:
for a threshold on the grid edge e_j, "v < e_j" holds iff
searchsorted(edges, v, 'right') <= j, so cumulative bin counts reproduce the
per-pixel rule exactly (tests/test_baselines.py checks this).
"""
import numpy as np

from .metrics import IGNORE

BAND_INDEX = {"B02": 0, "B03": 1, "B04": 2, "B8A": 3, "B11": 4, "B12": 5}
EDGES = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.005), 6)  # 401 candidate thresholds


def nbr(bands: np.ndarray) -> np.ndarray:
    """bands: (6, H, W) float reflectance in dataset band order. Returns (H, W), NaN where undefined."""
    nir = bands[BAND_INDEX["B8A"]].astype(np.float64)
    swir2 = bands[BAND_INDEX["B12"]].astype(np.float64)
    den = nir + swir2
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (nir - swir2) / den
    out[~np.isfinite(out) | (den == 0)] = np.nan
    return out


def predict_nbr(index: np.ndarray, t: float) -> np.ndarray:
    return (np.nan_to_num(index, nan=np.inf) < t).astype(np.int64)


def predict_all_unburned(target: np.ndarray) -> np.ndarray:
    return np.zeros_like(target, dtype=np.int64)


class NBRHistogram:
    """Accumulates burned / unburned valid-pixel counts per NBR bin over chips."""

    def __init__(self, edges=EDGES):
        self.edges = np.asarray(edges, dtype=np.float64)
        n = len(self.edges) + 1
        self.burned = np.zeros(n, np.int64)
        self.unburned = np.zeros(n, np.int64)
        self.nan_burned = 0  # undefined NBR -> always predicted unburned
        self.nan_unburned = 0

    def add(self, index: np.ndarray, target: np.ndarray):
        valid = target != IGNORE
        v, t = index[valid], target[valid]
        fin = np.isfinite(v)
        self.nan_burned += int(((~fin) & (t == 1)).sum())
        self.nan_unburned += int(((~fin) & (t == 0)).sum())
        k = np.searchsorted(self.edges, v[fin], side="right")
        n = len(self.edges) + 1
        self.burned += np.bincount(k[t[fin] == 1], minlength=n)
        self.unburned += np.bincount(k[t[fin] == 0], minlength=n)

    def iou_burned_curve(self) -> np.ndarray:
        """Dataset-level burned IoU for each threshold edges[j]."""
        tp = np.cumsum(self.burned)[: len(self.edges)]      # bins k <= j
        fp = np.cumsum(self.unburned)[: len(self.edges)]
        pos = self.burned.sum() + self.nan_burned
        fn = pos - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)

    def best_threshold(self) -> tuple[float, float]:
        curve = self.iou_burned_curve()
        j = int(np.nanargmax(curve))  # first maximum -> deterministic tie-break
        return float(self.edges[j]), float(curve[j])
