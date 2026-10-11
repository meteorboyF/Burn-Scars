"""Chip loading for training/evaluation (torch-free core; torch wrapper is optional).

Normalisation statistics are computed from the run's OWN training chips (never
from held-out chips), on a fixed pixel subsample, and recorded in the manifest.
Masks keep -1 as the ignore index.
"""
from pathlib import Path

import numpy as np
import rasterio

from ..manifest import sha256_json
from ..paths import data_dir
from .chips import build_index

N_BANDS = 6


class ChipStore:
    """Reads scenes/masks by chip_id from the extracted dataset."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or data_dir() / "hls_burn_scars")
        self.index = build_index(self.root).set_index("chip_id")

    def read(self, chip_id: str, size: int | None = None):
        r = self.index.loc[chip_id]
        with rasterio.open(self.root / r.scene_path) as s:
            x = s.read().astype(np.float32)
        with rasterio.open(self.root / r.mask_path) as m:
            y = m.read(1).astype(np.int64)
        if size is not None and size != x.shape[-1]:  # smoke only: strided subsample
            step = x.shape[-1] // size
            x, y = x[:, ::step, ::step][:, :size, :size], y[::step, ::step][:size, :size]
        return x, y


def band_stats(store: ChipStore, chip_ids, n_px_per_chip=4096, seed=0) -> dict:
    """Per-band mean/std over valid pixels of the given (training) chips."""
    rng = np.random.default_rng(seed)
    acc = []
    for cid in sorted(chip_ids):
        x, y = store.read(cid)
        valid = np.flatnonzero(y.ravel() != -1)
        take = rng.choice(valid, min(n_px_per_chip, len(valid)), replace=False)
        acc.append(x.reshape(N_BANDS, -1)[:, take])
    a = np.concatenate(acc, axis=1)
    stats = {"mean": a.mean(1).tolist(), "std": a.std(1).tolist(), "n_chips": len(chip_ids),
             "n_px_per_chip": n_px_per_chip, "seed": seed}
    stats["hash"] = sha256_json(stats)
    return stats


def normalise(x: np.ndarray, stats: dict) -> np.ndarray:
    m = np.asarray(stats["mean"], np.float32)[:, None, None]
    s = np.asarray(stats["std"], np.float32)[:, None, None]
    return (x - m) / s


def d4(x: np.ndarray, y: np.ndarray, k: int):
    """One of the 8 dihedral transforms (k in 0..7), applied to image (C,H,W) and mask (H,W)."""
    if k >= 4:
        x, y = x[:, :, ::-1], y[:, ::-1]
    r = k % 4
    return np.ascontiguousarray(np.rot90(x, r, axes=(1, 2))), np.ascontiguousarray(np.rot90(y, r))
