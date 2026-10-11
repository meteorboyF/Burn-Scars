"""Deterministic CV fold builders for the matched random-vs-grouped comparison.

random  : stratified random K-fold. Chips sorted by burned fraction, cut into
          consecutive blocks of K, each block shuffled across folds. Random with
          respect to geography, balanced on count and burned fraction like the
          grouped regime, so the regimes differ only in spatial structure.
grouped : whole groups (5 km single-linkage components) assigned to folds:
          largest-first greedy on count, then pairwise group swaps that reduce
          an objective = count imbalance^2 + burned-fraction-mean imbalance^2.

Equal training sizes: every (regime, fold) training set is subsampled, seeded,
to the minimum training size over both regimes and all folds.

All functions take explicit seeds and return plain lists sorted by id, so the
same inputs always give byte-identical JSON.
"""
import numpy as np


def _rng(seed):
    return np.random.default_rng(seed)


def stratified_random_folds(ids, burned_frac, k, seed) -> dict:
    ids = np.asarray(ids)
    bf = np.asarray(burned_frac, dtype=float)
    order = np.lexsort((ids, bf))  # by burned fraction, ties by id
    rng = _rng(seed)
    fold = np.empty(len(ids), dtype=int)
    for start in range(0, len(ids), k):
        block = order[start:start + k]
        fold[block] = rng.permutation(k)[: len(block)]
    return {str(i): int(f) for i, f in zip(ids, fold)}


def _objective(counts, bf_sums, n_total, bf_mean, bf_sd, k):
    target = n_total / k
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, bf_sums / counts, 0.0)
    return float(np.sum(((counts - target) / target) ** 2)
                 + np.sum(((means - bf_mean) / bf_sd) ** 2))


def grouped_folds(ids, groups, burned_frac, k, seed, max_iter=20000) -> dict:
    ids = np.asarray(ids)
    groups = np.asarray(groups)
    bf = np.asarray(burned_frac, dtype=float)
    labels = np.unique(groups)
    g_size = np.array([np.sum(groups == g) for g in labels])
    g_bf = np.array([bf[groups == g].sum() for g in labels])
    if g_size.max() > len(ids) / k:
        raise ValueError(f"largest group ({g_size.max()} chips) exceeds a fold's share "
                         f"({len(ids) / k:.1f}); cannot balance {k} folds")
    rng = _rng(seed)
    tie = rng.permutation(len(labels))
    order = np.lexsort((tie, -g_size))
    assign = np.empty(len(labels), dtype=int)
    counts = np.zeros(k)
    sums = np.zeros(k)
    for gi in order:
        f = int(np.argmin(counts))
        assign[gi] = f
        counts[f] += g_size[gi]
        sums[f] += g_bf[gi]

    bf_mean, bf_sd = bf.mean(), bf.std() or 1.0
    obj = _objective(counts, sums, len(ids), bf_mean, bf_sd, k)
    for _ in range(max_iter):
        a, b = rng.choice(len(labels), 2, replace=False)
        fa, fb = assign[a], assign[b]
        if fa == fb:
            continue
        c2, s2 = counts.copy(), sums.copy()
        c2[fa] += g_size[b] - g_size[a]
        c2[fb] += g_size[a] - g_size[b]
        s2[fa] += g_bf[b] - g_bf[a]
        s2[fb] += g_bf[a] - g_bf[b]
        o2 = _objective(c2, s2, len(ids), bf_mean, bf_sd, k)
        if o2 < obj - 1e-12:
            assign[a], assign[b] = fb, fa
            counts, sums, obj = c2, s2, o2
    g2f = dict(zip(labels.tolist(), assign.tolist()))
    return {str(i): int(g2f[g]) for i, g in zip(ids, groups.tolist())}


def fold_members(assign: dict, k: int) -> list[list[str]]:
    return [sorted(i for i, f in assign.items() if f == fold) for fold in range(k)]


def subsample(ids, n, seed) -> tuple[list[str], list[str]]:
    ids = sorted(ids)
    if n > len(ids):
        raise ValueError(f"cannot subsample {n} from {len(ids)}")
    keep = set(_rng(seed).choice(len(ids), n, replace=False).tolist())
    kept = [x for j, x in enumerate(ids) if j in keep]
    dropped = [x for j, x in enumerate(ids) if j not in keep]
    return kept, dropped
