"""Statistics for H1-v2 (docs/REALITY_CHECK.md, committed 4b6344c). CPU only.

Metric separation (D-016): headline gaps and rankings use POOLED burned IoU
(confusion summed over chips, then one IoU); per-chip IoU is used only as the
response of the distance regressions.

Cluster bootstrap: resample the 5-km connected components (groups) with
replacement; a resample is a weight vector w (count of each group), and the
pooled confusion of any arm is w @ (per-group confusion sums). Every arm in a
comparison uses the SAME draws (paired).
"""
import numpy as np

B_DEFAULT = 10_000


def iou_burned_from_cm(cm) -> np.ndarray:
    """cm (..., 4) flattened [[tn, fp], [fn, tp]] -> burned IoU tp / (tp + fp + fn)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp, fp, fn = cm[..., 3], cm[..., 1], cm[..., 2]
    den = tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, tp / den, np.nan)


def group_sums(chip_ids, cms: dict, group_of: dict, groups: list):
    """(n_groups, 4) confusion sums. cms: chip_id -> 2x2 (or flat 4)."""
    gi = {g: i for i, g in enumerate(groups)}
    out = np.zeros((len(groups), 4))
    for c in chip_ids:
        out[gi[group_of[c]]] += np.asarray(cms[c], dtype=np.float64).ravel()
    return out


def bootstrap_weights(n_groups: int, B: int, seed: int) -> np.ndarray:
    """(B, n_groups) resample counts; each row is one cluster-bootstrap draw."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(B, n_groups))
    W = np.zeros((B, n_groups), dtype=np.int32)
    np.add.at(W, (np.repeat(np.arange(B), n_groups), idx.ravel()), 1)
    return W


def _ci(x, level=0.95):
    a = (1 - level) / 2
    return [float(np.nanquantile(x, a)), float(np.nanquantile(x, 1 - a))]


def paired_gap(arms: dict, chip_ids, group_of: dict, B=B_DEFAULT, seed=0, level=0.95) -> dict:
    """arms: name -> {chip_id: confusion}. Every chip must appear in every arm (paired).
    Returns pooled burned IoU per arm, and draws (B,) per arm for derived contrasts."""
    groups = sorted({group_of[c] for c in chip_ids})
    W = bootstrap_weights(len(groups), B, seed)
    out = {"n_chips": len(chip_ids), "n_groups": len(groups), "B": B, "seed": seed, "arms": {}}
    draws = {}
    for name, cms in arms.items():
        missing = set(chip_ids) - set(cms)
        if missing:
            raise ValueError(f"arm {name!r} lacks {len(missing)} chips: not a paired design")
        G = group_sums(chip_ids, cms, group_of, groups)
        point = float(iou_burned_from_cm(G.sum(0)))
        draws[name] = iou_burned_from_cm(W @ G)
        out["arms"][name] = {"pooled_iou_burned": point, "ci": _ci(draws[name], level)}
    out["_draws"] = draws
    return out


def contrast(res: dict, a: str, b: str, level=0.95) -> dict:
    """Point and paired-bootstrap CI of IoU(a) - IoU(b), in IoU points (x100)."""
    d = 100 * (res["_draws"][a] - res["_draws"][b])
    point = 100 * (res["arms"][a]["pooled_iou_burned"] - res["arms"][b]["pooled_iou_burned"])
    ci = _ci(d, level)
    return {"point_pts": point, "ci_pts": ci, "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0)}


def diff_in_diff(res: dict, a1, b1, a2, b2, level=0.95) -> dict:
    """[IoU(a1)-IoU(b1)] - [IoU(a2)-IoU(b2)], e.g. G_learned - G_NBR, same draws."""
    d = 100 * ((res["_draws"][a1] - res["_draws"][b1]) - (res["_draws"][a2] - res["_draws"][b2]))
    A = res["arms"]
    point = 100 * ((A[a1]["pooled_iou_burned"] - A[b1]["pooled_iou_burned"])
                   - (A[a2]["pooled_iou_burned"] - A[b2]["pooled_iou_burned"]))
    ci = _ci(d, level)
    return {"point_pts": point, "ci_pts": ci, "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0)}


def h1_decision(res, learned_rand, learned_grp, nbr_rand, nbr_grp, min_effect_pts=2.0) -> dict:
    """The pre-registered H1-v2 rule: G >= 2.0 AND CI(G) excludes 0 AND CI(G - G_NBR) excludes 0."""
    g = contrast(res, learned_rand, learned_grp)
    dd = diff_in_diff(res, learned_rand, learned_grp, nbr_rand, nbr_grp)
    return {"G": g, "G_minus_G_NBR": dd,
            "supported": bool(g["point_pts"] >= min_effect_pts and g["ci_excludes_0"] and dd["ci_excludes_0"])}


def rank_reversal(res: dict, m1_rand, m2_rand, m1_grp, m2_grp, threshold=0.9) -> dict:
    """P_boot that the order of (m1, m2) by pooled IoU differs between regimes.
    A reversal COUNTS only if P >= threshold (pre-registered 0.9)."""
    D = res["_draws"]
    s_r = np.sign(D[m1_rand] - D[m2_rand])
    s_g = np.sign(D[m1_grp] - D[m2_grp])
    p = float(np.mean(s_r != s_g))
    A = res["arms"]
    point_r = A[m1_rand]["pooled_iou_burned"] - A[m2_rand]["pooled_iou_burned"]
    point_g = A[m1_grp]["pooled_iou_burned"] - A[m2_grp]["pooled_iou_burned"]
    return {"p_reversal": p, "point_reversed": bool(np.sign(point_r) != np.sign(point_g)),
            "counts": bool(p >= threshold), "threshold": threshold}


def kendall_tau_by_fold(scores_rand: dict, scores_grp: dict) -> dict:
    """scores_*: model -> list of per-fold pooled IoU. Kendall tau between the model
    rankings of the two regimes, per fold (with 2 models tau is +/-1)."""
    from scipy.stats import kendalltau
    models = sorted(scores_rand)
    k = len(next(iter(scores_rand.values())))
    taus = []
    for f in range(k):
        t, _ = kendalltau([scores_rand[m][f] for m in models], [scores_grp[m][f] for m in models])
        taus.append(float(t))
    return {"models": models, "tau_per_fold": taus, "mean_tau": float(np.nanmean(taus))}


def _design(df, terms, year_ref=2018):
    cols, names = [np.ones(len(df))], ["intercept"]
    for t in terms:
        if t == "year":
            for y in sorted(df["year"].unique()):
                if y != year_ref:
                    cols.append((df["year"] == y).astype(float).to_numpy())
                    names.append(f"year_{y}")
        elif t == "log1p_dist_km":
            cols.append(np.log1p(df["dist_km"].to_numpy(float)))
            names.append(t)
        else:
            cols.append(df[t].to_numpy(float))
            names.append(t)
    return np.column_stack(cols), names


REGRESSIONS = {
    # pre-registered (H1-v2 / E-01): all three reported, none dropped
    "R0": ("iou_learned", ["log1p_dist_km", "burned_frac", "year"]),
    "R1": ("iou_learned", ["log1p_dist_km", "burned_frac", "year", "iou_nbr"]),
    "R2": ("iou_learned_minus_nbr", ["log1p_dist_km", "burned_frac", "year"]),
}


def distance_regressions(df, B=2000, seed=0, level=0.95) -> dict:
    """df columns: chip_id, group, dist_km, burned_frac, year, iou_learned, iou_nbr (per-chip,
    out-of-fold). OLS point estimates; CIs by cluster bootstrap over groups."""
    df = df.copy()
    df["iou_learned_minus_nbr"] = df["iou_learned"] - df["iou_nbr"]
    groups = np.array(sorted(df["group"].unique()))
    rows_of = {g: np.flatnonzero(df["group"].to_numpy() == g) for g in groups}
    rng = np.random.default_rng(seed)
    out = {}
    for name, (y_col, terms) in REGRESSIONS.items():
        X, names = _design(df, terms)
        y = df[y_col].to_numpy(float)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        boots = []
        for _ in range(B):
            take = np.concatenate([rows_of[g] for g in rng.choice(groups, len(groups), replace=True)])
            Xb, yb = X[take], y[take]
            if np.linalg.matrix_rank(Xb) < X.shape[1]:
                continue  # a draw missing a year level: skip, counted below
            boots.append(np.linalg.lstsq(Xb, yb, rcond=None)[0])
        boots = np.array(boots)
        j = names.index("log1p_dist_km")
        ci = _ci(boots[:, j], level)
        out[name] = {"response": y_col, "terms": names, "coef": dict(zip(names, beta.tolist())),
                     "dist_coef": float(beta[j]), "dist_ci": ci,
                     "dist_negative_and_excludes_0": bool(ci[1] < 0),
                     "n": len(df), "n_groups": len(groups), "B_used": len(boots)}
    return out
