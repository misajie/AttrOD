"""Condition S pipeline (draft2).

λ = sum(T)/sum(R); score (T, λR) and (T, λRᵀ); Δ = CPC(T,λRᵀ)-CPC(T,λR);
independence null O_i^T D_j^T / sum D^T.
Day bootstrap 1000 when D≥2 valid weekdays; Lombardy has no day stack.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from attrOD.data.flow_table import h_intra
from attrOD.metrics.core import (
    cpc,
    delta_cpc,
    independence_table,
    score_pair,
)


def score_condition_s(
    T: np.ndarray,
    R: np.ndarray,
    distances: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    T = np.asarray(T, dtype=float)
    R = np.asarray(R, dtype=float)
    if float(R.sum()) <= 0:
        return {"skip": 1.0, "reason_zero_R": 1.0}
    lam, c_r, c_rt, delta = delta_cpc(T, R)
    R_down = lam * R
    scores_r = score_pair(T, R_down, distances)
    scores_rt = score_pair(T, lam * R.T, distances)
    ind = independence_table(T)
    scores_ind = score_pair(T, ind, distances)
    out = {
        "lambda": lam,
        "h_intra_T": h_intra(T),
        "CPC_R": c_r,
        "CPC_RT": c_rt,
        "Delta": delta,
        "CPL": scores_r["CPL"],
        "CPCd": scores_r["CPCd"],
        "R2": scores_r["R2"],
        "NRMSE": scores_r["NRMSE"],
        "JSD": scores_r["JSD"],
        "CPC_indep": scores_ind["CPC"],
        "CPC_RT_full": scores_rt["CPC"],
    }
    return out


def bootstrap_days(
    T_by_day: Sequence[np.ndarray],
    R_by_day: Sequence[np.ndarray],
    distances: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Recompute scores on each day; summarise mean + percentile bootstrap over days.

    Requires D≥2. Each resample draws days with replacement, averages day-level CPC/Δ.
    """
    D = len(T_by_day)
    if D < 2:
        raise ValueError("day bootstrap requires D>=2")
    if len(R_by_day) != D:
        raise ValueError("T_by_day and R_by_day length mismatch")
    rng = rng or np.random.default_rng(0)
    day_scores = [score_condition_s(T_by_day[d], R_by_day[d], distances) for d in range(D)]
    keys = ["CPC_R", "CPC_RT", "Delta", "lambda"]
    means = {k: float(np.nanmean([s[k] for s in day_scores])) for k in keys}

    boot = {k: [] for k in keys}
    idx = np.arange(D)
    for _ in range(n_resamples):
        take = rng.choice(idx, size=D, replace=True)
        for k in keys:
            boot[k].append(float(np.nanmean([day_scores[i][k] for i in take])))
    out = {f"mean_{k}": means[k] for k in keys}
    for k in keys:
        arr = np.asarray(boot[k], dtype=float)
        out[f"p2.5_{k}"] = float(np.nanpercentile(arr, 2.5))
        out[f"p97.5_{k}"] = float(np.nanpercentile(arr, 97.5))
    return out


def split_half_cpc(
    mats_by_day: Sequence[np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Noise ceiling: random disjoint halves of days, mass-matched, CPC of partition vs itself."""
    D = len(mats_by_day)
    if D < 2:
        return float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.permutation(D)
    half = D // 2
    a = sum(mats_by_day[i] for i in idx[:half])
    b = sum(mats_by_day[i] for i in idx[half : 2 * half])
    # mass-match
    sa, sb = float(a.sum()), float(b.sum())
    if sa <= 0 or sb <= 0:
        return float("nan")
    b = b * (sa / sb)
    return cpc(a, b)
