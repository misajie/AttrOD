"""Condition S pipeline (draft2) — Ticket 2 exact engine wiring.

λ = sum(T)/sum(R); score (T, λR) and (T, λRᵀ); Δ = CPC(T,λRᵀ)-CPC(T,λR);
independence null O_i^T D_j^T / sum D^T.
Day bootstrap 1000 when D≥2 valid weekdays; Lombardy has no day stack.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from attrOD.metrics.condition_s import (
    compute_delta,
    compute_lambda_r,
    compute_lambda_rt,
    day_bootstrap as day_bootstrap_engine,
    evaluate_independence_null,
    metrics_to_row,
)
from attrOD.metrics.core import cpc

# Re-export engine symbols for callers expecting pipeline API
__all__ = [
    "score_condition_s",
    "bootstrap_days",
    "split_half_cpc",
    "compute_lambda_r",
    "compute_lambda_rt",
    "compute_delta",
    "evaluate_independence_null",
    "day_bootstrap",
    "run_condition_s_full",
]


def score_condition_s(
    T: np.ndarray,
    R: np.ndarray,
    distances: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    T = np.asarray(T, dtype=float)
    R = np.asarray(R, dtype=float)
    if float(R.sum()) <= 0:
        return {"skip": 1.0, "reason_zero_R": 1.0}
    delta = compute_delta(T, R, distances=distances)
    ind = evaluate_independence_null(T, distances=distances, observed=delta)
    scores_r = delta.get("scores_R") or {}
    scores_rt = delta.get("scores_RT") or {}
    out = {
        "lambda": delta["lambda"],
        "h_intra_T": delta.get("h_intra_T") if delta.get("h_intra_T") is not None else float("nan"),
        "CPC_R": delta["CPC_R"],
        "CPC_RT": delta["CPC_RT"],
        "Delta": delta["Delta"],
        "CPL": scores_r.get("CPL", float("nan")),
        "CPCd": scores_r.get("CPCd", float("nan")),
        "R2": scores_r.get("R2", float("nan")),
        "NRMSE": scores_r.get("NRMSE", float("nan")),
        "JSD": scores_r.get("JSD", float("nan")),
        "CPC_indep": ind["CPC_indep"],
        "CPC_RT_full": scores_rt.get("CPC", delta["CPC_RT"]),
    }
    return out


def bootstrap_days(
    T_by_day: Sequence[np.ndarray],
    R_by_day: Sequence[np.ndarray],
    distances: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Compat wrapper: returns flat summary dict (legacy keys)."""
    seed = 0
    if rng is not None:
        # draw a seed from rng for reproducibility bridge
        seed = int(rng.integers(0, 2**31 - 1))
    full = day_bootstrap_engine(
        T_by_day, R_by_day, distances=distances, n_resamples=n_resamples, seed=seed
    )
    return dict(full["bootstrap_summary"])


def day_bootstrap(
    T_by_day: Sequence[np.ndarray],
    R_by_day: Sequence[np.ndarray],
    distances: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    seed: int = 0,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """Full day-bootstrap audit payload (Ticket 2 / P2-MP)."""
    return day_bootstrap_engine(
        T_by_day, R_by_day, distances=distances, n_resamples=n_resamples, seed=seed, n_jobs=n_jobs
    )


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
    sa, sb = float(a.sum()), float(b.sum())
    if sa <= 0 or sb <= 0:
        return float("nan")
    b = b * (sa / sb)
    return cpc(a, b)


def run_condition_s_full(
    T: np.ndarray,
    R: np.ndarray,
    *,
    distances: Optional[np.ndarray] = None,
    T_by_day: Optional[Sequence[np.ndarray]] = None,
    R_by_day: Optional[Sequence[np.ndarray]] = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
    n_jobs: int = 1,
    study_area: str = "",
    partition: str = "full",
    n_perm_null: int = 0,
) -> Dict[str, Any]:
    """Unified Condition S run returning json/parquet-ready payloads."""
    delta = compute_delta(T, R, distances=distances)
    null = evaluate_independence_null(
        T, distances=distances, observed=delta, n_perm=n_perm_null, seed=seed
    )
    point = score_condition_s(T, R, distances)
    row = metrics_to_row(delta, study_area=study_area, partition=partition)
    row["CPC_indep"] = null["CPC_indep"]
    payload: Dict[str, Any] = {
        "metrics": point,
        "metrics_row": row,
        "delta": {k: v for k, v in delta.items() if k not in ("scores_R", "scores_RT")},
        "independence_null": {k: v for k, v in null.items() if k != "perm_null" or n_perm_null},
        "bootstrap": None,
    }
    if T_by_day is not None and R_by_day is not None and len(T_by_day) >= 2:
        payload["bootstrap"] = day_bootstrap(
            T_by_day, R_by_day, distances=distances, n_resamples=n_bootstrap, seed=seed, n_jobs=n_jobs
        )
    return payload
