"""Condition S exact engine (Ticket 2 / draft2).

λ = sum(T)/sum(R); R↓ = λR
Δ = CPC(T, λ Rᵀ) − CPC(T, λ R)
Independence null: O_i^T D_j^T / sum D^T
Day bootstrap: resample days with replacement; keep within-day OD structure.
"""
from __future__ import annotations

import concurrent.futures
import os

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from attrOD.data.flow_table import h_intra
from attrOD.metrics.core import cpc, delta_cpc, independence_table, score_pair

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def compute_lambda(T: ArrayLike, R: ArrayLike) -> float:
    T = np.asarray(T, dtype=float)
    R = np.asarray(R, dtype=float)
    s_r = float(R.sum())
    if s_r <= 0:
        return float("nan")
    return float(T.sum()) / s_r


def compute_lambda_r(
    T: ArrayLike,
    R: ArrayLike,
    *,
    partition: Optional[Mapping[str, Any]] = None,
    distances: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Score (T, λR). Returns λ, CPC, companions, and intermediate mass."""
    T = np.asarray(T, dtype=float)
    R = np.asarray(R, dtype=float)
    lam = compute_lambda(T, R)
    if not np.isfinite(lam):
        return {
            "lambda": lam,
            "CPC": float("nan"),
            "R_down": np.zeros_like(R),
            "scores": {},
            "partition_id": (partition or {}).get("id"),
            "definition": "lambda_R = (sum T / sum R) * R",
        }
    R_down = lam * R
    scores = score_pair(T, R_down, distances)
    return {
        "lambda": lam,
        "CPC": scores["CPC"],
        "R_down": R_down,
        "scores": scores,
        "h_intra_T": h_intra(T),
        "partition_id": (partition or {}).get("id"),
        "definition": "lambda_R = (sum T / sum R) * R; score (T, λR)",
        "definition_version": "draft2",
    }


def compute_lambda_rt(
    T: ArrayLike,
    R: ArrayLike,
    *,
    lam: Optional[float] = None,
    partition: Optional[Mapping[str, Any]] = None,
    distances: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Score (T, λ Rᵀ) with the same λ as compute_lambda_r."""
    T = np.asarray(T, dtype=float)
    R = np.asarray(R, dtype=float)
    if lam is None:
        lam = compute_lambda(T, R)
    if not np.isfinite(lam):
        return {
            "lambda": lam,
            "CPC": float("nan"),
            "RT_down": np.zeros_like(R),
            "scores": {},
            "partition_id": (partition or {}).get("id"),
            "definition": "lambda_RT = λ * R.T",
        }
    RT_down = lam * R.T
    scores = score_pair(T, RT_down, distances)
    return {
        "lambda": float(lam),
        "CPC": scores["CPC"],
        "RT_down": RT_down,
        "scores": scores,
        "partition_id": (partition or {}).get("id"),
        "definition": "lambda_RT = λ * R.T; score (T, λRᵀ)",
        "definition_version": "draft2",
    }


def compute_delta(
    T: ArrayLike,
    R: ArrayLike,
    *,
    lam: Optional[float] = None,
    partition: Optional[Mapping[str, Any]] = None,
    distances: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Δ = CPC(T, λRᵀ) − CPC(T, λR) per draft2."""
    T = np.asarray(T, dtype=float)
    R = np.asarray(R, dtype=float)
    lam_out, c_r, c_rt, delta = delta_cpc(T, R, lam=lam)
    lr = compute_lambda_r(T, R, partition=partition, distances=distances)
    lrt = compute_lambda_rt(T, R, lam=lam_out, partition=partition, distances=distances)
    return {
        "lambda": lam_out,
        "CPC_R": c_r,
        "CPC_RT": c_rt,
        "Delta": delta,
        "definition": "Delta = CPC(T, λRᵀ) - CPC(T, λR)",
        "definition_version": "draft2",
        "precision_note": "float64 CPC via Lenormand / scikit-mobility denominator",
        "partition_id": (partition or {}).get("id"),
        "scores_R": lr["scores"],
        "scores_RT": lrt["scores"],
        "h_intra_T": lr.get("h_intra_T"),
    }


def evaluate_independence_null(
    T: ArrayLike,
    *,
    distances: Optional[np.ndarray] = None,
    observed: Optional[Mapping[str, Any]] = None,
    n_perm: int = 0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Independence table null O_i^T D_j^T / sum D^T vs T.

    When *n_perm* > 0, also shuffle destination margins to build a simple
    calibration reference (optional; draft2 primary null is the closed form).
    """
    T = np.asarray(T, dtype=float)
    ind = independence_table(T)
    scores = score_pair(T, ind, distances)
    out: Dict[str, Any] = {
        "null_name": "independence_O_i_D_j",
        "definition": "O_i^T D_j^T / sum_k D_k^T",
        "CPC_indep": scores["CPC"],
        "scores": scores,
        "row_sums_match": bool(np.allclose(ind.sum(1), T.sum(1))),
        "col_sums_match": bool(np.allclose(ind.sum(0), T.sum(0))),
        "total_match": bool(np.isclose(ind.sum(), T.sum())),
    }
    if observed is not None:
        out["observed_CPC_R"] = observed.get("CPC_R")
        out["observed_Delta"] = observed.get("Delta")
        if observed.get("CPC_R") is not None and np.isfinite(scores["CPC"]):
            out["CPC_R_minus_indep"] = float(observed["CPC_R"] - scores["CPC"])

    if n_perm > 0:
        rng = np.random.default_rng(seed)
        O = T.sum(axis=1)
        D = T.sum(axis=0)
        s = float(D.sum()) or 1.0
        null_cpcs = []
        for _ in range(n_perm):
            D_perm = rng.permutation(D)
            ind_p = np.outer(O, D_perm) / s
            null_cpcs.append(cpc(T, ind_p))
        arr = np.asarray(null_cpcs, dtype=float)
        out["perm_null"] = {
            "n_perm": n_perm,
            "seed": seed,
            "CPC_mean": float(np.nanmean(arr)),
            "CPC_p2.5": float(np.nanpercentile(arr, 2.5)),
            "CPC_p97.5": float(np.nanpercentile(arr, 97.5)),
            "samples": arr.tolist(),
        }
    return out


_BOOTSTRAP_STATE: dict = {}


def _bootstrap_init(T_by_day, R_by_day, distances) -> None:
    _BOOTSTRAP_STATE["T"] = T_by_day
    _BOOTSTRAP_STATE["R"] = R_by_day
    _BOOTSTRAP_STATE["D"] = distances


def _bootstrap_one(payload):
    """payload: (resample_idx, take_tuple) with precomputed day indices."""
    resample_idx, take = payload
    take = list(take)
    T_by_day = _BOOTSTRAP_STATE["T"]
    R_by_day = _BOOTSTRAP_STATE["R"]
    distances = _BOOTSTRAP_STATE["D"]
    Tb = sum(np.asarray(T_by_day[i], dtype=float) for i in take)
    Rb = sum(np.asarray(R_by_day[i], dtype=float) for i in take)
    est = compute_delta(Tb, Rb, distances=distances)
    return {
        "resample": int(resample_idx),
        "day_indices": take,
        "lambda": est["lambda"],
        "CPC_R": est["CPC_R"],
        "CPC_RT": est["CPC_RT"],
        "Delta": est["Delta"],
    }

def day_bootstrap(
    T_by_day: Sequence[np.ndarray],
    R_by_day: Sequence[np.ndarray],
    *,
    distances: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    seed: int = 0,
    keys: Sequence[str] = ("lambda", "CPC_R", "CPC_RT", "Delta"),
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """Day-unit bootstrap: resample days with replacement; keep within-day OD.

    For each resampled multiset of days, aggregate T and R by summing day
    matrices (preserving within-day structure), then recompute lambda/CPC/Delta.
    Also stores per-day point estimates for audit.

    n_jobs > 1 parallelises resamples across processes. Day indices are drawn
    up front with ``seed``, so multi-process results match single-process science
    (up to floating-point reduction order). On Linux, prefer fork so day stacks
    are inherited (no multi-GB pickle of N≈3909 matrices).
    """
    import multiprocessing as _mp

    D = len(T_by_day)
    if D < 2:
        raise ValueError("day bootstrap requires D>=2")
    if len(R_by_day) != D:
        raise ValueError("T_by_day and R_by_day length mismatch")

    rng = np.random.default_rng(seed)
    day_rows: List[Dict[str, Any]] = []
    for d in range(D):
        delta = compute_delta(T_by_day[d], R_by_day[d], distances=distances)
        day_rows.append(
            {
                "day_index": d,
                "lambda": delta.get("lambda"),
                "CPC_R": delta.get("CPC_R"),
                "CPC_RT": delta.get("CPC_RT"),
                "Delta": delta.get("Delta"),
            }
        )

    T_sum = sum(np.asarray(t, dtype=float) for t in T_by_day)
    R_sum = sum(np.asarray(r, dtype=float) for r in R_by_day)
    pooled = compute_delta(T_sum, R_sum, distances=distances)

    idx = np.arange(D)
    takes = [rng.choice(idx, size=D, replace=True) for _ in range(n_resamples)]
    jobs = [(b, tuple(int(x) for x in takes[b])) for b in range(n_resamples)]

    n_jobs = int(n_jobs) if n_jobs is not None else 1
    if n_jobs <= 0:
        n_jobs = os.cpu_count() or 1

    _bootstrap_init(list(T_by_day), list(R_by_day), distances)

    if n_jobs == 1 or n_resamples <= 1:
        boot_records = [_bootstrap_one(job) for job in jobs]
    else:
        try:
            ctx = _mp.get_context("fork")
            use_init = False
        except ValueError:
            ctx = _mp.get_context("spawn")
            use_init = True

        chunk = max(1, n_resamples // (n_jobs * 4))
        if use_init:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_jobs,
                mp_context=ctx,
                initializer=_bootstrap_init,
                initargs=(list(T_by_day), list(R_by_day), distances),
            ) as ex:
                boot_records = list(ex.map(_bootstrap_one, jobs, chunksize=chunk))
        else:
            # fork: children inherit _BOOTSTRAP_STATE (COW); avoid pickling stacks
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_jobs,
                mp_context=ctx,
            ) as ex:
                boot_records = list(ex.map(_bootstrap_one, jobs, chunksize=chunk))
        boot_records.sort(key=lambda r: r["resample"])

    summary: Dict[str, float] = {}
    for k in keys:
        arr = np.asarray([row[k] for row in boot_records], dtype=float)
        summary[f"mean_{k}"] = float(np.nanmean(arr))
        summary[f"p2.5_{k}"] = float(np.nanpercentile(arr, 2.5))
        summary[f"p97.5_{k}"] = float(np.nanpercentile(arr, 97.5))
        summary[f"pooled_{k}"] = float(pooled.get(k, float("nan")))

    return {
        "n_days": D,
        "n_resamples": n_resamples,
        "seed": seed,
        "n_jobs": n_jobs,
        "unit": "day",
        "within_day_od_preserved": True,
        "day_point_estimates": day_rows,
        "bootstrap_summary": summary,
        "bootstrap_samples": boot_records,
        "pooled": {
            "lambda": pooled["lambda"],
            "CPC_R": pooled["CPC_R"],
            "CPC_RT": pooled["CPC_RT"],
            "Delta": pooled["Delta"],
        },
    }

def day_bootstrap_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten bootstrap samples to a parquet-friendly DataFrame."""
    rows = result.get("bootstrap_samples") or []
    if not rows:
        return pd.DataFrame(columns=["resample", "lambda", "CPC_R", "CPC_RT", "Delta"])
    df = pd.DataFrame(rows)
    # drop heavy day_indices list for parquet schema stability unless needed
    if "day_indices" in df.columns:
        df = df.drop(columns=["day_indices"])
    return df


def metrics_to_row(delta_result: Mapping[str, Any], *, study_area: str = "", partition: str = "") -> Dict[str, Any]:
    """Unify Condition S point estimate to table2-like row + companions."""
    scores_r = delta_result.get("scores_R") or {}
    return {
        "study_area": study_area,
        "partition": partition or delta_result.get("partition_id") or "full",
        "lambda": delta_result.get("lambda"),
        "CPC_T_lambdaR": delta_result.get("CPC_R"),
        "CPC_T_lambdaRT": delta_result.get("CPC_RT"),
        "Delta": delta_result.get("Delta"),
        "CPL": scores_r.get("CPL"),
        "CPCd": scores_r.get("CPCd"),
        "R2": scores_r.get("R2"),
        "NRMSE": scores_r.get("NRMSE"),
        "JSD": scores_r.get("JSD"),
        "h_intra_T": delta_result.get("h_intra_T"),
        "definition_version": delta_result.get("definition_version", "draft2"),
    }
