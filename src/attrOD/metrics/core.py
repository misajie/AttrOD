"""Metric formulas exact as draft2 Implementation notes.

CPC uses Lenormand / scikit-mobility denominator:
  CPC(A,B) = 2 * sum min(A_ij, B_ij) / (sum A + sum B)

CPL: Sørensen–Dice on binary support (flow > 0 after aggregation; <1e-6 → 0).
CPCd: Sørensen–Dice on binned Euclidean (or network) trip-length distribution.
Pearson on log(1+A), log(1+B) over cells with A+B>0; R² = r².
NRMSE: RMSE over all N² cells (incl. zeros) / mean cell of first argument.
JSD: 20 log-spaced bins on [0, max(A_ij,B_ij)], empty bins kept (shared support).
Δ = CPC(T, λ Rᵀ) - CPC(T, λ R).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


def _as_float_2d(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    if x.ndim != 2:
        raise ValueError("expected 2D matrix")
    return x


def cpc(A: np.ndarray, B: np.ndarray) -> float:
    A = _as_float_2d(A)
    B = _as_float_2d(B)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch {A.shape} vs {B.shape}")
    denom = float(A.sum() + B.sum())
    if denom <= 0:
        return float("nan")
    return float(2.0 * np.minimum(A, B).sum() / denom)


def cpl(A: np.ndarray, B: np.ndarray) -> float:
    """Sørensen–Dice on binary support."""
    A = _as_float_2d(A)
    B = _as_float_2d(B)
    sa = A > 0
    sb = B > 0
    inter = np.logical_and(sa, sb).sum()
    denom = sa.sum() + sb.sum()
    if denom <= 0:
        return float("nan")
    return float(2.0 * inter / denom)


def _length_hist(
    flows: np.ndarray,
    distances: np.ndarray,
    bins: Sequence[float],
) -> np.ndarray:
    """Mass-weighted histogram of trip lengths (cell mass × length bin)."""
    f = flows.ravel()
    d = distances.ravel()
    if f.shape != d.shape:
        raise ValueError("flows and distances must share shape")
    # digitize
    edges = np.asarray(bins, dtype=float)
    # bins are edges; last is +inf style if needed
    h = np.zeros(len(edges) - 1, dtype=float)
    # only positive flows contribute
    mask = f > 0
    if not np.any(mask):
        return h
    idx = np.digitize(d[mask], edges[1:-1], right=False)
    # digitize with inner edges → index 0..n-1
    for k, mass in zip(idx, f[mask]):
        k = int(np.clip(k, 0, len(h) - 1))
        h[k] += float(mass)
    return h


def cpcd(
    A: np.ndarray,
    B: np.ndarray,
    distances: np.ndarray,
    bins: Optional[Sequence[float]] = None,
) -> float:
    """Sørensen–Dice on binned trip-length distributions of A and B."""
    A = _as_float_2d(A)
    B = _as_float_2d(B)
    D = _as_float_2d(distances)
    if bins is None:
        # default mobile-network-ish edges in km + open end
        bins = [0.0, 0.5, 2.0, 10.0, 50.0, np.inf]
    ha = _length_hist(A, D, bins)
    hb = _length_hist(B, D, bins)
    denom = ha.sum() + hb.sum()
    if denom <= 0:
        return float("nan")
    return float(2.0 * np.minimum(ha, hb).sum() / denom)


def pearson_log(A: np.ndarray, B: np.ndarray) -> float:
    """Pearson r on log(1+·) over cells with A_ij + B_ij > 0."""
    A = _as_float_2d(A)
    B = _as_float_2d(B)
    mask = (A + B) > 0
    if mask.sum() < 2:
        return float("nan")
    x = np.log1p(A[mask])
    y = np.log1p(B[mask])
    if np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def r2_log(A: np.ndarray, B: np.ndarray) -> float:
    r = pearson_log(A, B)
    return float(r * r) if np.isfinite(r) else float("nan")


def nrmse(A: np.ndarray, B: np.ndarray) -> float:
    """RMSE over all N² cells / mean cell of A."""
    A = _as_float_2d(A)
    B = _as_float_2d(B)
    mean_a = float(A.mean())
    if mean_a == 0:
        return float("nan")
    rmse = float(np.sqrt(np.mean((A - B) ** 2)))
    return rmse / mean_a


def jsd(A: np.ndarray, B: np.ndarray, n_bins: int = 20) -> float:
    """Jensen–Shannon divergence of normalised cell-mass histograms.

    20 log-spaced bins on [0, max(A_ij, B_ij)]; empty bins kept so support is shared.
    """
    A = _as_float_2d(A)
    B = _as_float_2d(B)
    flat_a = A.ravel()
    flat_b = B.ravel()
    vmax = float(max(flat_a.max(), flat_b.max()))
    if vmax <= 0:
        return 0.0
    # log-spaced edges on (0, vmax]; include a bin for exact zeros via left edge 0
    inner = np.logspace(np.log10(max(vmax / 1e6, 1e-12)), np.log10(vmax), n_bins)
    edges = np.concatenate([[0.0], inner])
    # ensure length n_bins+1 unique edges
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(0.0, vmax, n_bins + 1)
    ha, _ = np.histogram(flat_a, bins=edges)
    hb, _ = np.histogram(flat_b, bins=edges)
    pa = ha.astype(float)
    pb = hb.astype(float)
    sa, sb = pa.sum(), pb.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    pa /= sa
    pb /= sb
    m = 0.5 * (pa + pb)
    def _kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], EPS))))
    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


def delta_cpc(T: np.ndarray, R: np.ndarray, lam: Optional[float] = None) -> Tuple[float, float, float, float]:
    """Return (lambda, cpc_R, cpc_RT, delta)."""
    T = _as_float_2d(T)
    R = _as_float_2d(R)
    sum_r = float(R.sum())
    sum_t = float(T.sum())
    if lam is None:
        if sum_r <= 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        lam = sum_t / sum_r
    R_down = lam * R
    c_r = cpc(T, R_down)
    c_rt = cpc(T, lam * R.T)
    return float(lam), c_r, c_rt, float(c_rt - c_r)


def independence_table(T: np.ndarray) -> np.ndarray:
    """O_i^T D_j^T / sum D^T."""
    T = _as_float_2d(T)
    O = T.sum(axis=1, keepdims=True)
    D = T.sum(axis=0, keepdims=True)
    s = float(D.sum())
    if s <= 0:
        return np.zeros_like(T)
    return (O @ D) / s


def score_pair(
    A: np.ndarray,
    B: np.ndarray,
    distances: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    out = {
        "CPC": cpc(A, B),
        "CPL": cpl(A, B),
        "R2": r2_log(A, B),
        "pearson_r": pearson_log(A, B),
        "NRMSE": nrmse(A, B),
        "JSD": jsd(A, B),
    }
    if distances is not None:
        out["CPCd"] = cpcd(A, B, distances)
    else:
        out["CPCd"] = float("nan")
    return out
