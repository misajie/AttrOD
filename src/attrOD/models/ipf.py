"""IPF / Furness emission (not a law).

Seed: R (or gravity mean when a zero seed cell must be filled).
Row totals: O_i^T.
Column variants:
  (G-row) target inflows D_j^T
  (G-row-traincol) column totals from training-origin inflows of R, scaled to sum O^T
Stop: relative margin error 1e-4 or 1000 cycles.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np

from .emission import assert_production_constrained

Variant = Literal["G-row", "G-row-traincol"]


def ipf_emit(
    seed: np.ndarray,
    O_T: np.ndarray,
    *,
    D_T: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    train_origins: Optional[Sequence[int]] = None,
    variant: Variant = "G-row",
    tol: float = 1e-4,
    max_iter: int = 1000,
    fill_zero_with: Optional[np.ndarray] = None,
) -> np.ndarray:
    S = np.asarray(seed, dtype=float).copy()
    O = np.asarray(O_T, dtype=float).ravel()
    n = S.shape[0]
    if fill_zero_with is not None:
        fz = np.asarray(fill_zero_with, dtype=float)
        S = np.where(S > 0, S, fz)
    S = np.maximum(S, 0.0)
    # avoid all-zero rows/cols
    S = S + 1e-12

    if variant == "G-row":
        if D_T is None:
            raise ValueError("G-row requires D_T")
        D = np.asarray(D_T, dtype=float).ravel()
    elif variant == "G-row-traincol":
        if R is None:
            raise ValueError("G-row-traincol requires R")
        R = np.asarray(R, dtype=float)
        train = list(range(n) if train_origins is None else train_origins)
        col = R[train].sum(axis=0)
        s = col.sum()
        D = (col / s * O.sum()) if s > 0 else np.full(n, O.sum() / n)
    else:
        raise ValueError(variant)

    X = S.copy()
    for _ in range(max_iter):
        # row fit
        rs = X.sum(axis=1)
        X = X * np.divide(O, rs, out=np.zeros_like(O), where=rs > 0)[:, None]
        # col fit
        cs = X.sum(axis=0)
        X = X * np.divide(D, cs, out=np.zeros_like(D), where=cs > 0)[None, :]
        rs = X.sum(axis=1)
        cs = X.sum(axis=0)
        row_err = np.max(np.abs(rs - O) / np.maximum(O, 1e-12))
        col_err = np.max(np.abs(cs - D) / np.maximum(D, 1e-12))
        if max(row_err, col_err) < tol:
            break
    # Condition G scores production-constrained; after IPF rows match O approximately
    # Re-scale rows exactly to O for the assertion contract used elsewhere
    rs = X.sum(axis=1)
    X = np.divide(X, rs[:, None], out=np.zeros_like(X), where=rs[:, None] > 0) * O[:, None]
    assert_production_constrained(X, O, tol=1e-5)
    return X
