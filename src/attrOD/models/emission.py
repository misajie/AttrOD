"""Production-constrained emission: T̂_ij = O_i^T p_j|i.

Row sums match by construction; assertion is mandatory (draft2).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-8


def emit_production_constrained(O: np.ndarray, P: np.ndarray) -> np.ndarray:
    """O: (N,) target outflows; P: (N,N) rows are p_j|i (should sum ~1)."""
    O = np.asarray(O, dtype=float).ravel()
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != O.shape[0] or P.shape[0] != P.shape[1]:
        raise ValueError("P must be N×N and match O")
    # renormalize rows with mass; leave zero-outflow rows as zeros
    row_sum = P.sum(axis=1, keepdims=True)
    P_n = np.divide(P, row_sum, out=np.zeros_like(P), where=row_sum > 0)
    T_hat = O[:, None] * P_n
    assert_production_constrained(T_hat, O)
    return T_hat


def assert_production_constrained(T_hat: np.ndarray, O: np.ndarray, tol: float = 1e-6) -> None:
    O = np.asarray(O, dtype=float).ravel()
    rs = np.asarray(T_hat, dtype=float).sum(axis=1)
    if not np.allclose(rs, O, rtol=tol, atol=tol):
        err = float(np.max(np.abs(rs - O)))
        raise AssertionError(f"production-constrained row sums violated; max |Δ|={err}")
