"""Production constraints onto O_i^T (Ticket 3 / draft2)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from attrOD.models.emission import assert_production_constrained, emit_production_constrained


def project_to_O_i_T(
    prediction: np.ndarray,
    O_T: np.ndarray,
    *,
    non_negative: bool = True,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Project an unconstrained OD prediction onto production-constrained domain.

    If *prediction* already encodes destination probabilities (rows ~1), uses
    ``emit_production_constrained``. Otherwise rescales each row to *O_T*.
    """
    P = np.asarray(prediction, dtype=float).copy()
    O = np.asarray(O_T, dtype=float).ravel()
    if P.ndim != 2 or P.shape[0] != O.shape[0]:
        raise ValueError(f"prediction shape {P.shape} incompatible with O_T {O.shape}")

    before = {
        "sum": float(np.nansum(P)),
        "row_sums": P.sum(axis=1).tolist(),
        "n_nan": int(np.isnan(P).sum()),
        "n_inf": int(np.isinf(P).sum()),
        "n_neg": int((P < 0).sum()),
    }

    if non_negative:
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        P = np.maximum(P, 0.0)

    row = P.sum(axis=1)
    # Heuristic: if rows already sum ~1 (probability), emit; else treat as mass and rescale
    if P.shape[0] > 0 and np.nanmax(row) <= 1.0 + 1e-3 and np.nanmean(row) <= 1.0 + 1e-3:
        T_hat = emit_production_constrained(O, P)
        method = "emit_from_probabilities"
    else:
        T_hat = np.zeros_like(P)
        for i in range(P.shape[0]):
            if row[i] > 0 and O[i] > 0:
                T_hat[i] = P[i] * (O[i] / row[i])
            elif O[i] > 0 and row[i] <= 0:
                # uniform fallback for positive outflow with empty row
                T_hat[i] = O[i] / P.shape[1]
        assert_production_constrained(T_hat, O, tol=tol)
        method = "row_rescale_to_O_T"

    after = {
        "sum": float(T_hat.sum()),
        "row_sums": T_hat.sum(axis=1).tolist(),
        "max_abs_row_err": float(np.max(np.abs(T_hat.sum(1) - O))),
    }
    return {
        "T_hat": T_hat,
        "method": method,
        "before": before,
        "after": after,
        "O_T_sum": float(O.sum()),
        "ok": after["max_abs_row_err"] <= tol * max(1.0, float(np.max(np.abs(O)))),
    }


def audit_constraints(
    T_hat: np.ndarray,
    O_T: np.ndarray,
    *,
    zone_ids: Optional[Sequence[Any]] = None,
    tol: float = 1e-6,
    D_T: Optional[np.ndarray] = None,
    check_columns: bool = False,
) -> Dict[str, Any]:
    """Audit non-negativity, row totals, keys, NaN/Inf, duplicates, dimensions."""
    T = np.asarray(T_hat, dtype=float)
    O = np.asarray(O_T, dtype=float).ravel()
    failures: list = []

    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        failures.append(f"T_hat must be square; got {T.shape}")
    if T.shape[0] != O.shape[0]:
        failures.append(f"row dim {T.shape[0]} != len(O_T) {O.shape[0]}")

    n_nan = int(np.isnan(T).sum())
    n_inf = int(np.isinf(T).sum())
    n_neg = int((T < -tol).sum())
    if n_nan:
        failures.append(f"NaN cells: {n_nan}")
    if n_inf:
        failures.append(f"Inf cells: {n_inf}")
    if n_neg:
        failures.append(f"negative cells: {n_neg}")

    row_err = float(np.max(np.abs(T.sum(1) - O))) if T.size else float("nan")
    row_ok = bool(np.allclose(T.sum(1), O, rtol=tol, atol=tol))
    if not row_ok:
        failures.append(f"row-sum violation max|Δ|={row_err}")

    col_ok = True
    col_err = None
    if check_columns and D_T is not None:
        D = np.asarray(D_T, dtype=float).ravel()
        col_err = float(np.max(np.abs(T.sum(0) - D)))
        col_ok = bool(np.allclose(T.sum(0), D, rtol=tol, atol=tol))
        if not col_ok:
            failures.append(f"col-sum violation max|Δ|={col_err}")

    key_ok = True
    if zone_ids is not None:
        if len(zone_ids) != T.shape[0]:
            key_ok = False
            failures.append("zone_ids length mismatch")
        if len(set(map(str, zone_ids))) != len(zone_ids):
            key_ok = False
            failures.append("duplicate zone_ids")

    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "non_negative": n_neg == 0,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_neg": n_neg,
        "row_sums_ok": row_ok,
        "row_err_max": row_err,
        "col_sums_ok": col_ok,
        "col_err_max": col_err,
        "key_integrity_ok": key_ok,
        "shape": list(T.shape),
        "tol": tol,
    }
