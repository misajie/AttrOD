"""Closed-form gravity-like law (Cabanas-Tirapu et al.) — documented hook.

The published closed form (or a form re-discovered on training origins of R) is run
production-constrained as an extra gravity-family baseline. Exact symbolic
expression is unpublished in this repo; plug in via `deterrence_fn` or
`expression="published"` once encoded.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .emission import emit_production_constrained


class ClosedFormGravity:
    """Stub / hook for Cabanas closed-form gravity-like expression.

    Default fallback: power-law with β=1 (documented placeholder until expression wired).
    """

    def __init__(self, deterrence_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.deterrence_fn = deterrence_fn
        self.P_: Optional[np.ndarray] = None
        self.expression_note = (
            "Hook for Cabanas-Tirapu et al. closed-form; default placeholder f(d)=1/d."
        )

    def fit(
        self,
        R: np.ndarray,
        distances: np.ndarray,
        attractiveness: np.ndarray,
        train_origins: Optional[Sequence[int]] = None,
    ) -> "ClosedFormGravity":
        m = np.asarray(attractiveness, dtype=float).ravel()
        d = np.asarray(distances, dtype=float)
        if self.deterrence_fn is not None:
            f = self.deterrence_fn(d)
        else:
            f = 1.0 / np.maximum(d, 1e-12)
        w = m[None, :] * f
        row = w.sum(axis=1, keepdims=True)
        self.P_ = np.divide(w, row, out=np.zeros_like(w), where=row > 0)
        return self

    def predict_P(self) -> np.ndarray:
        if self.P_ is None:
            raise RuntimeError("not fitted")
        return self.P_

    def emit(self, O_T: np.ndarray) -> np.ndarray:
        return emit_production_constrained(O_T, self.predict_P())
