"""Gravity power / exponential with ML β on training-origin outflows of R."""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from .emission import emit_production_constrained

Deterrence = Literal["power", "exp"]


class GravityModel:
    def __init__(
        self,
        deterrence: Deterrence = "power",
        beta_bounds: Tuple[float, float] = (0.1, 3.0),
    ):
        self.deterrence = deterrence
        self.beta_bounds = beta_bounds
        if deterrence == "exp" and beta_bounds == (0.1, 3.0):
            self.beta_bounds = (0.01, 2.0)
        self.beta_: Optional[float] = None
        self.m_: Optional[np.ndarray] = None
        self.d_: Optional[np.ndarray] = None

    def _f(self, d: np.ndarray, beta: float) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        d_safe = np.maximum(d, 1e-12)
        if self.deterrence == "power":
            return d_safe ** (-beta)
        return np.exp(-beta * d)

    def probabilities(self, beta: float, m: np.ndarray, d: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=float).ravel()
        w = m[None, :] * self._f(d, beta)
        # zero self-weight handled by d_ii > 0 from equal-area radius
        row = w.sum(axis=1, keepdims=True)
        return np.divide(w, row, out=np.zeros_like(w), where=row > 0)

    def fit(
        self,
        R: np.ndarray,
        distances: np.ndarray,
        attractiveness: np.ndarray,
        train_origins: Optional[Sequence[int]] = None,
    ) -> "GravityModel":
        R = np.asarray(R, dtype=float)
        d = np.asarray(distances, dtype=float)
        m = np.asarray(attractiveness, dtype=float).ravel()
        n = R.shape[0]
        train = list(range(n)) if train_origins is None else list(train_origins)

        def nll(beta: float) -> float:
            P = self.probabilities(float(beta), m, d)
            # ℓ(β)= sum_{i in train} sum_j R_ij log p_j|i
            ll = 0.0
            for i in train:
                row = R[i]
                if row.sum() <= 0:
                    continue
                p = np.clip(P[i], 1e-300, 1.0)
                ll += float(np.sum(row * np.log(p)))
            return -ll

        res = minimize_scalar(nll, bounds=self.beta_bounds, method="bounded")
        self.beta_ = float(res.x)
        self.m_ = m
        self.d_ = d
        return self

    def predict_P(self) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError("not fitted")
        return self.probabilities(self.beta_, self.m_, self.d_)

    def emit(self, O_T: np.ndarray) -> np.ndarray:
        return emit_production_constrained(O_T, self.predict_P())
