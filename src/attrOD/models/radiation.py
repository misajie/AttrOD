"""Radiation law (Simini et al.) — no free parameter."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .emission import emit_production_constrained


class RadiationModel:
    def __init__(self):
        self.P_: Optional[np.ndarray] = None

    def fit(
        self,
        R: np.ndarray,
        distances: np.ndarray,
        mass: np.ndarray,
        train_origins: Optional[Sequence[int]] = None,
    ) -> "RadiationModel":
        R = np.asarray(R, dtype=float)
        d = np.asarray(distances, dtype=float)
        m = np.asarray(mass, dtype=float).ravel()
        n = R.shape[0]
        train = set(range(n) if train_origins is None else train_origins)
        # area-wide intra share of R
        total = R.sum()
        h_area = float(np.trace(R) / total) if total > 0 else 0.0
        P = np.zeros((n, n), dtype=float)
        for i in range(n):
            # s_ij = mass in closed disk radius d_ij excluding i and j
            di = d[i]
            for j in range(n):
                if i == j:
                    continue
                rij = di[j]
                # closed disk: d_ik <= d_ij, k != i,j
                in_disk = (di <= rij) & (np.arange(n) != i) & (np.arange(n) != j)
                s_ij = float(m[in_disk].sum())
                denom = (m[i] + s_ij) * (m[i] + m[j] + s_ij)
                P[i, j] = (m[i] * m[j] / denom) if denom > 0 else 0.0
            # p_ii
            if i in train and R[i].sum() > 0:
                P[i, i] = float(R[i, i] / R[i].sum())
            else:
                P[i, i] = h_area
            # renormalize so row sums to 1 (radiation formula for i≠j is not already normalized with p_ii)
            s = P[i].sum()
            if s > 0:
                P[i] /= s
        self.P_ = P
        return self

    def predict_P(self) -> np.ndarray:
        if self.P_ is None:
            raise RuntimeError("not fitted")
        return self.P_

    def emit(self, O_T: np.ndarray) -> np.ndarray:
        return emit_production_constrained(O_T, self.predict_P())
