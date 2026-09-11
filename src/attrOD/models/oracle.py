"""Oracle: refit gravity power (and optionally DG) on target partition train origins."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .gravity import GravityModel


class OracleGravity:
    """Measures how much of the G gap is the wrong kernel vs unpredictable destinations."""

    def __init__(self):
        self.model = GravityModel(deterrence="power")

    def fit(
        self,
        T: np.ndarray,
        distances: np.ndarray,
        attractiveness: np.ndarray,
        train_origins: Sequence[int],
    ) -> "OracleGravity":
        self.model.fit(T, distances, attractiveness, train_origins=train_origins)
        return self

    def emit(self, O_T: np.ndarray) -> np.ndarray:
        return self.model.emit(O_T)
