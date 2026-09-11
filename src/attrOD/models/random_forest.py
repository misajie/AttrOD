"""Random forest control (mechanism subset only).

For each training origin, destinations sampled with weight R_ij.
Features: log m_i, log m_j, log d_ij, area-normalised OSM counts.
Forest score → softmax over j for each i.
Default: 500 trees, min leaf 20 OD pairs.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .emission import emit_production_constrained


class RandomForestOD:
    def __init__(self, n_estimators: int = 500, min_samples_leaf: int = 20, random_state: int = 0):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_: Optional[RandomForestRegressor] = None
        self._ctx = None

    def _features(self, i, j, m, d, osm):
        feats = [np.log(max(m[i], 1e-12)), np.log(max(m[j], 1e-12)), np.log(max(d[i, j], 1e-12))]
        if osm is not None:
            feats.extend(osm[i].tolist())
            feats.extend(osm[j].tolist())
        return np.asarray(feats, dtype=float)

    def fit(
        self,
        R: np.ndarray,
        distances: np.ndarray,
        attractiveness: np.ndarray,
        osm_features: Optional[np.ndarray] = None,
        train_origins: Optional[Sequence[int]] = None,
        max_samples: int = 200_000,
    ) -> "RandomForestOD":
        R = np.asarray(R, dtype=float)
        d = np.asarray(distances, dtype=float)
        m = np.asarray(attractiveness, dtype=float).ravel()
        n = R.shape[0]
        train = list(range(n) if train_origins is None else train_origins)
        X, y = [], []
        rng = np.random.default_rng(self.random_state)
        for i in train:
            row = R[i]
            if row.sum() <= 0:
                continue
            # sample destinations proportional to flow (with replacement for mass)
            probs = row / row.sum()
            n_samp = max(int(row.sum()), 1)
            # cap per origin
            n_samp = min(n_samp, 2000)
            js = rng.choice(n, size=n_samp, p=probs)
            for j in js:
                X.append(self._features(i, j, m, d, osm_features))
                y.append(np.log1p(row[j]))
            if len(y) >= max_samples:
                break
        if not X:
            raise RuntimeError("no training samples for RF")
        X = np.asarray(X)
        y = np.asarray(y)
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X, y)
        self._ctx = (m, d, osm_features, n)
        return self

    def predict_P(self) -> np.ndarray:
        if self.model_ is None or self._ctx is None:
            raise RuntimeError("not fitted")
        m, d, osm, n = self._ctx
        scores = np.zeros((n, n), dtype=float)
        for i in range(n):
            X = np.stack([self._features(i, j, m, d, osm) for j in range(n)])
            s = self.model_.predict(X)
            e = np.exp(s - np.max(s))
            scores[i] = e / e.sum()
        return scores

    def emit(self, O_T: np.ndarray) -> np.ndarray:
        return emit_production_constrained(O_T, self.predict_P())
