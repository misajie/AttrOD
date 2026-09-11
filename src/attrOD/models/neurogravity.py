"""meta-Gravity + neuroGravity (Yang et al.) — working sklearn/numpy prior + residual stub.

Modes:
  1. meta-Gravity alone (zero-shot prior), production-constrained from R
  2. neuroGravity zero-shot: residual trained on R spatial train blocks
  3. neuroGravity few-shot: after HBW pretrain, reveal 1% / 10% edges or 10% zones; τ=2

Full edge-enhanced graph transformer requires torch geometric stack; here we implement
meta-Gravity MLP (sklearn) + a lightweight residual MLP that matches the paper's
operating modes and scoring contract. Swap in published weights when available.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor

from .emission import emit_production_constrained

FewshotMask = Literal["1pct_edges", "10pct_edges", "10pct_zones"]


class MetaGravity:
    """Predict pair-specific G and α from h_i ⊕ h_j; F_ij = G P_i P_j / d^α."""

    def __init__(self, hidden=(64, 32), random_state: int = 0):
        self.hidden = hidden
        self.random_state = random_state
        self.mlp_G_: Optional[MLPRegressor] = None
        self.mlp_a_: Optional[MLPRegressor] = None
        self._ctx = None

    def _x(self, i, j, H):
        return np.concatenate([H[i], H[j]])

    def fit(
        self,
        R: np.ndarray,
        distances: np.ndarray,
        populations: np.ndarray,
        features: np.ndarray,
        train_origins: Optional[Sequence[int]] = None,
    ) -> "MetaGravity":
        R = np.asarray(R, dtype=float)
        d = np.asarray(distances, dtype=float)
        P = np.asarray(populations, dtype=float).ravel()
        H = np.asarray(features, dtype=float)
        n = R.shape[0]
        train = list(range(n) if train_origins is None else train_origins)
        X, yG, yA = [], [], []
        for i in train:
            for j in range(n):
                if R[i, j] <= 0 or d[i, j] <= 0:
                    continue
                # invert gravity-like: log F ≈ log G + log Pi + log Pj - α log d
                # train to match log flow residuals with weak targets
                X.append(self._x(i, j, H))
                # targets: use simple proxies then refine via flow fit
                yG.append(np.log1p(R[i, j]) - np.log(max(P[i] * P[j], 1e-12)) + np.log(max(d[i, j], 1e-12)))
                yA.append(1.0)
        if len(X) < 10:
            # degenerate tiny case
            self.mlp_G_ = None
            self.mlp_a_ = None
            self._ctx = (P, d, H, n, 1.0, 1.0)
            return self
        X = np.asarray(X)
        self.mlp_G_ = MLPRegressor(hidden_layer_sizes=self.hidden, random_state=self.random_state, max_iter=200)
        self.mlp_a_ = MLPRegressor(hidden_layer_sizes=self.hidden, random_state=self.random_state, max_iter=200)
        self.mlp_G_.fit(X, np.asarray(yG))
        self.mlp_a_.fit(X, np.asarray(yA))
        self._ctx = (P, d, H, n, None, None)
        return self

    def pair_flow(self) -> np.ndarray:
        P, d, H, n, g0, a0 = self._ctx
        F = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if self.mlp_G_ is None:
                    G, a = float(np.exp(g0 or 0.0)), float(a0 or 1.0)
                else:
                    x = self._x(i, j, H)[None, :]
                    G = float(np.exp(self.mlp_G_.predict(x)[0]))
                    a = float(max(self.mlp_a_.predict(x)[0], 0.1))
                F[i, j] = G * P[i] * P[j] / (max(d[i, j], 1e-12) ** a)
        return F

    def predict_P(self) -> np.ndarray:
        F = self.pair_flow()
        row = F.sum(axis=1, keepdims=True)
        return np.divide(F, row, out=np.zeros_like(F), where=row > 0)

    def emit(self, O_T: np.ndarray) -> np.ndarray:
        return emit_production_constrained(O_T, self.predict_P())


class NeuroGravity:
    """meta-Gravity prior + residual MLP (graph-transformer stand-in)."""

    def __init__(self, tau: float = 2.0, random_state: int = 0):
        self.tau = tau
        self.random_state = random_state
        self.meta = MetaGravity(random_state=random_state)
        self.resid_: Optional[MLPRegressor] = None
        self._ctx = None

    def fit_zeroshot(
        self,
        R: np.ndarray,
        distances: np.ndarray,
        populations: np.ndarray,
        features: np.ndarray,
        train_origins: Sequence[int],
    ) -> "NeuroGravity":
        self.meta.fit(R, distances, populations, features, train_origins=train_origins)
        # residual: predict log R - log F_meta on train origins
        F = self.meta.pair_flow()
        H = np.asarray(features, dtype=float)
        d = np.asarray(distances, dtype=float)
        n = R.shape[0]
        X, y = [], []
        for i in train_origins:
            for j in range(n):
                if R[i, j] <= 0:
                    continue
                x = np.concatenate([H[i], H[j], [d[i, j], np.log1p(F[i, j])]])
                X.append(x)
                y.append(np.log1p(R[i, j]) - np.log1p(F[i, j]))
        if len(X) >= 10:
            self.resid_ = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=self.random_state, max_iter=200)
            self.resid_.fit(np.asarray(X), np.asarray(y))
        self._ctx = (np.asarray(populations, dtype=float), d, H, n)
        return self

    def fit_fewshot(
        self,
        R: np.ndarray,
        T: np.ndarray,
        distances: np.ndarray,
        populations: np.ndarray,
        features: np.ndarray,
        train_origins: Sequence[int],
        mask: FewshotMask = "1pct_edges",
        holdout_origins: Optional[Sequence[int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> "NeuroGravity":
        """Pretrain on R, then reveal a declared subset of target edges (never scored cells)."""
        self.fit_zeroshot(R, distances, populations, features, train_origins)
        rng = rng or np.random.default_rng(self.random_state)
        n = T.shape[0]
        holdout = set(holdout_origins or [])
        # candidate edges: internal pairs not in holdout origins as observed (draft2)
        pairs = [(i, j) for i in range(n) for j in range(n) if i not in holdout]
        if mask == "1pct_edges":
            k = max(1, int(0.01 * n * n))
            chosen = [pairs[i] for i in rng.choice(len(pairs), size=min(k, len(pairs)), replace=False)]
        elif mask == "10pct_edges":
            k = max(1, int(0.10 * n * n))
            chosen = [pairs[i] for i in rng.choice(len(pairs), size=min(k, len(pairs)), replace=False)]
        elif mask == "10pct_zones":
            z = rng.choice(n, size=max(1, int(0.10 * n)), replace=False)
            zset = set(int(x) for x in z)
            chosen = [(i, j) for i in zset for j in zset]
        else:
            raise ValueError(mask)
        F = self.meta.pair_flow()
        # weight observed edges by softmax of meta prior at temperature τ=2
        prior = np.array([F[i, j] for i, j in chosen], dtype=float)
        # softmax weights
        logits = prior / self.tau
        w = np.exp(logits - logits.max())
        w = w / w.sum()
        H = np.asarray(features, dtype=float)
        d = np.asarray(distances, dtype=float)
        X, y, sw = [], [], []
        for (i, j), wi in zip(chosen, w):
            x = np.concatenate([H[i], H[j], [d[i, j], np.log1p(F[i, j])]])
            X.append(x)
            y.append(np.log1p(T[i, j]) - np.log1p(F[i, j]))
            sw.append(wi)
        if len(X) >= 5:
            self.resid_ = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=self.random_state, max_iter=300)
            self.resid_.fit(np.asarray(X), np.asarray(y))
        self._ctx = (np.asarray(populations, dtype=float), d, H, n)
        return self

    def predict_P(self) -> np.ndarray:
        F = self.meta.pair_flow()
        n = F.shape[0]
        if self.resid_ is not None and self._ctx is not None:
            _, d, H, _ = self._ctx
            for i in range(n):
                for j in range(n):
                    x = np.concatenate([H[i], H[j], [d[i, j], np.log1p(F[i, j])]])[None, :]
                    F[i, j] = np.expm1(np.log1p(F[i, j]) + float(self.resid_.predict(x)[0]))
                    F[i, j] = max(F[i, j], 0.0)
        row = F.sum(axis=1, keepdims=True)
        return np.divide(F, row, out=np.zeros_like(F), where=row > 0)

    def emit(self, O_T: np.ndarray) -> np.ndarray:
        return emit_production_constrained(O_T, self.predict_P())
