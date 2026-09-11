"""Condition G pipeline (draft2).

Train on origin rows of R in the training zone set.
Emit T̂_ij = O_i^T p_j|i (row sums match by construction).
Scores on (T̂, T) with no extra λ.
IPF two column-margin variants: G-row, G-row-traincol.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from attrOD.metrics.core import score_pair
from attrOD.models.closed_form import ClosedFormGravity
from attrOD.models.deep_gravity import DeepGravityModel
from attrOD.models.emission import assert_production_constrained
from attrOD.models.gravity import GravityModel
from attrOD.models.ipf import ipf_emit
from attrOD.models.neurogravity import MetaGravity, NeuroGravity
from attrOD.models.oracle import OracleGravity
from attrOD.models.radiation import RadiationModel
from attrOD.models.random_forest import RandomForestOD


def score_condition_g(
    T_hat: np.ndarray,
    T: np.ndarray,
    O_T: np.ndarray,
    distances: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    assert_production_constrained(T_hat, O_T)
    return score_pair(T_hat, T, distances)


def run_generators(
    R: np.ndarray,
    T: np.ndarray,
    distances: np.ndarray,
    attractiveness: np.ndarray,
    train_origins: Sequence[int],
    *,
    populations: Optional[np.ndarray] = None,
    osm_features: Optional[np.ndarray] = None,
    mechanism: bool = False,
    fewshot: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Fit named generators on R train origins; emit with O_i^T; score vs T.

    Always runs: gravity power/exp, radiation, IPF (both variants), meta-Gravity, oracle.
    Mechanism subset additionally: Deep Gravity, RF, closed-form, neuroGravity modes.
    """
    O_T = T.sum(axis=1)
    D_T = T.sum(axis=0)
    pop = populations if populations is not None else attractiveness
    feats = osm_features if osm_features is not None else np.column_stack([pop, attractiveness])
    results: Dict[str, Dict[str, float]] = {}

    def _store(name: str, T_hat: np.ndarray) -> None:
        results[name] = score_condition_g(T_hat, T, O_T, distances)

    # Gravity power / exp
    for det, name in [("power", "Gravity_power"), ("exp", "Gravity_exp")]:
        g = GravityModel(deterrence=det)
        g.fit(R, distances, attractiveness, train_origins=train_origins)
        _store(name, g.emit(O_T))

    rad = RadiationModel().fit(R, distances, pop, train_origins=train_origins)
    _store("Radiation", rad.emit(O_T))

    # IPF
    for variant in ("G-row", "G-row-traincol"):
        T_hat = ipf_emit(
            R,
            O_T,
            D_T=D_T,
            R=R,
            train_origins=train_origins,
            variant=variant,
        )
        _store(f"IPF_{variant}", T_hat)

    meta = MetaGravity().fit(R, distances, pop, feats, train_origins=train_origins)
    _store("meta-Gravity", meta.emit(O_T))

    ora = OracleGravity().fit(T, distances, attractiveness, train_origins=train_origins)
    _store("Oracle", ora.emit(O_T))

    if mechanism:
        dg = DeepGravityModel()
        dg.fit(R, distances, feats, pop, train_origins=train_origins)
        _store("DeepGravity", dg.emit(O_T))

        cf = ClosedFormGravity().fit(R, distances, attractiveness, train_origins=train_origins)
        _store("ClosedFormGravity", cf.emit(O_T))

        rf = RandomForestOD()
        rf.fit(R, distances, attractiveness, osm_features=feats, train_origins=train_origins)
        _store("RandomForest", rf.emit(O_T))

        ng = NeuroGravity()
        ng.fit_zeroshot(R, distances, pop, feats, train_origins=train_origins)
        _store("neuroGravity_zeroshot", ng.emit(O_T))

        if fewshot:
            holdout = [i for i in range(R.shape[0]) if i not in set(train_origins)]
            for mask in ("1pct_edges", "10pct_edges", "10pct_zones"):
                ng2 = NeuroGravity()
                ng2.fit_fewshot(
                    R, T, distances, pop, feats, train_origins,
                    mask=mask, holdout_origins=holdout,
                )
                _store(f"neuroGravity_{mask}", ng2.emit(O_T))

    return results
