"""HBW fit + production-constrained generate via registry (Ticket 3)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from attrOD.condition_g.constraints import audit_constraints, project_to_O_i_T
from attrOD.metrics.core import score_pair
from attrOD.models.ipf import ipf_emit
from attrOD.models.registry import fit_hbw_generator, get_model, list_models


def generate_production_constrained(
    model: Any,
    O_T: np.ndarray,
    *,
    zone_ids: Optional[Sequence[Any]] = None,
    emit_fn=None,
) -> Dict[str, Any]:
    if emit_fn is not None:
        raw = emit_fn(model, O_T)
    elif hasattr(model, "generate_production_constrained"):
        return model.generate_production_constrained(O_T, zone_ids=zone_ids)
    elif hasattr(model, "emit"):
        raw = model.emit(O_T)
    else:
        raise TypeError("model has no emit/generate_production_constrained")
    proj = project_to_O_i_T(raw, O_T)
    qa = audit_constraints(proj["T_hat"], O_T, zone_ids=zone_ids)
    return {"T_hat": proj["T_hat"], "projection": proj, "constraint_qa": qa}


def run_registered_generators(
    R: np.ndarray,
    T: np.ndarray,
    distances: np.ndarray,
    attractiveness: np.ndarray,
    train_origins: Sequence[int],
    *,
    model_names: Optional[Sequence[str]] = None,
    populations: Optional[np.ndarray] = None,
    osm_features: Optional[np.ndarray] = None,
    zone_ids: Optional[Sequence[Any]] = None,
    include_deep_adapters: bool = False,
    third_party_root: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Fit selected registry models on HBW R; emit/audit O_i^T; score vs T."""
    O_T = T.sum(axis=1)
    D_T = T.sum(axis=0)
    names = list(model_names) if model_names is not None else [
        n for n in list_models(include_stub=False, include_oracle=True)
        if n not in ("neuroGravity_stub", "neuroGravity_official", "DeepGravity")
    ]
    # classic defaults for smoke
    default_classic = [
        "Gravity_power", "Gravity_exp", "Radiation", "RandomForest",
        "IPF_G-row", "IPF_G-row-traincol", "Oracle", "ClosedFormGravity", "meta-Gravity",
    ]
    if model_names is None:
        names = [n for n in default_classic if n in list_models(include_oracle=True)]

    results: Dict[str, Dict[str, Any]] = {}
    pop = populations if populations is not None else attractiveness
    feats = osm_features

    for name in names:
        entry = get_model(name)
        meta: Dict[str, Any] = {
            "model": name,
            "is_oracle": entry.is_oracle,
            "is_stub": entry.is_stub,
            "is_diagnostic": entry.is_diagnostic,
            "paper_mainline_eligible": entry.paper_mainline_eligible,
            "hbw_only_fit": entry.hbw_only_fit,
        }
        if name.startswith("IPF_"):
            variant = "G-row" if name.endswith("G-row") else "G-row-traincol"
            T_hat = ipf_emit(
                R, O_T, D_T=D_T, R=R, train_origins=train_origins, variant=variant
            )
            qa = audit_constraints(T_hat, O_T, zone_ids=zone_ids)
            scores = score_pair(T_hat, T, distances)
            results[name] = {
                "T_hat": T_hat,
                "metrics": scores,
                "constraint_qa": qa,
                "meta": meta,
            }
            continue

        model = fit_hbw_generator(
            name,
            R,
            distances=distances,
            attractiveness=attractiveness,
            train_origins=train_origins,
            populations=pop,
            osm_features=feats,
            T_for_oracle=T if entry.is_oracle else None,
        )
        packed = generate_production_constrained(model, O_T, zone_ids=zone_ids)
        scores = score_pair(packed["T_hat"], T, distances)
        results[name] = {
            "T_hat": packed["T_hat"],
            "metrics": scores,
            "constraint_qa": packed["constraint_qa"],
            "meta": meta,
        }

    if include_deep_adapters:
        from attrOD.models.adapters.deep_gravity import DeepGravityAdapter
        from attrOD.models.adapters.neuro_gravity import NeuroGravityAdapter
        dg = DeepGravityAdapter(third_party_root=third_party_root, seed=seed)
        feat_mat = feats if feats is not None else np.column_stack([pop, attractiveness])
        dg.fit(R, distances, feat_mat, pop, train_origins=train_origins)
        packed = dg.generate_production_constrained(O_T, zone_ids=zone_ids)
        results["DeepGravityAdapter"] = {
            "T_hat": packed["T_hat"],
            "metrics": score_pair(packed["T_hat"], T, distances),
            "constraint_qa": packed["constraint_qa"],
            "meta": {"model": "DeepGravityAdapter", "pin": dg.PIN, "is_stub": False},
            "provenance": dg.provenance(),
        }
        ng = NeuroGravityAdapter(
            third_party_root=third_party_root, backend="stub", seed=seed
        )
        ng.fit_zeroshot(R, distances, pop, feat_mat, train_origins)
        packed = ng.generate_production_constrained(O_T, zone_ids=zone_ids)
        results["neuroGravity_stub"] = {
            "T_hat": packed["T_hat"],
            "metrics": score_pair(packed["T_hat"], T, distances),
            "constraint_qa": packed["constraint_qa"],
            "meta": {
                "model": "neuroGravity_stub",
                "pin": ng.PIN,
                "is_stub": True,
                "formal_main_results": False,
                "paper_mainline_eligible": False,
            },
            "provenance": ng.provenance(),
        }

    return results


def predictions_to_frame(T_hat: np.ndarray, zone_ids: Optional[Sequence[Any]] = None) -> pd.DataFrame:
    n = T_hat.shape[0]
    z = list(zone_ids) if zone_ids is not None else list(range(n))
    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({
                "origin": z[i],
                "destination": z[j],
                "flow": float(T_hat[i, j]),
            })
    return pd.DataFrame(rows)
