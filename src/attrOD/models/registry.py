"""Model registry for Condition G generators (Ticket 3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from attrOD.models.closed_form import ClosedFormGravity
from attrOD.models.deep_gravity import DeepGravityModel
from attrOD.models.gravity import GravityModel
from attrOD.models.ipf import ipf_emit
from attrOD.models.neurogravity import MetaGravity, NeuroGravity
from attrOD.models.oracle import OracleGravity
from attrOD.models.radiation import RadiationModel
from attrOD.models.random_forest import RandomForestOD
from attrOD.data.manifest import DEEPGRAVITY_PIN, NEUROGRAVITY_PIN

FitFn = Callable[..., Any]
EmitFn = Callable[..., np.ndarray]


@dataclass
class ModelEntry:
    name: str
    version: str
    kind: str  # law | emission | oracle | deep | fewshot
    fit: Optional[FitFn] = None
    emit: Optional[EmitFn] = None
    requires_torch: bool = False
    requires_osm: bool = False
    paper_mainline_eligible: bool = True
    is_stub: bool = False
    is_oracle: bool = False
    is_diagnostic: bool = False
    hbw_only_fit: bool = True
    resources: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


_REGISTRY: Dict[str, ModelEntry] = {}


def register_model(entry: ModelEntry) -> ModelEntry:
    if entry.name in _REGISTRY:
        raise ValueError(f"model already registered: {entry.name}")
    _REGISTRY[entry.name] = entry
    return entry


def get_model(name: str) -> ModelEntry:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model: {name}; known={list(_REGISTRY)}")
    return _REGISTRY[name]


def list_models(*, include_stub: bool = True, include_oracle: bool = True) -> List[str]:
    names = []
    for n, e in _REGISTRY.items():
        if e.is_stub and not include_stub:
            continue
        if e.is_oracle and not include_oracle:
            continue
        names.append(n)
    return names


def _register_builtins() -> None:
    if _REGISTRY:
        return

    def _fit_grav(det):
        def fit(R, distances, attractiveness, train_origins=None, **_):
            m = GravityModel(deterrence=det)
            m.fit(R, distances, attractiveness, train_origins=train_origins)
            return m
        return fit

    register_model(ModelEntry(
        name="Gravity_power", version="1.0", kind="law",
        fit=_fit_grav("power"),
        emit=lambda model, O_T: model.emit(O_T),
        notes="ML β power deterrence on HBW R train origins",
    ))
    register_model(ModelEntry(
        name="Gravity_exp", version="1.0", kind="law",
        fit=_fit_grav("exp"),
        emit=lambda model, O_T: model.emit(O_T),
    ))
    register_model(ModelEntry(
        name="Radiation", version="1.0", kind="law",
        fit=lambda R, distances, attractiveness, train_origins=None, populations=None, **_:
            RadiationModel().fit(R, distances, populations if populations is not None else attractiveness, train_origins=train_origins),
        emit=lambda model, O_T: model.emit(O_T),
    ))
    register_model(ModelEntry(
        name="RandomForest", version="1.0", kind="law",
        fit=lambda R, distances, attractiveness, train_origins=None, osm_features=None, **_:
            RandomForestOD().fit(R, distances, attractiveness, osm_features=osm_features, train_origins=train_origins),
        emit=lambda model, O_T: model.emit(O_T),
        requires_osm=True,
    ))
    register_model(ModelEntry(
        name="IPF_G-row", version="1.0", kind="emission",
        fit=None,  # stateless emission
        notes="IPF emission rule, not a law",
    ))
    register_model(ModelEntry(
        name="IPF_G-row-traincol", version="1.0", kind="emission",
        fit=None,
        notes="IPF with train-origin column margins from R",
    ))
    register_model(ModelEntry(
        name="Oracle", version="1.0", kind="oracle",
        fit=lambda T, distances, attractiveness, train_origins, **_:
            OracleGravity().fit(T, distances, attractiveness, train_origins=train_origins),
        emit=lambda model, O_T: model.emit(O_T),
        is_oracle=True,
        is_diagnostic=True,
        hbw_only_fit=False,  # intentionally fits on T — upper bound only
        paper_mainline_eligible=False,
        notes="Upper-bound/diagnostic only — not deployable",
    ))
    register_model(ModelEntry(
        name="ClosedFormGravity", version="1.0", kind="law",
        fit=lambda R, distances, attractiveness, train_origins=None, **_:
            ClosedFormGravity().fit(R, distances, attractiveness, train_origins=train_origins),
        emit=lambda model, O_T: model.emit(O_T),
    ))
    register_model(ModelEntry(
        name="meta-Gravity", version="1.0", kind="law",
        fit=lambda R, distances, attractiveness, train_origins=None, populations=None, osm_features=None, **_:
            MetaGravity().fit(
                R, distances,
                populations if populations is not None else attractiveness,
                osm_features if osm_features is not None else np.column_stack([
                    populations if populations is not None else attractiveness,
                    attractiveness,
                ]),
                train_origins=train_origins,
            ),
        emit=lambda model, O_T: model.emit(O_T),
    ))
    register_model(ModelEntry(
        name="DeepGravity", version="1.0", kind="deep",
        fit=lambda R, distances, attractiveness, train_origins=None, populations=None, osm_features=None, **_:
            DeepGravityModel().fit(
                R, distances,
                osm_features if osm_features is not None else np.column_stack([
                    populations if populations is not None else attractiveness,
                    attractiveness,
                ]),
                populations if populations is not None else attractiveness,
                train_origins=train_origins,
            ),
        emit=lambda model, O_T: model.emit(O_T),
        requires_torch=True,
        requires_osm=True,
        resources={"pin": DEEPGRAVITY_PIN},
        notes=f"Package DeepGravityModel; official adapter wraps third_party@{DEEPGRAVITY_PIN}",
    ))
    register_model(ModelEntry(
        name="neuroGravity_stub", version="stub-1.0", kind="fewshot",
        fit=None,
        is_stub=True,
        paper_mainline_eligible=False,
        resources={"pin": NEUROGRAVITY_PIN, "backend": "stub"},
        notes=(
            "Lightweight meta-Gravity + residual MLP stub. "
            "NOT formal edge-enhanced NeuroGravity main results. "
            f"Official backend reserved (commit {NEUROGRAVITY_PIN})."
        ),
    ))
    register_model(ModelEntry(
        name="neuroGravity_official", version="reserved-0.0", kind="fewshot",
        fit=None,
        paper_mainline_eligible=False,  # until official integration decided
        resources={"pin": NEUROGRAVITY_PIN, "backend": "official"},
        notes="Reserved hook for official NeuroGravity; not enabled for main results yet.",
    ))


_register_builtins()


def fit_hbw_generator(
    model_name: str,
    R: np.ndarray,
    *,
    distances: np.ndarray,
    attractiveness: np.ndarray,
    train_origins: Sequence[int],
    populations: Optional[np.ndarray] = None,
    osm_features: Optional[np.ndarray] = None,
    T_for_oracle: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Any:
    """HBW-only fit contract: training mass must come from R (except oracle)."""
    entry = get_model(model_name)
    if entry.hbw_only_fit:
        # soft audit: caller must pass R; we refuse if a 'T' kw sneaks in as training
        if kwargs.get("T") is not None and model_name != "Oracle":
            raise ValueError(
                f"{model_name}: HBW-only fit contract violated — unexpected T in fit kwargs"
            )
    if entry.fit is None:
        raise ValueError(f"{model_name} has no fit() (emission-only or adapter-handled)")
    if entry.is_oracle:
        if T_for_oracle is None:
            raise ValueError("Oracle requires T_for_oracle")
        return entry.fit(
            T_for_oracle, distances, attractiveness, train_origins,
            populations=populations, osm_features=osm_features, **kwargs,
        )
    return entry.fit(
        R, distances, attractiveness,
        train_origins=train_origins,
        populations=populations,
        osm_features=osm_features,
        **kwargs,
    )
