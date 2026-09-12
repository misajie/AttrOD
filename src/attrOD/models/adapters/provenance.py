"""Feature normalisation + third-party provenance helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from attrOD.data.manifest import (
    DEEPGRAVITY_PIN,
    NEUROGRAVITY_PIN,
    record_third_party_provenance,
    write_json,
)

PathLike = Union[str, Path]


def normalize_features(
    features: np.ndarray,
    *,
    feature_names: Optional[Sequence[str]] = None,
    missing: float = 0.0,
    order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Fix feature order, fill missing, return array + manifest rules."""
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(X.shape[1])]
    if order is not None:
        idx = [names.index(n) for n in order]
        X = X[:, idx]
        names = list(order)
    X = np.nan_to_num(X, nan=missing, posinf=missing, neginf=missing)
    rules = {
        "feature_names": names,
        "missing_fill": missing,
        "order_fixed": True,
        "n_features": len(names),
        "train_infer_consistent": True,
    }
    return {"X": X, "rules": rules}


def record_adapter_provenance(
    *,
    model_name: str,
    third_party_root: PathLike,
    out_path: Optional[PathLike] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[str] = None,
    input_hash: Optional[str] = None,
    feature_rules: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
    is_stub: bool = False,
    paper_mainline: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = record_third_party_provenance(third_party_root)
    doc: Dict[str, Any] = {
        "model_name": model_name,
        "third_party": base,
        "pins": {"deepgravity": DEEPGRAVITY_PIN, "neurogravity": NEUROGRAVITY_PIN},
        "seed": seed,
        "checkpoint": checkpoint,
        "input_hash": input_hash,
        "feature_rules": feature_rules or {},
        "backend": backend,
        "is_stub": is_stub,
        "paper_mainline": paper_mainline,
        "formal_main_results": bool(paper_mainline) and not is_stub,
    }
    if is_stub:
        doc["label"] = "stub_or_diagnostic_not_formal_main_results"
    if extra:
        doc["extra"] = extra
    if out_path is not None:
        write_json(out_path, doc)
    return doc
