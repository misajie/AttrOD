"""Condition S partition builders + disjointness audit (Ticket 2)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from attrOD.partitions.builders import (
    MITMA_LENGTH_BANDS,
    PERIOD_BINS_MITMA,
    PURPOSE_COARSE,
    apply_length_band,
    partition_labels_purpose_time,
)

PARTITION_SCHEMA_VERSION = "draft2-condition-s-v1"

# Primary purpose–time targets vs R (draft2 Table 2)
PRIMARY_PURPOSE_TIME = ("other|AM", "home|PM", "home|AM")


def build_partitions(
    *,
    n_zones: int,
    partition_config: Optional[Mapping[str, Any]] = None,
    zone_ids: Optional[Sequence[Any]] = None,
    matrices: Optional[Mapping[str, np.ndarray]] = None,
    distances: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build a versioned partition index.

    *matrices* maps partition label → OD matrix when already materialised.
    Config may list labels under ``purpose_time``, ``length_bands``, ``full``.
    """
    cfg = dict(partition_config or {})
    labels: List[str] = []
    if cfg.get("include_full", True):
        labels.append("full")
    for lab in cfg.get("purpose_time", list(PRIMARY_PURPOSE_TIME)):
        labels.append(str(lab))
    for band in cfg.get("length_bands", []):
        labels.append(f"length|{band}")
    for lab in cfg.get("extra", []):
        labels.append(str(lab))

    # de-dupe preserving order
    seen = set()
    ordered = []
    for lab in labels:
        if lab not in seen:
            seen.add(lab)
            ordered.append(lab)

    index = []
    for i, lab in enumerate(ordered):
        entry: Dict[str, Any] = {
            "id": lab,
            "ordinal": i,
            "kind": lab.split("|", 1)[0] if "|" in lab else lab,
            "n_zones": n_zones,
            "zone_ids": [str(z) for z in zone_ids] if zone_ids is not None else None,
            "has_matrix": bool(matrices and lab in matrices),
        }
        if matrices and lab in matrices:
            M = np.asarray(matrices[lab], dtype=float)
            entry["sum"] = float(M.sum())
            entry["shape"] = list(M.shape)
        if distances is not None and lab.startswith("length|"):
            band = lab.split("|", 1)[1]
            entry["band"] = band
        index.append(entry)

    return {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "n_zones": n_zones,
        "partitions": index,
        "config": cfg,
    }


def audit_partition_disjointness(
    partition_index: Mapping[str, Any],
    *,
    membership: Optional[Mapping[str, Sequence[Any]]] = None,
) -> Dict[str, Any]:
    """Audit coverage / overlap for partitions that claim a zone membership.

    For purpose–time labels that are mutually exclusive cuts of the same flow
    cube, *membership* maps label → iterable of cell-keys or zone-ids. When
    membership is omitted, only structural checks on the index are returned.
    """
    parts = list(partition_index.get("partitions") or [])
    ids = [p["id"] for p in parts]
    empty = [p["id"] for p in parts if p.get("has_matrix") and p.get("sum", 1) == 0]

    overlap_pairs: List[Dict[str, Any]] = []
    coverage: Dict[str, Any] = {}
    if membership:
        universe: set = set()
        for lab, members in membership.items():
            universe |= set(map(str, members))
        for lab, members in membership.items():
            s = set(map(str, members))
            coverage[lab] = {
                "n": len(s),
                "frac_of_universe": (len(s) / len(universe)) if universe else 0.0,
            }
        labs = list(membership.keys())
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                a, b = labs[i], labs[j]
                inter = set(map(str, membership[a])) & set(map(str, membership[b]))
                if inter:
                    overlap_pairs.append(
                        {"a": a, "b": b, "n_overlap": len(inter), "example": next(iter(inter))}
                    )

    ok = len(empty) == 0
    # purpose-time primary trio should not silently overlap if membership given
    primary = [x for x in ids if x in PRIMARY_PURPOSE_TIME]
    if membership and primary:
        ok = ok and not any(
            o["a"] in primary and o["b"] in primary for o in overlap_pairs
        )

    return {
        "ok": ok,
        "n_partitions": len(parts),
        "partition_ids": ids,
        "empty_partitions": empty,
        "coverage": coverage,
        "overlap_pairs": overlap_pairs,
        "schema_version": partition_index.get("schema_version"),
    }


def materialise_length_partitions(
    mat: np.ndarray,
    distances: np.ndarray,
    bands: Sequence[str] = MITMA_LENGTH_BANDS,
) -> Dict[str, np.ndarray]:
    return {f"length|{b}": apply_length_band(mat, distances, b) for b in bands}
