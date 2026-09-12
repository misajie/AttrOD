"""Metric distance gate (plan: spatial_blocks.distance_gate.validate_metric_distance)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from attrOD.data.readiness import PROVISIONAL_DISTANCE, validate_metric_distance

__all__ = ["validate_metric_distance", "PROVISIONAL_DISTANCE", "gate_or_raise"]


def gate_or_raise(
    *,
    distance_status: Optional[str] = None,
    distances: Optional[np.ndarray] = None,
    crs: Optional[str] = None,
    require_mainline: bool = False,
) -> Dict[str, Any]:
    report = validate_metric_distance(
        distance_status=distance_status,
        distances=distances,
        crs=crs,
    )
    if require_mainline and not report["ok_mainline"]:
        raise RuntimeError(
            f"distance gate blocked paper_mainline: {report['notes']} "
            f"(status={report['distance_status']})"
        )
    if not report["ok_smoke"]:
        raise RuntimeError(f"distance gate blocked even smoke: {report}")
    return report
