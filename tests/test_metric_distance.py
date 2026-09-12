"""P1 metric distance unit tests (no geodata files)."""
from __future__ import annotations

import math

import numpy as np

from attrOD.spatial.metric_distance import (
    METRIC_DISTANCE,
    build_metric_distance_matrix,
    intra_zonal_distance,
)


def test_intra_zonal_pi():
    assert abs(intra_zonal_distance(math.pi) - 1.0) < 1e-9


def test_matrix_diag_and_status():
    xy = np.array([[0.0, 0.0], [3000.0, 0.0], [0.0, 4000.0]], dtype=float)
    areas = np.array([math.pi * 100.0**2, math.pi * 50.0**2, math.pi * 25.0**2])
    D, qa = build_metric_distance_matrix(
        xy, areas, crs="EPSG:25830", zone_ids=["a", "b", "c"], study_area="mitma"
    )
    assert qa["distance_status"] == METRIC_DISTANCE
    assert qa["d_ii_formula_confirmed"] is True
    assert qa["ok_mainline"] is True
    assert abs(D[0, 1] - 3000.0) < 1e-6
    assert abs(D[0, 2] - 4000.0) < 1e-6
    assert abs(D[0, 0] - 100.0) < 1e-6
    assert abs(D[1, 1] - 50.0) < 1e-6
    assert qa["summary"]["units"] == "meters"
