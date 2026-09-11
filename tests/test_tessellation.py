from __future__ import annotations

import numpy as np
import pytest

from attrOD.tessellation import inclusion_ok, intra_zonal_distance


def test_dii():
    assert intra_zonal_distance(np.pi) == pytest.approx(1.0)


def test_inclusion():
    n = 30
    mat = np.ones((n, n))
    np.fill_diagonal(mat, 0.0)
    ok, info = inclusion_ok(mat)
    assert ok
    assert info["N"] == 30
