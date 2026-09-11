"""Synthetic N≈5 unit tests for metrics + production-constrained assertion."""
from __future__ import annotations

import numpy as np
import pytest

from attrOD.metrics import cpc, cpl, cpcd, delta_cpc, independence_table, jsd, nrmse, pearson_log, r2_log
from attrOD.models.emission import assert_production_constrained, emit_production_constrained
from attrOD.models.gravity import GravityModel
from attrOD.condition_s import score_condition_s


def _synth(n=5, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.random((n, n)) * 10
    T = R * 0.5 + rng.random((n, n))
    xy = rng.random((n, 2)) * 10
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.sqrt(1.0 / np.pi))  # unit area
    m = rng.random(n) * 100 + 1
    return T, R, d, m


def test_cpc_identical_is_one():
    T, R, d, m = _synth()
    assert cpc(T, T) == pytest.approx(1.0)


def test_cpc_formula():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 1.0], [0.0, 0.0]])
    # 2*(min11 + min12 + min21 + min22) / (2+2) = 2*(1+0+0+0)/4 = 0.5
    assert cpc(A, B) == pytest.approx(0.5)


def test_delta_cpc_transpose():
    T = np.array([[0.0, 5.0], [1.0, 0.0]])
    R = np.array([[0.0, 1.0], [5.0, 0.0]])  # closer to T^T? actually R ~ T.T scaled
    # R is roughly T.T
    lam, c_r, c_rt, delta = delta_cpc(T, R)
    assert lam == pytest.approx(T.sum() / R.sum())
    assert c_rt >= c_r - 1e-9
    assert delta == pytest.approx(c_rt - c_r)


def test_companions_finite():
    T, R, d, m = _synth()
    lam = T.sum() / R.sum()
    Rd = lam * R
    assert np.isfinite(cpl(T, Rd))
    assert np.isfinite(cpcd(T, Rd, d))
    assert np.isfinite(pearson_log(T, Rd))
    assert np.isfinite(r2_log(T, Rd))
    assert np.isfinite(nrmse(T, Rd))
    assert np.isfinite(jsd(T, Rd))


def test_independence_table_margins():
    T, _, _, _ = _synth()
    ind = independence_table(T)
    assert ind.sum() == pytest.approx(T.sum())
    np.testing.assert_allclose(ind.sum(1), T.sum(1), atol=1e-8)
    np.testing.assert_allclose(ind.sum(0), T.sum(0), atol=1e-8)


def test_production_constrained_assertion():
    T, R, d, m = _synth()
    g = GravityModel(deterrence="power")
    g.fit(R, d, m)
    O = T.sum(1)
    T_hat = g.emit(O)
    assert_production_constrained(T_hat, O)
    np.testing.assert_allclose(T_hat.sum(1), O, atol=1e-8)
    # broken matrix must raise
    bad = T_hat.copy()
    bad[0, 0] += 1.0
    with pytest.raises(AssertionError):
        assert_production_constrained(bad, O)


def test_emit_helper():
    P = np.array([[0.5, 0.5], [0.2, 0.8]])
    O = np.array([10.0, 5.0])
    T_hat = emit_production_constrained(O, P)
    assert T_hat.sum() == pytest.approx(15.0)


def test_condition_s_smoke():
    T, R, d, m = _synth()
    s = score_condition_s(T, R, d)
    assert "CPC_R" in s and "Delta" in s and "lambda" in s
    assert s["lambda"] == pytest.approx(T.sum() / R.sum())
