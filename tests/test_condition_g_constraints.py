"""O_i^T projection / audit + few-shot mask no-leakage (Ticket 3)."""
from __future__ import annotations

import numpy as np
import pytest

from attrOD.condition_g.constraints import audit_constraints, project_to_O_i_T
from attrOD.condition_g.masks import build_few_shot_masks
from attrOD.models.emission import assert_production_constrained
from attrOD.models.gravity import GravityModel
from attrOD.models.registry import fit_hbw_generator, get_model, list_models


def test_project_and_audit_O_i_T():
    rng = np.random.default_rng(0)
    n = 4
    raw = rng.random((n, n)) * 10
    O = np.array([10.0, 5.0, 7.0, 3.0])
    proj = project_to_O_i_T(raw, O)
    assert proj["ok"]
    assert_production_constrained(proj["T_hat"], O)
    qa = audit_constraints(proj["T_hat"], O, zone_ids=list(range(n)))
    assert qa["ok"]
    assert qa["row_sums_ok"]
    assert qa["non_negative"]


def test_audit_catches_nan_and_row_break():
    O = np.array([1.0, 2.0])
    bad = np.array([[1.0, np.nan], [0.5, 1.5]])
    qa = audit_constraints(bad, O)
    assert not qa["ok"]
    assert qa["n_nan"] == 1


def test_few_shot_masks_exclude_holdout():
    doc = build_few_shot_masks(10, seed=0, holdout_origins=[0, 1, 2])
    for name, m in doc["masks"].items():
        assert m["ok_no_holdout_leak"]
        assert all(e[0] not in {0, 1, 2} for e in m["edges"])
    assert doc["tau"] == 2.0
    assert "1pct_edges" in doc["masks"]


def test_registry_hbw_only_and_oracle_flag():
    assert "Gravity_power" in list_models()
    ora = get_model("Oracle")
    assert ora.is_oracle and ora.is_diagnostic
    assert ora.paper_mainline_eligible is False
    stub = get_model("neuroGravity_stub")
    assert stub.is_stub and stub.paper_mainline_eligible is False

    rng = np.random.default_rng(0)
    n = 5
    R = rng.random((n, n)) + 0.1
    d = rng.random((n, n)) + 0.2
    np.fill_diagonal(d, 0.5)
    m = rng.random(n) + 1
    model = fit_hbw_generator("Gravity_power", R, distances=d, attractiveness=m, train_origins=list(range(n)))
    T_hat = model.emit(R.sum(1))
    assert_production_constrained(T_hat, R.sum(1))


def test_hbw_contract_rejects_T_kw():
    rng = np.random.default_rng(0)
    n = 3
    R = rng.random((n, n)) + 0.1
    d = np.ones((n, n))
    m = np.ones(n)
    with pytest.raises(ValueError, match="HBW-only"):
        fit_hbw_generator(
            "Gravity_power", R, distances=d, attractiveness=m,
            train_origins=[0, 1], T=R,  # sneaky leak
        )
