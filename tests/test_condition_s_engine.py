"""Tiny synthetic matrices: λ / Δ / null / day bootstrap (Ticket 2)."""
from __future__ import annotations

import numpy as np
import pytest

from attrOD.condition_s import (
    compute_delta,
    compute_lambda_r,
    compute_lambda_rt,
    day_bootstrap,
    evaluate_independence_null,
    run_condition_s_full,
    score_condition_s,
)
from attrOD.metrics.core import cpc, delta_cpc
from attrOD.partitions.condition_s import audit_partition_disjointness, build_partitions


def _mats(n=5, seed=1):
    rng = np.random.default_rng(seed)
    R = rng.random((n, n)) * 10 + 0.1
    T = R.T * 0.8 + rng.random((n, n)) * 0.5  # somewhat transposed
    return T, R


def test_compute_lambda_r_matches_delta_cpc():
    T, R = _mats()
    lr = compute_lambda_r(T, R)
    lam, c_r, _, _ = delta_cpc(T, R)
    assert lr["lambda"] == pytest.approx(lam)
    assert lr["CPC"] == pytest.approx(c_r)
    assert lr["R_down"].sum() == pytest.approx(T.sum())


def test_compute_lambda_rt_and_delta_definition():
    T, R = _mats()
    lrt = compute_lambda_rt(T, R)
    d = compute_delta(T, R)
    assert d["Delta"] == pytest.approx(d["CPC_RT"] - d["CPC_R"])
    assert d["definition"].startswith("Delta = CPC")
    assert lrt["CPC"] == pytest.approx(d["CPC_RT"])
    # For this synth (T closer to R.T), Delta should tend positive
    assert d["Delta"] > -1e-9


def test_independence_null():
    T, _ = _mats()
    null = evaluate_independence_null(T)
    assert null["row_sums_match"]
    assert null["col_sums_match"]
    assert null["total_match"]
    assert 0.0 <= null["CPC_indep"] <= 1.0 + 1e-9


def test_day_bootstrap_preserves_within_day_and_reproducible():
    T, R = _mats()
    T_days = [T * (0.4 + 0.1 * i) for i in range(3)]
    R_days = [R * (0.4 + 0.1 * i) for i in range(3)]
    a = day_bootstrap(T_days, R_days, n_resamples=20, seed=42)
    b = day_bootstrap(T_days, R_days, n_resamples=20, seed=42)
    assert a["within_day_od_preserved"] is True
    assert a["unit"] == "day"
    assert a["bootstrap_summary"]["mean_Delta"] == pytest.approx(
        b["bootstrap_summary"]["mean_Delta"]
    )
    assert "p2.5_Delta" in a["bootstrap_summary"]
    assert len(a["bootstrap_samples"]) == 20
    # day indices recorded for audit
    assert "day_indices" in a["bootstrap_samples"][0]


def test_day_bootstrap_requires_d2():
    T, R = _mats()
    with pytest.raises(ValueError):
        day_bootstrap([T], [R], n_resamples=5, seed=0)


def test_run_condition_s_full_and_score_compat():
    T, R = _mats()
    s = score_condition_s(T, R)
    full = run_condition_s_full(T, R, study_area="synth", partition="full")
    assert s["lambda"] == pytest.approx(full["metrics"]["lambda"])
    assert "CPC_indep" in full["metrics_row"] or "CPC_indep" in full["metrics"]


def test_partition_builders_audit():
    idx = build_partitions(
        n_zones=5,
        partition_config={"include_full": True, "purpose_time": ["other|AM", "home|PM"]},
        matrices={"full": np.ones((5, 5))},
    )
    audit = audit_partition_disjointness(idx)
    assert audit["ok"]
    assert "full" in audit["partition_ids"]
