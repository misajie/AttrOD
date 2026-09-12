"""DeepGravity / NeuroGravity adapter dry-run without real HK 1.2G."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from attrOD.data.manifest import DEEPGRAVITY_PIN, NEUROGRAVITY_PIN
from attrOD.models.adapters.deep_gravity import DeepGravityAdapter
from attrOD.models.adapters.neuro_gravity import NeuroGravityAdapter


def test_deepgravity_schema_roundtrip_and_emit():
    root = Path(__file__).resolve().parents[1] / "third_party"
    dg = DeepGravityAdapter(root if root.exists() else None, seed=0)
    assert dg.PIN == DEEPGRAVITY_PIN
    n = 5
    rng = np.random.default_rng(0)
    R = rng.random((n, n)) + 0.1
    d = rng.random((n, n)) + 0.2
    np.fill_diagonal(d, 0.5)
    pop = rng.random(n) + 1
    feats = rng.random((n, 2))
    rt = dg.schema_roundtrip(R, feats, pop, d)
    assert rt["roundtrip_ok"]
    dg.fit(R, d, feats, pop, train_origins=list(range(n)))
    packed = dg.generate_production_constrained(R.sum(1))
    assert packed["constraint_qa"]["ok"]


def test_neurogravity_stub_fit_emit():
    root = Path(__file__).resolve().parents[1] / "third_party"
    ng = NeuroGravityAdapter(root if root.exists() else None, backend="stub", seed=0)
    assert ng.PIN == NEUROGRAVITY_PIN
    n = 6
    rng = np.random.default_rng(1)
    R = rng.random((n, n)) + 0.2
    T = rng.random((n, n)) + 0.2
    d = rng.random((n, n)) + 0.3
    np.fill_diagonal(d, 0.4)
    pop = rng.random(n) + 1
    feats = rng.random((n, 3))
    ng.fit_zeroshot(R, d, pop, feats, train_origins=list(range(n - 1)))
    packed = ng.generate_production_constrained(T.sum(1))
    assert packed["constraint_qa"]["ok"]
    assert packed["constraint_qa"]["formal_main_results"] is False
