"""NOW-4 run ledger smoke tests (no dataset)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from attrOD.runs.ledger import RunLedger, load_status, new_run_id, stable_input_hash


def test_input_hash_stable():
    a = stable_input_hash({"seed": 0, "x": 1})
    b = stable_input_hash({"x": 1, "seed": 0})
    assert a == b


def test_sample_run_layout(tmp_path: Path):
    # Use a fake server-ish data root name under tmp to avoid Mac path reject.
    data_root = tmp_path / "AttrOD" / "data"
    data_root.mkdir(parents=True)
    out = tmp_path / "outputs"
    run_id = new_run_id(prefix="test")
    ledger = RunLedger(
        run_id=run_id,
        dataset="test",
        data_root=data_root,
        output_root=out,
        seed=7,
        code_root=tmp_path,
        allow_backup_data_root=True,
        distance_status="provisional_wgs84_not_metric",
        missing_fids=["9"],
    )
    status = ledger.execute(sample=True)
    base = out / "runs" / run_id
    assert (base / "run_manifest.json").exists()
    assert (base / "status.json").exists()
    assert (base / "logs").is_dir()
    assert (base / "work" / "sample_marker.json").exists()
    assert status["state"] == "ok"
    assert status["gates"]["paper_mainline"] is False
    assert status["gates"]["verification_only"] is True
    assert status["input_hash"]


def test_failure_retains_stage_hash_error(tmp_path: Path):
    data_root = tmp_path / "AttrOD" / "data"
    data_root.mkdir(parents=True)
    out = tmp_path / "outputs"
    run_id = new_run_id(prefix="fail")
    ledger = RunLedger(
        run_id=run_id,
        dataset="test",
        data_root=data_root,
        output_root=out,
        seed=1,
        code_root=tmp_path,
        allow_backup_data_root=True,
        stage_command="python -c \"raise SystemExit(2)\"",
    )
    with pytest.raises(RuntimeError):
        ledger.execute(sample=False)
    st = load_status(out, run_id)
    assert st["state"] == "failed"
    assert st["stage"] == "command"
    assert st["input_hash"]
    assert st["error"] and st["error"]["message"]
    assert (out / "runs" / run_id / "run_manifest.json").exists()
