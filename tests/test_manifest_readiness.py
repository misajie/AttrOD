"""Fixture tests: Mac path rejection + verification_only / HK flags (Ticket 1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from attrOD.data.manifest import (
    DEEPGRAVITY_PIN,
    NEUROGRAVITY_PIN,
    build_run_manifest,
    is_mac_path,
    validate_server_data_root,
)
from attrOD.data.readiness import (
    quarantine_hk_fid9,
    validate_daily_trips_completeness,
    validate_metric_distance,
)
from attrOD.models.adapters.neuro_gravity import NeuroGravityAdapter


def test_mac_paths_rejected():
    assert is_mac_path("/Users/jj/Library/CloudStorage/OneDrive-X/AttrOD/data")
    with pytest.raises(ValueError, match="Mac"):
        validate_server_data_root("/Users/jj/AttrOD/data")
    with pytest.raises(ValueError, match="Mac"):
        validate_server_data_root(
            "/Users/jj/Library/CloudStorage/OneDrive-TheUniversityofHongKong-Connect/ODpros/AttrOD/data/"
        )


def test_workspace_backup_requires_flag(tmp_path):
    # Use real /workspace path if present; else skip role-specific assert
    backup = Path("/workspace/AttrOD/data")
    if not backup.exists():
        pytest.skip("no /workspace/AttrOD/data on this host")
    with pytest.raises(ValueError, match="backup"):
        validate_server_data_root(backup, allow_backup=False)
    info = validate_server_data_root(backup, allow_backup=True)
    assert info["role"] == "backup"
    assert info["ok"]


def test_hk_manifest_forces_verification_only(tmp_path):
    data = tmp_path / "AttrOD" / "data"
    data.mkdir(parents=True)
    # Pretend canonical by naming AttrOD/data under tmp — role may be canonical
    with pytest.raises(ValueError, match="HK"):
        build_run_manifest(
            run_id="x",
            dataset="hk",
            data_root=data,
            output_root=tmp_path / "out",
            verification_only=False,
            paper_mainline=False,
            allow_backup_data_root=True,
        )
    with pytest.raises(ValueError, match="HK"):
        build_run_manifest(
            run_id="x",
            dataset="hk_teralytics",
            data_root=data,
            output_root=tmp_path / "out",
            verification_only=True,
            paper_mainline=True,
            allow_backup_data_root=True,
        )
    man = build_run_manifest(
        run_id="hk_ok",
        dataset="hk",
        data_root=data,
        output_root=tmp_path / "out",
        verification_only=True,
        paper_mainline=False,
        distance_status="provisional_wgs84_not_metric",
        missing_fids=["9"],
        allow_backup_data_root=True,
    )
    assert man["flags"]["verification_only"] is True
    assert man["flags"]["paper_mainline"] is False
    assert man["pins"]["deepgravity"] == DEEPGRAVITY_PIN
    assert man["pins"]["neurogravity"] == NEUROGRAVITY_PIN


def test_provisional_distance_blocks_mainline():
    g = validate_metric_distance(distance_status="provisional_wgs84_not_metric")
    assert g["ok_smoke"]
    assert not g["ok_mainline"]
    assert g["verification_only_required"]


def test_fid9_quarantine_no_silent_fill():
    zones = [str(i) for i in list(range(1, 9)) + list(range(10, 33))]
    rep = quarantine_hk_fid9(zones, missing_fids=["9"])
    assert rep["ok"]
    assert "9" in rep["quarantined_fids"]
    assert "9" not in rep["present_zone_ids"]
    bad = quarantine_hk_fid9(zones + ["9"], missing_fids=["9"])
    assert not bad["ok"]


def test_daily_trips_completeness_mock(tmp_path):
    root = tmp_path / "daily_trips"
    root.mkdir()
    # only 2 of 3 expected
    (root / "20221001_Viajes_distritos.csv.gz").write_bytes(b"x" * 10)
    (root / "20221002_Viajes_distritos.csv.gz").write_bytes(b"x" * 10)
    rep = validate_daily_trips_completeness(
        root, expected_dates=["2022-10-01", "2022-10-02", "2022-10-03"]
    )
    assert not rep["ok"]
    assert "2022-10-03" in rep["missing_dates"]
    (root / "20221003_Viajes_distritos.csv.gz").write_bytes(b"x" * 10)
    rep2 = validate_daily_trips_completeness(
        root, expected_dates=["2022-10-01", "2022-10-02", "2022-10-03"]
    )
    assert rep2["ok"]


def test_neurogravity_stub_not_formal():
    # third_party may exist on box
    root = Path(__file__).resolve().parents[1] / "third_party"
    ng = NeuroGravityAdapter(root if root.exists() else None, backend="stub")
    assert ng.is_stub
    assert ng.formal_main_results is False
    prov = ng.provenance()
    assert prov["is_stub"] is True
    assert prov["formal_main_results"] is False
    assert "STUB" in prov["extra"]["label_warning"]
