"""5-row pipe fixture tests for HK Teralytics adapter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attrOD.data.teralytics import (
    MORNING_PART_CANONICAL,
    aggregate_od_from_chunk,
    build_proxy_R_long,
    build_total_long,
    detect_count_column,
    has_all_preaggregates,
    mask_atomic_attribute_rows,
    materialise_hk_teralytics,
    normalize_part_of_day,
    prepare_chunk,
    provisional_distance_matrix,
    read_dbf_zones,
    sha256_file,
)
from attrOD.data.flow_table import long_to_matrix


FIXTURE_HEADER = (
    "StartId|EndId|Date|PartOfDay|TripPurpose|Age|Gender|TripDistance|TripDuration|Count\n"
)


def _write_fixture(path: Path, rows: list[str]) -> Path:
    path.write_text(FIXTURE_HEADER + "\n".join(rows) + "\n")
    return path


def test_detect_count_prefers_count():
    assert detect_count_column(["StartId", "Trips", "Count", "Flow"]) == "Count"
    assert detect_count_column(["StartId", "Flow"]) == "Flow"
    assert detect_count_column(["trips"]) == "trips"


def test_normalize_morning_bin():
    assert normalize_part_of_day("4:00-12:00") == MORNING_PART_CANONICAL
    assert normalize_part_of_day("04:00-12:00") == MORNING_PART_CANONICAL
    assert normalize_part_of_day("morning") == MORNING_PART_CANONICAL


def test_aggregation_weekday_proxy_and_duplicates(tmp_path: Path):
    """5-row fixture: weekday filter, proxy def, deterministic duplicate sum."""
    rows = [
        # weekday Mon 2020-01-27, morning, to-work — contributes to proxy (dup pair)
        "1|2|2020-01-27|4:00-12:00|to-work|25-34|Female|<5 km|0-30 min|3",
        "1|2|2020-01-27|4:00-12:00|to-work|35-44|Male|<5 km|0-30 min|2",
        # weekend — excluded from weekday/proxy
        "1|2|2020-01-25|4:00-12:00|to-work|25-34|Female|<5 km|0-30 min|100",
        # weekday afternoon to-work — total yes, proxy no
        "1|2|2020-01-28|12:00-20:00|to-work|25-34|Male|5 to 10 km|30-60 min|7",
        # weekday morning to-home — total yes, proxy no
        "2|1|2020-01-27|4:00-12:00|to-home|25-34|Female|<5 km|0-30 min|4",
    ]
    csv_path = _write_fixture(tmp_path / "m.csv", rows)
    zones = ["1", "2"]

    total, col, meta = build_total_long(csv_path, zones, chunksize=2, weekdays_only=True)
    assert col == "Count"
    assert meta["total_aggregation"] == "sum_all_rows_matrix_appears_atomic"
    # weekday flows: 3+2+7+4 = 16 (weekend 100 dropped)
    assert total["flow"].sum() == pytest.approx(16.0)

    proxy, _ = build_proxy_R_long(csv_path, zones, chunksize=2)
    # proxy: only first two rows → 5 on 1→2
    assert proxy["flow"].sum() == pytest.approx(5.0)
    mat = long_to_matrix(proxy, zones, partition="R_HBW_proxy")
    assert mat[0, 1] == pytest.approx(5.0)
    assert mat[1, 0] == pytest.approx(0.0)


def test_all_preaggregate_exclusion(tmp_path: Path):
    rows = [
        "1|2|2020-01-27|4:00-12:00|to-work|25-34|Female|<5 km|0-30 min|10",
        "1|2|2020-01-27|4:00-12:00|to-work|All|All|All|All|999",
        "1|2|2020-01-27|4:00-12:00|to-work|35-44|Male|<5 km|0-30 min|5",
        "3|1|2020-01-27|4:00-12:00|to-other|45-54|Female|<5 km|0-30 min|1",
        "1|2|2020-01-26|4:00-12:00|to-work|25-34|Female|<5 km|0-30 min|2",  # Sunday
    ]
    csv_path = _write_fixture(tmp_path / "all.csv", rows)
    total, _, meta = build_total_long(csv_path, ["1", "2", "3"], weekdays_only=True)
    assert meta["saw_all_preaggregates"] is True
    assert "atomic" in meta["total_aggregation"]
    # 10+5+1 = 16; All 999 and Sunday 2 excluded
    assert total["flow"].sum() == pytest.approx(16.0)


def test_unmatched_id_handling(tmp_path: Path):
    rows = [
        "1|2|2020-01-27|4:00-12:00|to-work|25-34|Female|<5 km|0-30 min|1",
        "9|1|2020-01-27|4:00-12:00|to-work|25-34|Female|<5 km|0-30 min|50",
        "1|9|2020-01-27|4:00-12:00|to-work|25-34|Male|<5 km|0-30 min|60",
        "2|1|2020-01-27|12:00-20:00|to-home|35-44|Female|<5 km|0-30 min|3",
        "9|9|2020-01-27|4:00-12:00|to-work|25-34|Female|<5 km|0-30 min|70",
    ]
    csv_path = _write_fixture(tmp_path / "u.csv", rows)
    zone_ids = ["1", "2"]  # 9 unmatched
    total, _, _ = build_total_long(csv_path, zone_ids, weekdays_only=True)
    assert total["flow"].sum() == pytest.approx(4.0)  # 1 + 3
    assert set(total["origin"]).issubset({"1", "2"})
    assert set(total["destination"]).issubset({"1", "2"})


def test_deterministic_duplicate_groupby():
    raw = pd.DataFrame(
        {
            "StartId": ["1", "1", "1"],
            "EndId": ["2", "2", "2"],
            "Date": ["2020-01-27"] * 3,
            "PartOfDay": ["4:00-12:00"] * 3,
            "TripPurpose": ["to-work"] * 3,
            "Age": ["25-34", "35-44", "25-34"],
            "Gender": ["Female", "Male", "Female"],
            "TripDistance": ["<5 km"] * 3,
            "TripDuration": ["0-30 min"] * 3,
            "Count": ["1.5", "2.5", "1.0"],
        }
    )
    df, col = prepare_chunk(raw)
    assert col == "Count"
    g = aggregate_od_from_chunk(
        df, zone_ids=["1", "2"], weekdays_only=True, part_of_day="4:00-12:00", trip_purpose="to-work"
    )
    assert len(g) == 1
    assert g["flow"].iloc[0] == pytest.approx(5.0)


def test_read_dbf_and_distance_smoke():
    dbf = Path("/workspace/hk_samples/CHKU_Shapes.dbf")
    if not dbf.exists():
        pytest.skip("hk_samples DBF not on box")
    z = read_dbf_zones(dbf)
    assert len(z) == 31
    assert "9" not in set(z["fid"])
    assert set(z.columns) >= {"fid", "name", "lon", "lat"}
    d = provisional_distance_matrix(z["lon"], z["lat"])
    assert d.shape == (31, 31)
    assert np.isnan(d[0, 0])
    assert d[0, 1] > 0
    h = sha256_file(dbf)
    assert len(h) == 64


def test_materialise_end_to_end(tmp_path: Path):
    rows = [
        "1|10|2020-01-27|4:00-12:00|to-work|25-34|Female|<5 km|0-30 min|8",
        "10|1|2020-01-27|4:00-12:00|to-home|35-44|Male|<5 km|0-30 min|3",
        "1|10|2020-01-25|4:00-12:00|to-work|25-34|Female|<5 km|0-30 min|99",
        "1|9|2020-01-27|4:00-12:00|to-work|25-34|Female|<5 km|0-30 min|40",
        "1|10|2020-01-28|12:00-20:00|to-other|45-54|Female|5 to 10 km|30-60 min|2",
    ]
    csv_path = _write_fixture(tmp_path / "full.csv", rows)
    shp = tmp_path / "shp"
    shp.mkdir()
    # minimal: copy real dbf for zones 1..32 missing 9
    import shutil

    src = Path("/workspace/hk_samples")
    for ext in (".dbf", ".prj", ".shp"):
        shutil.copy(src / f"CHKU_Shapes{ext}", shp / f"CHKU_Shapes{ext}")
    out = tmp_path / "out"
    result = materialise_hk_teralytics(
        csv_path, shp, out, chunksize=2, write_purpose_period_partitions=False
    )
    assert (out / "zones.csv").exists()
    assert (out / "total.npy").exists()
    assert (out / "R_HBW_proxy.npy").exists()
    assert (out / "distance_provisional.npy").exists()
    assert (out / "qa.json").exists()
    assert (out / "manifest.json").exists()
    qa = result["qa"]
    assert "9" in qa["csv_ids_unmatched"]
    assert qa["count_column"] == "Count"
    assert qa["delimiter"] == "|"
    assert qa["proxy_definition"]["PartOfDay"] == "4:00-12:00"
    assert "strict 07-10" in qa["inclusion_note"]
    assert qa["distance_label"] == "provisional_wgs84_not_metric"
    proxy = np.load(out / "R_HBW_proxy.npy")
    # zone order by numeric fid; 1 then 10 are indices of fids 1 and 10
    zids = qa["zone_ids"]
    i1, i10 = zids.index("1"), zids.index("10")
    assert proxy[i1, i10] == pytest.approx(8.0)
