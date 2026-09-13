"""Synthetic smoke for P5-MAT Lombardy materialise (no geopandas required)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from attrOD.data.lombardy_materialise import (
    build_partition_long,
    densify_parts,
    write_lombardy_freeze,
)
from attrOD.spatial.metric_distance import build_metric_distance_matrix


def test_lombardy_materialise_synthetic():
    n = 25
    zone_ids = [f"Z{i:02d}" for i in range(n)]
    rows = []
    rng = np.random.default_rng(0)
    for i, o in enumerate(zone_ids):
        for j, d in enumerate(zone_ids):
            if i == j:
                continue
            # morning lavoro
            rows.append(
                {
                    "origin": o,
                    "destination": d,
                    "flow": float(rng.integers(1, 5)),
                    "motive": "lavoro",
                    "FASCIA_ORARIA": "mattina",
                }
            )
            # studio must not enter R
            rows.append(
                {
                    "origin": o,
                    "destination": d,
                    "flow": 10.0,
                    "motive": "studio",
                    "FASCIA_ORARIA": "mattina",
                }
            )
            # home PM
            rows.append(
                {
                    "origin": o,
                    "destination": d,
                    "flow": float(rng.integers(1, 3)),
                    "motive": "rientri a casa",
                    "FASCIA_ORARIA": "pomeriggio",
                }
            )
    raw = pd.DataFrame(rows)
    parts = build_partition_long(raw, zone_ids=zone_ids)
    mats = densify_parts(parts, zone_ids)

    # R must have positive mass; studio-only contribution should not match full
    assert float(mats["R"].sum()) > 0
    assert float(mats["full"].sum()) > float(mats["R"].sum())
    # home|PM present
    assert "home|PM" in mats
    assert float(mats["home|PM"].sum()) > 0

    # synthetic metric distances (unit squares ~1 area)
    xy = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    areas = np.ones(n, dtype=float)
    D, qa = build_metric_distance_matrix(
        xy, areas, crs="EPSG:3003", zone_ids=zone_ids, study_area="lombardy"
    )
    assert qa["distance_status"] == "metric_projected"

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "lombardy_cs"
        manifest = write_lombardy_freeze(
            out,
            matrices=mats,
            distances=D,
            distance_qa=qa,
            zone_ids=zone_ids,
            clip_rule="synthetic_smoke",
        )
        assert manifest["n_zones"] == n
        assert manifest["day_stacks"] is False
        assert manifest["studio_held_out_of_R"] is True
        assert (out / "R.npy").exists()
        assert (out / "T_full.npy").exists()
        assert (out / "T_home__PM.npy").exists()
        assert (out / "distances.npy").exists()
        assert (out / "distance_qa.json").exists()
        assert (out / "zone_ids.json").exists()
        assert (out / "materialise_manifest.json").exists()
        assert not (out / "R_by_day").exists()
        assert not (out / "T_by_day").exists()
        qa2 = json.loads((out / "distance_qa.json").read_text(encoding="utf-8"))
        assert qa2["distance_status"] == "metric_projected"
        assert "R.npy" in manifest["sha256_16"]
        assert len(manifest["sha256_16"]["R.npy"]) == 16


if __name__ == "__main__":
    test_lombardy_materialise_synthetic()
    print("SMOKE_OK")
