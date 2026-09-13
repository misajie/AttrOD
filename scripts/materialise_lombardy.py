#!/usr/bin/env python3
"""P5-MAT: materialise Lombardy freeze (MITMA-shaped, no day stacks).

Example (compute node, existing venv — do not run on login node):

  attrOD-materialise-lombardy \\
    --od ~/AttrOD/data/lombardy/od.parquet \\
    --zones ~/AttrOD/data/lombardy/zones.gpkg \\
    --zone-id-col id \\
    --fua ~/AttrOD/data/lombardy/milan_fua.gpkg \\
    --out outputs/materialised/lombardy_cs

Writes R.npy (morning lavoro; studio held out), T_full.npy + purpose/time
T_*.npy, distances.npy + distance_qa.json (metric_projected, EPSG:3003 default),
zone_ids.json, materialise_manifest.json (sha256-16). Never writes R_by_day/T_by_day.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from attrOD.data.lombardy_materialise import materialise_lombardy


def main() -> None:
    p = argparse.ArgumentParser(description="P5-MAT Lombardy materialise + freeze")
    p.add_argument("--od", required=True, help="raw OD table (csv/parquet/xlsx)")
    p.add_argument("--zones", required=True, help="zone polygons (gpkg/geojson/shp)")
    p.add_argument("--zone-id-col", default="id")
    p.add_argument("--fua", default=None, help="optional FUA/provincial polygon for clip")
    p.add_argument("--out", default="outputs/materialised/lombardy_cs")
    p.add_argument("--data-root", default="~/AttrOD/data/lombardy")
    p.add_argument("--crs", default=None, help="override metric CRS (default EPSG:3003)")
    p.add_argument("--min-n", type=int, default=25)
    p.add_argument(
        "--clip-rule",
        default="milan_fua_or_provincial_internal",
        help="recorded in materialise_manifest.json",
    )
    p.add_argument(
        "--external-id",
        action="append",
        default=[],
        help="zone id to drop as external gate (repeatable)",
    )
    args = p.parse_args()

    manifest = materialise_lombardy(
        od_path=args.od,
        zones_path=args.zones,
        out_dir=args.out,
        zone_id_col=args.zone_id_col,
        fua_path=args.fua,
        external_ids=args.external_id or None,
        min_n=args.min_n,
        crs=args.crs,
        clip_rule=args.clip_rule,
    )
    print(json.dumps(manifest, indent=2))
    if manifest.get("n_zones", 0) < args.min_n:
        raise SystemExit(2)
    if manifest.get("distance_status") != "metric_projected":
        raise SystemExit(3)
    if (Path(args.out) / "R_by_day").exists() or (Path(args.out) / "T_by_day").exists():
        raise SystemExit("day stacks must not be written")


if __name__ == "__main__":
    main()
