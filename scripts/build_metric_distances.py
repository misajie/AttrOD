#!/usr/bin/env python3
"""Build draft2 metric distance matrix for a study area (P1).

Reads zone polygons (GeoPackage / GeoJSON / Shapefile), projects to a metric CRS,
writes Euclidean d_ij with d_ii = sqrt(A_i/pi).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from attrOD.spatial.metric_distance import (
    METRIC_DISTANCE,
    build_from_lonlat,
    build_metric_distance_matrix,
    resolve_metric_crs,
    write_distance_artifacts,
)


def _load_zones(path: Path, zone_id_col: str):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit(
            "geopandas is required to read zone polygons (pip install 'attrOD[geo]')"
        ) from exc
    gdf = gpd.read_file(path)
    if zone_id_col not in gdf.columns:
        raise SystemExit(
            f"zone id column {zone_id_col!r} not in {list(gdf.columns)}"
        )
    if gdf.crs is None:
        raise SystemExit("zone file has no CRS; set it before building distances")
    # area in m^2: project to a temporary equal-area or metric CRS for area if geographic
    gdf = gdf.copy()
    gdf["_zid"] = gdf[zone_id_col].astype(str)
    return gdf


def main() -> None:
    p = argparse.ArgumentParser(description="Build draft2 metric d_ij / d_ii")
    p.add_argument("--zones", required=True, help="polygon file (gpkg/geojson/shp)")
    p.add_argument("--zone-id-col", default="id")
    p.add_argument("--study-area", default="mitma", help="mitma | lombardy | ...")
    p.add_argument("--crs", default=None, help="override metric CRS (e.g. EPSG:25830)")
    p.add_argument("--out", default="outputs/distances/mitma")
    p.add_argument("--run-id", default="metric_distance")
    args = p.parse_args()

    gdf = _load_zones(Path(args.zones), args.zone_id_col)
    # representative lon/lat from WGS84 centroids for CRS pick
    gdf_wgs = gdf.to_crs("EPSG:4326")
    cents = gdf_wgs.geometry.centroid
    lon = cents.x.to_numpy()
    lat = cents.y.to_numpy()
    crs = resolve_metric_crs(
        study_area=args.study_area,
        lon=float(np.nanmean(lon)),
        lat=float(np.nanmean(lat)),
        crs=args.crs,
    )
    gdf_m = gdf.to_crs(crs)
    areas = gdf_m.geometry.area.to_numpy(dtype=float)
    cents_m = gdf_m.geometry.centroid
    xy = np.column_stack([cents_m.x.to_numpy(dtype=float), cents_m.y.to_numpy(dtype=float)])
    zone_ids = gdf["_zid"].tolist()

    D, qa = build_metric_distance_matrix(
        xy, areas, crs=crs, zone_ids=zone_ids, study_area=args.study_area
    )
    qa["run_id"] = args.run_id
    qa["zones_path"] = str(Path(args.zones).resolve())
    paths = write_distance_artifacts(
        args.out, D, qa, centroids_xy=xy, areas=areas
    )
    print(
        json.dumps(
            {
                "distance_status": qa["distance_status"],
                "crs": qa["crs"],
                "d_ii_formula": qa["d_ii_formula"],
                "d_ii_formula_confirmed": qa["d_ii_formula_confirmed"],
                "summary": qa["summary"],
                "paths": paths,
            },
            indent=2,
        )
    )
    if qa["distance_status"] != METRIC_DISTANCE or not qa["d_ii_formula_confirmed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
