"""Lombardy materialise + freeze (P5-MAT).

Writes an immutable MITMA-shaped snapshot under outputs/materialised/lombardy_cs/:
R.npy (morning lavoro HBW; studio held out), T partitions, metric distances
(EPSG:3003 Monte Mario by default), zone_ids, materialise_manifest with sha256-16.

No R_by_day / T_by_day — Condition S day bootstrap is undefined for Lombardy.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from attrOD.data.flow_table import long_to_matrix
from attrOD.data.lombardy import (
    MOTIVE_HOME,
    MOTIVE_OTHER,
    MOTIVE_WORK,
    LombardyAdapter,
    build_reference_R_lombardy,
)
from attrOD.spatial.metric_distance import (
    METRIC_DISTANCE,
    build_metric_distance_matrix,
    resolve_metric_crs,
)

PathLike = Union[str, Path]

PM_BANDS = ("pomeriggio", "sera", "pm", "afternoon", "evening", "14-20", "16-20", "12-20")
AM_BANDS = ("mattina", "am", "7-10", "07-10", "morning")


def sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _safe_name(label: str) -> str:
    return str(label).replace("|", "__").replace("/", "_").replace(" ", "_")


def _norm_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def load_od_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"unsupported OD table format: {path}")


def normalise_od_columns(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    renames = {
        "ORIGINE": "origin",
        "DESTINAZIONE": "destination",
        "origin_id": "origin",
        "destination_id": "destination",
        "MOTIVO": "motive",
        "motivo": "motive",
        "FASCIA_ORARIA": "FASCIA_ORARIA",
        "fascia_oraria": "FASCIA_ORARIA",
        "TRIPS": "flow",
        "n_trips": "flow",
        "VALORE": "flow",
    }
    for a, b in renames.items():
        if a in df.columns and b not in df.columns:
            df = df.rename(columns={a: b})
    if "flow" not in df.columns:
        for c in ("TRIPS", "n_trips", "VALORE", "flow"):
            if c in df.columns:
                df = df.rename(columns={c: "flow"})
                break
    if "motive" not in df.columns:
        raise ValueError("OD table needs a motive column (MOTIVO/motive)")
    if "origin" not in df.columns or "destination" not in df.columns:
        raise ValueError("OD table needs origin/destination columns")
    return df


def clip_zone_ids(
    candidate_ids: Sequence[str],
    *,
    zones_gdf=None,
    fua_gdf=None,
    external_ids: Optional[Iterable[str]] = None,
    min_n: int = 25,
) -> List[str]:
    """Drop external gates; optionally keep zones whose centroids fall in FUA.

    Returns sorted unique zone ids. Raises if N < min_n after clip.
    """
    ids = [str(z) for z in candidate_ids]
    ext = {str(x) for x in (external_ids or [])}
    ids = [z for z in ids if z not in ext and not z.lower().startswith("ext")]

    if zones_gdf is not None and fua_gdf is not None:
        import geopandas as gpd

        z = zones_gdf.copy()
        z["_zid"] = z["_zid"].astype(str)
        z = z.loc[z["_zid"].isin(ids)]
        if z.crs != fua_gdf.crs:
            fua = fua_gdf.to_crs(z.crs)
        else:
            fua = fua_gdf
        union = fua.unary_union
        cents = z.geometry.centroid
        keep = cents.within(union) | cents.intersects(union)
        ids = z.loc[keep, "_zid"].astype(str).tolist()

    # stable order
    ids = sorted(set(ids), key=lambda x: (len(x), x))
    if len(ids) < int(min_n):
        raise ValueError(
            f"after clip N={len(ids)} < min_n={min_n}; "
            "widen FUA/provincial subset or lower --min-n"
        )
    return ids


def _time_mask(df: pd.DataFrame, bands: Sequence[str], time_col: str = "FASCIA_ORARIA") -> pd.Series:
    if time_col not in df.columns:
        return pd.Series(True, index=df.index)
    t = _norm_series(df[time_col])
    keys = {b.lower() for b in bands}
    return t.isin(keys) | t.apply(lambda x: any(k in x for k in keys if len(k) >= 2))


def build_partition_long(
    raw: pd.DataFrame,
    *,
    zone_ids: Sequence[str],
    adapter: Optional[LombardyAdapter] = None,
) -> Dict[str, pd.DataFrame]:
    """Build long tables for R and draft2-ish purpose/time T cuts.

    studio is never included in R; it may appear in T_full only as part of all trips
    (held out of R always).
    """
    adapter = adapter or LombardyAdapter()
    df = normalise_od_columns(raw)
    R_long, study_held = build_reference_R_lombardy(df, zone_ids=zone_ids)
    # studio is held out of R by build_reference_R_lombardy; keep study_held for audit only

    zset = set(map(str, zone_ids))
    df = df.loc[
        df["origin"].astype(str).isin(zset) & df["destination"].astype(str).isin(zset)
    ].copy()

    mvals = _norm_series(df["motive"])
    am = _time_mask(df, AM_BANDS)
    pm = _time_mask(df, PM_BANDS)

    parts: Dict[str, pd.DataFrame] = {}
    parts["R"] = R_long
    parts["study_heldout"] = study_held

    # T_full: all internal motives/times (includes studio in total demand; R still excludes it)
    parts["full"] = adapter.to_long(df, partition="full", zone_ids=zone_ids)

    # purpose/time cuts
    home_pm = df.loc[mvals.isin(MOTIVE_HOME) & pm]
    home_am = df.loc[mvals.isin(MOTIVE_HOME) & am]
    other_am = df.loc[mvals.isin(MOTIVE_OTHER) & am]
    work_am = df.loc[mvals.isin(MOTIVE_WORK) & am]

    parts["home|PM"] = adapter.to_long(home_pm, partition="home|PM", zone_ids=zone_ids)
    parts["home|AM"] = adapter.to_long(home_am, partition="home|AM", zone_ids=zone_ids)
    parts["other|AM"] = adapter.to_long(other_am, partition="other|AM", zone_ids=zone_ids)
    # morning lavoro as a T cut (same flows as R science-wise; distinct role vs reference R)
    parts["work|AM"] = adapter.to_long(work_am, partition="work|AM", zone_ids=zone_ids)

    return parts


def densify_parts(
    parts: Mapping[str, pd.DataFrame], zone_ids: Sequence[str]
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for lab, long_df in parts.items():
        if lab in ("R", "study_heldout"):
            continue
        out[lab] = long_to_matrix(long_df, zone_ids)
    out["R"] = long_to_matrix(parts["R"], zone_ids)
    return out


def load_zones_for_distance(
    zones_path: Path,
    zone_id_col: str,
    zone_ids: Sequence[str],
    *,
    study_area: str = "lombardy",
    crs: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any], str]:
    import geopandas as gpd

    gdf = gpd.read_file(zones_path)
    if zone_id_col not in gdf.columns:
        raise ValueError(f"zone id column {zone_id_col!r} not in {list(gdf.columns)}")
    if gdf.crs is None:
        raise ValueError("zone file has no CRS")
    gdf = gdf.copy()
    gdf["_zid"] = gdf[zone_id_col].astype(str)
    # preserve freeze order
    gdf = gdf.set_index("_zid").loc[list(map(str, zone_ids))].reset_index()
    gdf_wgs = gdf.to_crs("EPSG:4326")
    cents = gdf_wgs.geometry.centroid
    crs_res = resolve_metric_crs(
        study_area=study_area,
        lon=float(np.nanmean(cents.x.to_numpy())),
        lat=float(np.nanmean(cents.y.to_numpy())),
        crs=crs,
    )
    gdf_m = gdf.to_crs(crs_res)
    areas = gdf_m.geometry.area.to_numpy(dtype=float)
    cents_m = gdf_m.geometry.centroid
    xy = np.column_stack([cents_m.x.to_numpy(dtype=float), cents_m.y.to_numpy(dtype=float)])
    D, qa = build_metric_distance_matrix(
        xy, areas, crs=crs_res, zone_ids=list(map(str, zone_ids)), study_area=study_area
    )
    return D, qa, crs_res


def write_lombardy_freeze(
    out_dir: PathLike,
    *,
    matrices: Mapping[str, np.ndarray],
    distances: np.ndarray,
    distance_qa: Mapping[str, Any],
    zone_ids: Sequence[str],
    clip_rule: str,
    extra_manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Refuse day stacks
    for bad in ("R_by_day", "T_by_day"):
        p = out / bad
        if p.exists():
            raise RuntimeError(f"refusing to materialise beside existing day stack {p}")

    artifacts: Dict[str, Path] = {}
    np.save(out / "R.npy", np.asarray(matrices["R"], dtype=float))
    artifacts["R.npy"] = out / "R.npy"

    # T_full
    T_full = matrices.get("full")
    if T_full is None:
        raise ValueError("matrices must include 'full' for T_full.npy")
    np.save(out / "T_full.npy", np.asarray(T_full, dtype=float))
    artifacts["T_full.npy"] = out / "T_full.npy"

    for lab, mat in matrices.items():
        if lab in ("R", "full", "study_heldout"):
            continue
        name = f"T_{_safe_name(lab)}.npy"
        np.save(out / name, np.asarray(mat, dtype=float))
        artifacts[name] = out / name

    np.save(out / "distances.npy", np.asarray(distances, dtype=float))
    artifacts["distances.npy"] = out / "distances.npy"
    # also alias distance.npy for older readers
    np.save(out / "distance.npy", np.asarray(distances, dtype=float))
    artifacts["distance.npy"] = out / "distance.npy"

    qa_path = out / "distance_qa.json"
    qa = dict(distance_qa)
    qa["distance_status"] = qa.get("distance_status", METRIC_DISTANCE)
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    artifacts["distance_qa.json"] = qa_path

    zpath = out / "zone_ids.json"
    zpath.write_text(json.dumps(list(map(str, zone_ids)), indent=2), encoding="utf-8")
    artifacts["zone_ids.json"] = zpath

    # Ensure no day dirs created
    for bad in ("R_by_day", "T_by_day"):
        if (out / bad).exists():
            raise RuntimeError(f"day stack directory must not exist: {out / bad}")

    hashes = {name: sha256_16(path) for name, path in artifacts.items()}
    manifest: Dict[str, Any] = {
        "ticket": "P5-MAT",
        "study_area": "lombardy",
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "n_zones": len(zone_ids),
        "crs": qa.get("crs"),
        "distance_status": qa.get("distance_status"),
        "clip_rule": clip_rule,
        "day_stacks": False,
        "studio_held_out_of_R": True,
        "artifacts": {k: {"path": str(v), "sha256_16": hashes[k]} for k, v in artifacts.items()},
        "sha256_16": hashes,
    }
    if extra_manifest:
        manifest.update(dict(extra_manifest))
    mpath = out / "materialise_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def materialise_lombardy(
    *,
    od_path: PathLike,
    zones_path: PathLike,
    out_dir: PathLike,
    zone_id_col: str = "id",
    fua_path: Optional[PathLike] = None,
    external_ids: Optional[Sequence[str]] = None,
    min_n: int = 25,
    crs: Optional[str] = None,
    clip_rule: str = "provincial_or_fua_internal",
    study_area: str = "lombardy",
) -> Dict[str, Any]:
    """End-to-end materialise from raw OD + zone polygons."""
    import geopandas as gpd

    raw = load_od_table(Path(od_path))
    raw = normalise_od_columns(raw)
    zones = gpd.read_file(zones_path)
    if zone_id_col not in zones.columns:
        raise ValueError(f"zone id column {zone_id_col!r} missing")
    zones = zones.copy()
    zones["_zid"] = zones[zone_id_col].astype(str)

    fua = gpd.read_file(fua_path) if fua_path else None
    # candidate ids = zones present in OD or polygon file
    od_zones = sorted(set(raw["origin"].astype(str)) | set(raw["destination"].astype(str)))
    poly_ids = zones["_zid"].astype(str).tolist()
    candidates = [z for z in poly_ids if z in set(od_zones)] or poly_ids

    zone_ids = clip_zone_ids(
        candidates,
        zones_gdf=zones,
        fua_gdf=fua,
        external_ids=external_ids,
        min_n=min_n,
    )
    parts = build_partition_long(raw, zone_ids=zone_ids)
    mats = densify_parts(parts, zone_ids)
    D, qa, crs_res = load_zones_for_distance(
        Path(zones_path), zone_id_col, zone_ids, study_area=study_area, crs=crs
    )
    # verify studio not in R mass from raw motives overlapping R cells — soft check via heldout
    return write_lombardy_freeze(
        out_dir,
        matrices=mats,
        distances=D,
        distance_qa=qa,
        zone_ids=zone_ids,
        clip_rule=clip_rule,
        extra_manifest={"crs_resolved": crs_res, "od_path": str(od_path), "zones_path": str(zones_path)},
    )
