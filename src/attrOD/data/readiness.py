"""Dataset readiness checks (Ticket 1 / NOW-0).

- daily trips completeness (MITMA 2022-10 expected 31 days)
- zone ↔ population join coverage
- HK FID 9 quarantine (never silent fill)
- provisional distance gate (blocks paper_mainline)
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

HK_FID_MISSING_DEFAULT = ("9",)
PROVISIONAL_DISTANCE = "provisional_wgs84_not_metric"


def _parse_yyyymmdd(token: str) -> Optional[date]:
    token = token.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    return None


def expected_date_range(start: date, end: date) -> List[date]:
    out: List[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def validate_daily_trips_completeness(
    daily_trips_root: PathLike,
    expected_dates: Optional[Sequence[Union[str, date]]] = None,
    *,
    pattern: str = r"(?P<ymd>\d{8})",
    min_bytes: int = 1,
) -> Dict[str, Any]:
    """Check daily trip files under *daily_trips_root*.

    Default expected set when *expected_dates* is None: 2022-10-01..2022-10-31
    (MITMA October 2022 mainline window from SOURCES.md — not invented cities).
    """
    root = Path(daily_trips_root)
    if expected_dates is None:
        expected = expected_date_range(date(2022, 10, 1), date(2022, 10, 31))
    else:
        expected = []
        for x in expected_dates:
            if isinstance(x, date):
                expected.append(x)
            else:
                parsed = _parse_yyyymmdd(str(x).replace("-", ""))
                if parsed is None:
                    raise ValueError(f"unparseable expected date: {x}")
                expected.append(parsed)

    expected_set = set(expected)
    found: Dict[date, Path] = {}
    duplicates: List[str] = []
    partial: List[Dict[str, Any]] = []
    unparsed: List[str] = []

    if not root.exists():
        return {
            "ok": False,
            "root": str(root),
            "expected_n": len(expected_set),
            "found_n": 0,
            "complete_dates": [],
            "missing_dates": sorted(d.isoformat() for d in expected_set),
            "duplicate_dates": [],
            "partial_days": [],
            "unparsed_files": [],
            "checksums": {},
        }

    rx = re.compile(pattern)
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.suffix in {".md", ".txt"}:
            continue
        m = rx.search(p.name)
        if not m:
            unparsed.append(p.name)
            continue
        ymd = m.group("ymd") if "ymd" in m.groupdict() and m.group("ymd") else m.group(0)
        d = _parse_yyyymmdd(ymd)
        if d is None:
            unparsed.append(p.name)
            continue
        if d in found:
            duplicates.append(f"{d.isoformat()}: {found[d].name} and {p.name}")
        found[d] = p
        size = p.stat().st_size
        if size < min_bytes:
            partial.append({"date": d.isoformat(), "path": str(p), "size": size, "reason": "too_small"})

    complete = sorted(d for d in found if d in expected_set)
    missing = sorted(expected_set - set(found))
    # days present but outside expected window are noted, not fatal for completeness of window
    extras = sorted(set(found) - expected_set)

    checksums: Dict[str, Dict[str, Any]] = {}
    for d, p in found.items():
        if d in expected_set:
            checksums[d.isoformat()] = {
                "path": str(p),
                "size": p.stat().st_size,
            }

    ok = (
        len(missing) == 0
        and len(duplicates) == 0
        and len(partial) == 0
        and len(expected_set) > 0
        and len(complete) == len(expected_set)
    )
    return {
        "ok": ok,
        "root": str(root.resolve()) if root.exists() else str(root),
        "expected_n": len(expected_set),
        "found_n": len(complete),
        "complete_dates": [d.isoformat() for d in complete],
        "missing_dates": [d.isoformat() for d in missing],
        "duplicate_dates": duplicates,
        "partial_days": partial,
        "extra_dates": [d.isoformat() for d in extras],
        "unparsed_files": unparsed,
        "checksums": checksums,
    }


def validate_zone_population_join(
    zones: Union[pd.DataFrame, Sequence[Any], Mapping[str, Any]],
    population: Union[pd.DataFrame, Sequence[Any], Mapping[str, Any]],
    od_keys: Optional[Sequence[Any]] = None,
    *,
    zone_key: str = "zone_id",
    pop_key: str = "zone_id",
    pop_value: str = "population",
) -> Dict[str, Any]:
    """Audit join coverage between zones, population, and optional OD keys."""

    def _to_ids(obj, key: str) -> List[str]:
        if isinstance(obj, pd.DataFrame):
            if key not in obj.columns:
                # try common alternates
                for alt in ("fid", "id", "distrito", "ID", "zone"):
                    if alt in obj.columns:
                        key = alt
                        break
                else:
                    raise KeyError(f"zone key {key!r} not in columns {list(obj.columns)}")
            return [str(x) for x in obj[key].tolist()]
        if isinstance(obj, Mapping):
            if key in obj:
                val = obj[key]
                if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
                    return [str(x) for x in val]
            return [str(k) for k in obj.keys()]
        return [str(x) for x in obj]

    zone_ids = _to_ids(zones, zone_key)
    pop_ids = _to_ids(population, pop_key)
    zone_set, pop_set = set(zone_ids), set(pop_ids)

    dup_zones = sorted({z for z in zone_ids if zone_ids.count(z) > 1})
    dup_pop = sorted({z for z in pop_ids if pop_ids.count(z) > 1})
    zones_missing_pop = sorted(zone_set - pop_set)
    pop_orphans = sorted(pop_set - zone_set)

    od_report: Dict[str, Any] = {}
    if od_keys is not None:
        od_set = {str(x) for x in od_keys}
        od_report = {
            "od_n": len(od_set),
            "od_missing_in_zones": sorted(od_set - zone_set),
            "zones_missing_in_od": sorted(zone_set - od_set),
        }

    # unit consistency: if DataFrame with population column, check non-negative finite
    unit_ok = True
    unit_notes: List[str] = []
    if isinstance(population, pd.DataFrame) and pop_value in population.columns:
        vals = pd.to_numeric(population[pop_value], errors="coerce")
        if vals.isna().any():
            unit_ok = False
            unit_notes.append("NaN population values")
        if (vals < 0).any():
            unit_ok = False
            unit_notes.append("negative population values")

    coverage = 1.0 if not zone_set else (len(zone_set & pop_set) / len(zone_set))
    ok = (
        coverage == 1.0
        and not dup_zones
        and not dup_pop
        and unit_ok
        and (not od_report or not od_report.get("od_missing_in_zones"))
    )
    return {
        "ok": ok,
        "n_zones": len(zone_set),
        "n_population": len(pop_set),
        "join_coverage": coverage,
        "zones_missing_population": zones_missing_pop,
        "population_orphan_keys": pop_orphans,
        "duplicate_zone_keys": dup_zones,
        "duplicate_population_keys": dup_pop,
        "unit_ok": unit_ok,
        "unit_notes": unit_notes,
        "od": od_report,
    }


def quarantine_hk_fid9(
    zone_ids: Sequence[Any],
    *,
    missing_fids: Sequence[Any] = HK_FID_MISSING_DEFAULT,
    matrix: Optional[np.ndarray] = None,
    reason: str = "FID absent from CHKU shapefile; never silent-fill",
) -> Dict[str, Any]:
    """Explicitly quarantine HK missing FIDs (default FID 9).

    Does **not** impute. If *matrix* is provided it must already exclude the
    missing FID from its axis (shape matches present zones); otherwise a
    warning is recorded.
    """
    present = [str(z) for z in zone_ids]
    missing = [str(f) for f in missing_fids]
    leaked = [f for f in missing if f in present]
    report = {
        "quarantined_fids": missing,
        "present_zone_ids": present,
        "leaked_into_present": leaked,
        "reason": reason,
        "ok": len(leaked) == 0,
        "n_present": len(present),
    }
    if matrix is not None:
        mat = np.asarray(matrix)
        report["matrix_shape"] = list(mat.shape)
        if mat.shape[0] != len(present) or mat.shape[1] != len(present):
            report["ok"] = False
            report["matrix_mismatch"] = (
                f"matrix {mat.shape} vs n_present={len(present)}; "
                "FID quarantine axis mismatch"
            )
    return report


def validate_metric_distance(
    *,
    distance_status: Optional[str] = None,
    distances: Optional[np.ndarray] = None,
    crs: Optional[str] = None,
    allow_provisional_for_smoke: bool = True,
) -> Dict[str, Any]:
    """Gate metric CRS distances; provisional WGS84 blocks paper_mainline."""
    status = distance_status
    notes: List[str] = []
    paper_mainline_allowed = True
    smoke_allowed = True

    if status == PROVISIONAL_DISTANCE or status == "provisional_wgs84_not_metric":
        paper_mainline_allowed = False
        smoke_allowed = bool(allow_provisional_for_smoke)
        notes.append("provisional WGS84 distances — verification/smoke only")
        status = PROVISIONAL_DISTANCE
    elif status in (None, ""):
        if crs and ("4326" in str(crs) or "WGS84" in str(crs).upper()):
            paper_mainline_allowed = False
            notes.append(f"CRS {crs} is not a metric projected CRS")
            status = status or "unverified_crs"
        else:
            status = status or "unspecified"

    stats: Dict[str, Any] = {}
    if distances is not None:
        d = np.asarray(distances, dtype=float)
        finite = d[np.isfinite(d)]
        stats = {
            "shape": list(d.shape),
            "n_nan": int(np.isnan(d).sum()),
            "n_inf": int(np.isinf(d).sum()),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "n_negative": int((finite < 0).sum()) if finite.size else 0,
        }
        if stats["n_negative"] > 0 or stats["n_inf"] > 0:
            paper_mainline_allowed = False
            smoke_allowed = False
            notes.append("non-finite or negative distances")

    return {
        "ok_smoke": smoke_allowed,
        "ok_mainline": paper_mainline_allowed,
        "distance_status": status,
        "crs": crs,
        "stats": stats,
        "notes": notes,
        "verification_only_required": not paper_mainline_allowed,
    }


def build_dataset_readiness(
    *,
    data_root: PathLike,
    dataset: str = "mitma",
    hk_zone_ids: Optional[Sequence[Any]] = None,
    hk_qa: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate readiness document for outputs/readiness/dataset_readiness.json."""
    root = Path(data_root)
    out: Dict[str, Any] = {"dataset": dataset, "data_root": str(root), "checks": {}}

    if dataset in ("mitma", "all"):
        trips = root / "mitma" / "daily_trips"
        out["checks"]["daily_trips"] = validate_daily_trips_completeness(trips)
        # zone-pop join if files exist
        pop_path = root / "mitma" / "population" / "poblacion_distritos.csv"
        zones_csv = root / "mitma" / "districts" / "nombres_distritos.csv"
        if pop_path.exists() and zones_csv.exists():
            def _read_mitma_pipe(path: Path) -> pd.DataFrame:
                # MITMA district/population extracts are pipe-delimited, often headerless
                df = pd.read_csv(path, sep="|", header=None, dtype=str, engine="python")
                if df.shape[1] >= 2:
                    df = df.iloc[:, :2]
                    df.columns = ["zone_id", "value"]
                else:
                    df.columns = ["zone_id"]
                return df
            try:
                pop_df = _read_mitma_pipe(pop_path)
                zones_df = _read_mitma_pipe(zones_csv)
                # population value column
                if "value" in pop_df.columns:
                    pop_df = pop_df.rename(columns={"value": "population"})
                out["checks"]["zone_population"] = validate_zone_population_join(
                    zones_df, pop_df, zone_key="zone_id", pop_key="zone_id",
                    pop_value="population" if "population" in pop_df.columns else "value",
                )
            except Exception as exc:
                out["checks"]["zone_population"] = {
                    "ok": False,
                    "reason": f"failed to parse districts/population: {exc}",
                }
        else:
            out["checks"]["zone_population"] = {
                "ok": False,
                "reason": "population or districts CSV missing",
            }

    if dataset in ("hk", "all"):
        qa = dict(hk_qa or {})
        zone_ids = list(hk_zone_ids or qa.get("zone_ids") or [])
        missing = qa.get("fid_missing_in_shapefile") or HK_FID_MISSING_DEFAULT
        out["checks"]["hk_fid_quarantine"] = quarantine_hk_fid9(
            zone_ids, missing_fids=missing
        )
        out["checks"]["hk_distance"] = validate_metric_distance(
            distance_status=qa.get("distance_label", PROVISIONAL_DISTANCE),
            crs=qa.get("crs_note"),
        )
        out["checks"]["lombardy_osm_note"] = {
            "lombardy_empty_gate": "see data/lombardy readiness when populated",
            "osm_empty_gate": "see data/osm/extracts",
        }

    # overall: mainline blocked if any critical check fails
    mainline_ok = True
    if "daily_trips" in out["checks"] and not out["checks"]["daily_trips"].get("ok"):
        mainline_ok = False
    # Hard gate (AttrOD room): any provisional distance kills paper mainline
    for _name, chk in out["checks"].items():
        if not isinstance(chk, dict):
            continue
        if chk.get("distance_status") == PROVISIONAL_DISTANCE:
            mainline_ok = False
        if chk.get("verification_only_required") is True:
            mainline_ok = False
        if "distance" in _name and chk.get("ok_mainline") is False:
            mainline_ok = False
    if dataset == "hk":
        mainline_ok = False
    out["paper_mainline_allowed"] = bool(mainline_ok)
    out["verification_only"] = (not mainline_ok) or dataset == "hk"
    out["ok_mainline"] = bool(mainline_ok)
    return out
