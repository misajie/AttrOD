"""Teralytics Hong Kong OD matrix adapter (S1 materialisation).

Reference R_HBW_proxy (strict 07–10 HBW unavailable in this product):
  - weekdays only (Mon–Fri) in the analysis date window
  - PartOfDay matching morning bin ``4:00-12:00`` (normalised variants)
  - TripPurpose ``to-work``

TOTAL matrix aggregation choice (documented in qa.json):
  If any of Age / Gender / TripDistance / TripDuration equals the literal
  pre-aggregate token ``All``, sum **only atomic rows** where those four
  dims are not ``All`` (avoids double-counting vendor roll-ups). If the
  extract is purely atomic (no ``All`` in those dims), sum all rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .flow_table import FLOW_COLUMNS, assert_flow_schema, long_to_matrix

DELIMITER = "|"
COUNT_CANDIDATES = ("Count", "Trips", "Flow", "count", "trips", "flow", "n_trips")
ATTR_DIMS_FOR_ALL = ("Age", "Gender", "TripDistance", "TripDuration")
MORNING_PART_CANONICAL = "4:00-12:00"
PURPOSE_TO_WORK = "to-work"
ALL_TOKEN = "All"

# Normalise common PartOfDay spellings onto the anonymisation-stats forms.
_PART_ALIASES = {
    "4:00-12:00": "4:00-12:00",
    "04:00-12:00": "4:00-12:00",
    "4:00–12:00": "4:00-12:00",
    "4-12": "4:00-12:00",
    "04-12": "4:00-12:00",
    "morning": "4:00-12:00",
    "0:00-4:00": "0:00-4:00",
    "00:00-4:00": "0:00-4:00",
    "00:00-04:00": "0:00-4:00",
    "12:00-20:00": "12:00-20:00",
    "20:00-0:00": "20:00-0:00",
    "20:00-00:00": "20:00-0:00",
    "20:00-24:00": "20:00-0:00",
}


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _strip_dbf_str(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("latin1", errors="replace").strip()


def read_dbf_zones(dbf_path: Path | str) -> pd.DataFrame:
    """Read CHKU-style DBF with fields name (C), lon (N), lat (N), FID (C).

    Returns columns: fid, name, lon, lat — ordered by numeric fid.
    """
    path = Path(dbf_path)
    data = path.read_bytes()
    if len(data) < 32:
        raise ValueError(f"DBF too short: {path}")
    nrec = struct.unpack_from("<I", data, 4)[0]
    hlen = struct.unpack_from("<H", data, 8)[0]
    rlen = struct.unpack_from("<H", data, 10)[0]
    fields: List[Tuple[str, str, int, int]] = []
    off = 32
    while off < hlen and data[off] != 0x0D:
        name = data[off : off + 11].split(b"\x00", 1)[0].decode("ascii", "replace")
        typ = chr(data[off + 11])
        flen = data[off + 16]
        dec = data[off + 17]
        fields.append((name, typ, flen, dec))
        off += 32
    rows = []
    pos = hlen
    for _ in range(nrec):
        if pos + rlen > len(data):
            break
        rec = data[pos + 1 : pos + rlen]
        vals: Dict[str, Any] = {}
        p = 0
        for name, typ, flen, dec in fields:
            raw = rec[p : p + flen]
            p += flen
            if typ in ("C", "c"):
                vals[name] = _strip_dbf_str(raw)
            elif typ in ("N", "F", "n", "f"):
                s = _strip_dbf_str(raw)
                vals[name] = float(s) if s else float("nan")
            else:
                vals[name] = _strip_dbf_str(raw)
        pos += rlen
        fid = str(vals.get("FID", vals.get("fid", ""))).strip()
        rows.append(
            {
                "fid": fid,
                "name": str(vals.get("name", "")).strip(),
                "lon": float(vals.get("lon", float("nan"))),
                "lat": float(vals.get("lat", float("nan"))),
            }
        )
    z = pd.DataFrame(rows)
    z = z.loc[z["fid"] != ""].copy()
    z["_fid_num"] = pd.to_numeric(z["fid"], errors="coerce")
    z = z.sort_values("_fid_num", kind="mergesort").drop(columns="_fid_num").reset_index(drop=True)
    z["fid"] = z["fid"].astype(str)
    return z


def detect_count_column(columns: Sequence[str]) -> str:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for cand in COUNT_CANDIDATES:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    # last resort: last numeric-looking name hint
    for c in cols:
        if c.lower() in {"value", "n", "weight"}:
            return c
    raise KeyError(
        f"No count measure column among {cols}; tried {COUNT_CANDIDATES}"
    )


def normalize_part_of_day(val: object) -> str:
    s = str(val).strip()
    key = s.replace("–", "-").replace("—", "-")
    if key in _PART_ALIASES:
        return _PART_ALIASES[key]
    low = key.lower()
    if low in _PART_ALIASES:
        return _PART_ALIASES[low]
    # collapse spaces
    compact = key.replace(" ", "")
    if compact in _PART_ALIASES:
        return _PART_ALIASES[compact]
    return key


def is_weekday(dates: pd.Series) -> pd.Series:
    dts = pd.to_datetime(dates, errors="coerce")
    return dts.dt.dayofweek < 5


def has_all_preaggregates(df: pd.DataFrame, dims: Sequence[str] = ATTR_DIMS_FOR_ALL) -> bool:
    for d in dims:
        if d in df.columns and (df[d].astype(str).str.strip() == ALL_TOKEN).any():
            return True
    return False


def mask_atomic_attribute_rows(
    df: pd.DataFrame,
    dims: Sequence[str] = ATTR_DIMS_FOR_ALL,
) -> pd.Series:
    """True for rows that are not vendor ``All`` roll-ups on attribute dims."""
    mask = pd.Series(True, index=df.index)
    for d in dims:
        if d in df.columns:
            mask &= df[d].astype(str).str.strip() != ALL_TOKEN
    return mask


def filter_exclude_all_on_dims(
    df: pd.DataFrame,
    dims: Sequence[str],
) -> pd.DataFrame:
    """When materialising a fine partition, drop rows where any listed dim is All."""
    out = df
    for d in dims:
        if d in out.columns:
            out = out.loc[out[d].astype(str).str.strip() != ALL_TOKEN]
    return out


def iter_matrix_chunks(
    csv_path: Path | str,
    *,
    chunksize: int = 500_000,
    delimiter: str = DELIMITER,
) -> Iterator[pd.DataFrame]:
    path = Path(csv_path)
    reader = pd.read_csv(
        path,
        sep=delimiter,
        chunksize=chunksize,
        dtype=str,
        low_memory=False,
    )
    for chunk in reader:
        yield chunk


def _ensure_ids_str(s: pd.Series) -> pd.Series:
    # numeric strings like "9.0" → "9" when clean ints
    num = pd.to_numeric(s, errors="coerce")
    out = s.astype(str).str.strip()
    intish = num.notna() & (num == num.round(0))
    out = out.where(~intish, num.loc[intish].astype("int64").astype(str))
    return out


def prepare_chunk(
    chunk: pd.DataFrame,
    *,
    count_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Normalise a raw matrix chunk: ids, count, PartOfDay."""
    df = chunk.copy()
    rename = {}
    for a, b in [
        ("startid", "StartId"),
        ("endid", "EndId"),
        ("start_id", "StartId"),
        ("end_id", "EndId"),
    ]:
        for c in df.columns:
            if c.lower() == a and b not in df.columns:
                rename[c] = b
    df = df.rename(columns=rename)
    if "StartId" not in df.columns or "EndId" not in df.columns:
        raise KeyError(f"StartId/EndId required; got {list(df.columns)}")
    if count_col is None:
        count_col = detect_count_column(df.columns)
    if count_col not in df.columns:
        raise KeyError(f"count column {count_col!r} missing")
    df["StartId"] = _ensure_ids_str(df["StartId"])
    df["EndId"] = _ensure_ids_str(df["EndId"])
    df["_flow"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0.0)
    if "PartOfDay" in df.columns:
        df["PartOfDay"] = df["PartOfDay"].map(normalize_part_of_day)
    if "TripPurpose" in df.columns:
        df["TripPurpose"] = df["TripPurpose"].astype(str).str.strip()
    if "Date" in df.columns:
        df["Date"] = df["Date"].astype(str).str.strip()
    return df, count_col


def provisional_distance_matrix(
    lon: Sequence[float],
    lat: Sequence[float],
    *,
    method: str = "equirectangular",
) -> np.ndarray:
    """Pairwise provisional distances (metres) on WGS84 lon/lat.

    Label in qa: ``provisional_wgs84_not_metric``. Diagonal set to NaN
    (approximate area / d_ii deferred).
    """
    lon_a = np.asarray(lon, dtype=float)
    lat_a = np.asarray(lat, dtype=float)
    n = len(lon_a)
    R = 6_371_000.0  # mean Earth radius (m)
    lat_r = np.radians(lat_a)
    lon_r = np.radians(lon_a)
    if method == "haversine":
        dlat = lat_r[:, None] - lat_r[None, :]
        dlon = lon_r[:, None] - lon_r[None, :]
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2.0) ** 2
        )
        dist = 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    else:
        # equirectangular
        lat0 = np.radians(float(np.nanmean(lat_a)))
        x = R * lon_r * math.cos(lat0)
        y = R * lat_r
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.nan)
    return dist.astype(float)


def aggregate_od_from_chunk(
    df: pd.DataFrame,
    *,
    zone_ids: Sequence[str],
    weekdays_only: bool = False,
    part_of_day: Optional[str] = None,
    trip_purpose: Optional[str] = None,
    atomic_attrs_only: bool = True,
    drop_all_on: Optional[Sequence[str]] = None,
    keep_date: bool = False,
) -> pd.DataFrame:
    """Filter + aggregate one prepared chunk to origin/destination[/date] flow."""
    zset = set(map(str, zone_ids))
    d = df.loc[df["StartId"].isin(zset) & df["EndId"].isin(zset)].copy()
    if atomic_attrs_only and has_all_preaggregates(d):
        d = d.loc[mask_atomic_attribute_rows(d)]
    if drop_all_on:
        d = filter_exclude_all_on_dims(d, drop_all_on)
    if weekdays_only and "Date" in d.columns:
        d = d.loc[is_weekday(d["Date"])]
    if part_of_day is not None and "PartOfDay" in d.columns:
        want = normalize_part_of_day(part_of_day)
        d = d.loc[d["PartOfDay"].map(normalize_part_of_day) == want]
    if trip_purpose is not None and "TripPurpose" in d.columns:
        d = d.loc[d["TripPurpose"].astype(str).str.strip() == trip_purpose]
    if d.empty:
        cols = ["origin", "destination", "flow"] + (["date"] if keep_date else [])
        return pd.DataFrame(columns=cols)
    if keep_date and "Date" in d.columns:
        g = (
            d.groupby(["StartId", "EndId", "Date"], dropna=False, as_index=False)["_flow"]
            .sum()
            .rename(columns={"StartId": "origin", "EndId": "destination", "Date": "date", "_flow": "flow"})
        )
    else:
        g = (
            d.groupby(["StartId", "EndId"], dropna=False, as_index=False)["_flow"]
            .sum()
            .rename(columns={"StartId": "origin", "EndId": "destination", "_flow": "flow"})
        )
    return g


def collect_csv_ids(csv_path: Path | str, *, chunksize: int = 500_000, delimiter: str = DELIMITER) -> set[str]:
    ids: set[str] = set()
    count_col: Optional[str] = None
    for chunk in iter_matrix_chunks(csv_path, chunksize=chunksize, delimiter=delimiter):
        df, count_col = prepare_chunk(chunk, count_col=count_col)
        ids.update(df["StartId"].unique())
        ids.update(df["EndId"].unique())
    return ids


def build_proxy_R_long(
    csv_path: Path | str,
    zone_ids: Sequence[str],
    *,
    chunksize: int = 500_000,
    delimiter: str = DELIMITER,
    count_col: Optional[str] = None,
    morning_part: str = MORNING_PART_CANONICAL,
    purpose: str = PURPOSE_TO_WORK,
) -> Tuple[pd.DataFrame, str]:
    """Materialise R_HBW_proxy long table (date=null after sum across weekdays)."""
    acc: Dict[Tuple[str, str], float] = {}
    detected = count_col
    for chunk in iter_matrix_chunks(csv_path, chunksize=chunksize, delimiter=delimiter):
        df, detected = prepare_chunk(chunk, count_col=detected)
        g = aggregate_od_from_chunk(
            df,
            zone_ids=zone_ids,
            weekdays_only=True,
            part_of_day=morning_part,
            trip_purpose=purpose,
            atomic_attrs_only=True,
            keep_date=False,
        )
        for o, dest, f in g.itertuples(index=False):
            key = (str(o), str(dest))
            acc[key] = acc.get(key, 0.0) + float(f)
    rows = [
        {"origin": o, "destination": d, "flow": f, "partition": "R_HBW_proxy", "date": pd.NaT}
        for (o, d), f in acc.items()
    ]
    out = assert_flow_schema(pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(FLOW_COLUMNS)))
    return out, detected or "Count"


def build_total_long(
    csv_path: Path | str,
    zone_ids: Sequence[str],
    *,
    chunksize: int = 500_000,
    delimiter: str = DELIMITER,
    count_col: Optional[str] = None,
    weekdays_only: bool = True,
    keep_date: bool = False,
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """Materialise TOTAL long table with documented All-handling."""
    acc: Dict[Tuple, float] = {}
    detected = count_col
    saw_all = False
    source_dates: set[str] = set()
    analysis_dates: set[str] = set()
    flow_sum_raw = 0.0
    for chunk in iter_matrix_chunks(csv_path, chunksize=chunksize, delimiter=delimiter):
        df, detected = prepare_chunk(chunk, count_col=detected)
        flow_sum_raw += float(df["_flow"].sum())
        if "Date" in df.columns:
            source_dates.update(df["Date"].dropna().astype(str).unique())
        if has_all_preaggregates(df):
            saw_all = True
        g = aggregate_od_from_chunk(
            df,
            zone_ids=zone_ids,
            weekdays_only=weekdays_only,
            part_of_day=None,
            trip_purpose=None,
            atomic_attrs_only=True,
            keep_date=keep_date,
        )
        if keep_date:
            for o, dest, date, f in g.itertuples(index=False):
                analysis_dates.add(str(date))
                key = (str(o), str(dest), str(date))
                acc[key] = acc.get(key, 0.0) + float(f)
        else:
            if weekdays_only and "Date" in df.columns:
                analysis_dates.update(
                    df.loc[is_weekday(df["Date"]), "Date"].dropna().astype(str).unique()
                )
            for o, dest, f in g.itertuples(index=False):
                key = (str(o), str(dest))
                acc[key] = acc.get(key, 0.0) + float(f)
    if keep_date:
        rows = [
            {"origin": o, "destination": d, "flow": f, "partition": "TOTAL", "date": dt}
            for (o, d, dt), f in acc.items()
        ]
    else:
        rows = [
            {"origin": o, "destination": d, "flow": f, "partition": "TOTAL", "date": pd.NaT}
            for (o, d), f in acc.items()
        ]
    out = assert_flow_schema(pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(FLOW_COLUMNS)))
    meta = {
        "saw_all_preaggregates": saw_all,
        "total_aggregation": (
            "sum_atomic_Age_Gender_TripDistance_TripDuration_not_All"
            if saw_all
            else "sum_all_rows_matrix_appears_atomic"
        ),
        "source_dates": sorted(source_dates),
        "analysis_dates": sorted(analysis_dates) if analysis_dates else sorted(
            d for d in source_dates if _date_is_weekday(d)
        ),
        "flow_sum_source_approx": flow_sum_raw,
        "weekdays_only": weekdays_only,
    }
    return out, detected or "Count", meta


def _date_is_weekday(d: str) -> bool:
    ts = pd.to_datetime(d, errors="coerce")
    if pd.isna(ts):
        return False
    return int(ts.dayofweek) < 5


def build_partition_long(
    csv_path: Path | str,
    zone_ids: Sequence[str],
    *,
    partition_label: str,
    chunksize: int = 500_000,
    delimiter: str = DELIMITER,
    count_col: Optional[str] = None,
    weekdays_only: bool = True,
    part_of_day: Optional[str] = None,
    trip_purpose: Optional[str] = None,
) -> pd.DataFrame:
    acc: Dict[Tuple[str, str], float] = {}
    detected = count_col
    drop_dims = list(ATTR_DIMS_FOR_ALL)
    for chunk in iter_matrix_chunks(csv_path, chunksize=chunksize, delimiter=delimiter):
        df, detected = prepare_chunk(chunk, count_col=detected)
        g = aggregate_od_from_chunk(
            df,
            zone_ids=zone_ids,
            weekdays_only=weekdays_only,
            part_of_day=part_of_day,
            trip_purpose=trip_purpose,
            atomic_attrs_only=True,
            drop_all_on=drop_dims,
            keep_date=False,
        )
        for o, dest, f in g.itertuples(index=False):
            key = (str(o), str(dest))
            acc[key] = acc.get(key, 0.0) + float(f)
    rows = [
        {
            "origin": o,
            "destination": d,
            "flow": f,
            "partition": partition_label,
            "date": pd.NaT,
        }
        for (o, d), f in acc.items()
    ]
    return assert_flow_schema(pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(FLOW_COLUMNS)))


def write_long_table(df: pd.DataFrame, path: Path | str) -> str:
    """Write long flow table; prefer parquet if pyarrow available else csv.gz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401

        out = path if path.suffix == ".parquet" else path.with_suffix(".parquet")
        # if caller passed foo.parquet.csv-style, normalise
        if str(path).endswith(".parquet"):
            out = path
        else:
            stem = path.name
            if stem.endswith(".csv.gz"):
                out = path.with_name(stem[: -len(".csv.gz")] + ".parquet")
            elif path.suffix in {".csv", ".gz"}:
                out = path.with_suffix(".parquet")
            else:
                out = Path(str(path) + ".parquet") if path.suffix == "" else path.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        return str(out)
    except Exception:
        if str(path).endswith(".csv.gz"):
            out = path
        elif path.suffix == ".parquet":
            out = path.with_name(path.stem + ".csv.gz")
        else:
            out = Path(str(path) + ".csv.gz")
        df.to_csv(out, index=False, compression="gzip")
        return str(out)


def materialise_hk_teralytics(
    csv_path: Path | str,
    shapefile_dir: Path | str,
    out_dir: Path | str,
    *,
    chunksize: int = 500_000,
    delimiter: str = DELIMITER,
    write_purpose_period_partitions: bool = True,
) -> Dict[str, Any]:
    """Run full S1 HK Teralytics materialisation; write artefacts under out_dir."""
    csv_path = Path(csv_path)
    shapefile_dir = Path(shapefile_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dbf_candidates = list(shapefile_dir.glob("*.dbf"))
    if not dbf_candidates:
        raise FileNotFoundError(f"No .dbf in {shapefile_dir}")
    dbf_path = dbf_candidates[0]
    zones = read_dbf_zones(dbf_path)
    zone_ids = list(zones["fid"].astype(str))
    zones.to_csv(out / "zones.csv", index=False)

    # peek columns
    peek = pd.read_csv(csv_path, sep=delimiter, nrows=5, dtype=str)
    count_col = detect_count_column(peek.columns)
    columns_detected = list(peek.columns)

    csv_ids = collect_csv_ids(csv_path, chunksize=chunksize, delimiter=delimiter)
    zone_set = set(zone_ids)
    unmatched = sorted(csv_ids - zone_set, key=lambda x: (len(x), x))

    total_long, count_col, total_meta = build_total_long(
        csv_path,
        zone_ids,
        chunksize=chunksize,
        delimiter=delimiter,
        count_col=count_col,
        weekdays_only=True,
        keep_date=False,
    )
    proxy_long, _ = build_proxy_R_long(
        csv_path,
        zone_ids,
        chunksize=chunksize,
        delimiter=delimiter,
        count_col=count_col,
    )

    total_path = write_long_table(total_long, out / "long_total.parquet")
    proxy_path = write_long_table(proxy_long, out / "long_R_HBW_proxy.parquet")

    total_mat = long_to_matrix(total_long, zone_ids, partition="TOTAL")
    proxy_mat = long_to_matrix(proxy_long, zone_ids, partition="R_HBW_proxy")
    np.save(out / "total.npy", total_mat)
    np.save(out / "R_HBW_proxy.npy", proxy_mat)

    dist = provisional_distance_matrix(zones["lon"].to_numpy(), zones["lat"].to_numpy())
    np.save(out / "distance_provisional.npy", dist)

    partition_files: Dict[str, str] = {}
    if write_purpose_period_partitions:
        part_dir = out / "partitions"
        part_dir.mkdir(exist_ok=True)
        purposes = ("to-work", "to-home", "to-other")
        periods = ("4:00-12:00", "12:00-20:00", "20:00-0:00", "0:00-4:00")
        for purpose in purposes:
            for period in periods:
                label = f"{purpose}__{period.replace(':', '').replace('-', '_')}"
                plong = build_partition_long(
                    csv_path,
                    zone_ids,
                    partition_label=label,
                    chunksize=chunksize,
                    delimiter=delimiter,
                    count_col=count_col,
                    weekdays_only=True,
                    part_of_day=period,
                    trip_purpose=purpose,
                )
                ppath = write_long_table(plong, part_dir / f"{label}.parquet")
                partition_files[label] = ppath

    proxy_definition = {
        "name": "R_HBW_proxy",
        "weekdays_only": True,
        "PartOfDay": MORNING_PART_CANONICAL,
        "TripPurpose": PURPOSE_TO_WORK,
        "note": (
            "Strict 07–10 morning HBW is unavailable in Teralytics HK PartOfDay bins; "
            "proxy uses morning bin 4:00-12:00 + to-work + weekdays."
        ),
        "inclusion_note": "strict 07-10 HBW unavailable",
    }

    qa = {
        "proxy_definition": proxy_definition,
        "crs_note": "zone centroids lon/lat from DBF; CRS WGS84 (from .prj)",
        "distance_label": "provisional_wgs84_not_metric",
        "distance_method": "equirectangular_metres_mean_earth_radius",
        "d_ii": "nan — approximate zone area deferred",
        "n_zones_shapefile": int(len(zones)),
        "zone_ids": zone_ids,
        "csv_ids_unmatched": unmatched,
        "flow_sum_source_approx": total_meta["flow_sum_source_approx"],
        "flow_sum_total_matrix": float(total_mat.sum()),
        "flow_sum_proxy_matrix": float(proxy_mat.sum()),
        "total_aggregation": total_meta["total_aggregation"],
        "saw_all_preaggregates": total_meta["saw_all_preaggregates"],
        "source_dates": total_meta["source_dates"],
        "analysis_dates": total_meta["analysis_dates"],
        "columns_detected": columns_detected,
        "count_column": count_col,
        "delimiter": delimiter,
        "inclusion_note": "strict 07-10 HBW unavailable; R_HBW_proxy uses 4:00-12:00 + to-work + weekdays",
        "fid_missing_in_shapefile": ["9"] if "9" in unmatched else [u for u in unmatched],
    }
    (out / "qa.json").write_text(json.dumps(qa, indent=2) + "\n")

    manifest = {
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "dbf": str(dbf_path),
        "dbf_sha256": sha256_file(dbf_path),
        "outputs": {
            "zones": str(out / "zones.csv"),
            "long_total": total_path,
            "long_R_HBW_proxy": proxy_path,
            "total_npy": str(out / "total.npy"),
            "R_HBW_proxy_npy": str(out / "R_HBW_proxy.npy"),
            "distance_provisional_npy": str(out / "distance_provisional.npy"),
            "qa": str(out / "qa.json"),
            "partitions": partition_files,
        },
        "n_zones": len(zone_ids),
        "chunksize": chunksize,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"qa": qa, "manifest": manifest, "zones": zones}
