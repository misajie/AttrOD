"""MITMA / MITMS Estudio de movilidad de viajeros v2 adapter.

Reference R (morning HBW):
  - weekdays in analysis window
  - AM hours default 7,8,9
  - activity_origin = home AND activity_destination = work_or_study
  - if estudio_destino_posible → exclude from R, store as study partition
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

from .flow_table import FLOW_COLUMNS, assert_flow_schema

# Vendor activity aliases
HOME = {"home", "casa"}
WORK_OR_STUDY = {"work_or_study", "trabajo_estudio"}
FREQUENT = {"frequent_activity", "frecuente"}
INFREQUENT = {"infrequent_activity", "no_frecuente"}

LENGTH_BANDS = ("0.5-2", "2-10", "10-50", ">50")
AGE_BANDS = ("0-25", "25-45", "45-65", "65-100")


def _norm_activity(x: object) -> str:
    s = str(x).strip().lower()
    if s in HOME:
        return "home"
    if s in WORK_OR_STUDY:
        return "work_or_study"
    if s in FREQUENT:
        return "frequent_activity"
    if s in INFREQUENT:
        return "infrequent_activity"
    return s


class MitmaAdapter:
    """Map vendor daily-trip rows into the AttrOD flow long table + attribute columns."""

    def __init__(self, hour_am: Sequence[int] = (7, 8, 9)):
        self.hour_am = tuple(hour_am)

    def to_long(
        self,
        raw: pd.DataFrame,
        *,
        partition: str,
        zone_ids: Optional[Iterable[str]] = None,
        flow_col: str = "n_trips",
    ) -> pd.DataFrame:
        """Expect columns at least: origin, destination, n_trips, and optional date."""
        df = raw.copy()
        # flexible column aliases
        rename = {}
        for a, b in [
            ("origin_id", "origin"),
            ("destino", "destination"),
            ("destination_id", "destination"),
            ("origen", "origin"),
        ]:
            if a in df.columns and b not in df.columns:
                rename[a] = b
        df = df.rename(columns=rename)
        if flow_col not in df.columns:
            raise KeyError(f"flow column {flow_col!r} missing")
        out = pd.DataFrame(
            {
                "origin": df["origin"].astype(str),
                "destination": df["destination"].astype(str),
                "flow": pd.to_numeric(df[flow_col], errors="coerce").fillna(0.0),
                "partition": partition,
                "date": df["date"] if "date" in df.columns else pd.NaT,
            }
        )
        if zone_ids is not None:
            z = set(map(str, zone_ids))
            out = out.loc[out["origin"].isin(z) & out["destination"].isin(z)]
        return assert_flow_schema(out)


def build_reference_R_mitma(
    raw: pd.DataFrame,
    *,
    hour_am: Sequence[int] = (7, 8, 9),
    weekdays_only: bool = True,
    zone_ids: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (R_long, study_partition_long) from vendor rows.

    Rows with estudio_destino_posible true are excluded from R and returned as study.
    """
    df = raw.copy()
    # normalize columns
    colmap = {
        "hora": "hour",
        "activity_origin": "activity_origin",
        "activity_destination": "activity_destination",
        "actividad_origen": "activity_origin",
        "actividad_destino": "activity_destination",
        "estudio_destino_posible": "estudio_destino_posible",
        "n_trips": "n_trips",
        "origin": "origin",
        "destination": "destination",
        "origen": "origin",
        "destino": "destination",
    }
    for k, v in colmap.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    if "hour" not in df.columns:
        raise KeyError("MITMA raw needs hour")
    df = df.loc[df["hour"].isin(list(hour_am))]
    if weekdays_only and "date" in df.columns:
        dts = pd.to_datetime(df["date"], errors="coerce")
        df = df.loc[dts.dt.dayofweek < 5]
    ao = df["activity_origin"].map(_norm_activity) if "activity_origin" in df.columns else None
    ad = df["activity_destination"].map(_norm_activity) if "activity_destination" in df.columns else None
    if ao is None or ad is None:
        raise KeyError("activity_origin / activity_destination required")
    mask_hbw = (ao == "home") & (ad == "work_or_study")
    df = df.loc[mask_hbw].copy()
    study_flag = df.get("estudio_destino_posible")
    if study_flag is not None:
        is_study = study_flag.astype(str).str.lower().isin({"1", "true", "t", "yes"}) | (pd.to_numeric(study_flag, errors="coerce") == 1)
    else:
        is_study = pd.Series(False, index=df.index)
    study = df.loc[is_study]
    ref = df.loc[~is_study]
    adapter = MitmaAdapter(hour_am=hour_am)
    R = adapter.to_long(ref, partition="R_HBW_AM", zone_ids=zone_ids)
    S = adapter.to_long(study, partition="study_AM", zone_ids=zone_ids) if len(study) else assert_flow_schema(
        pd.DataFrame(columns=list(FLOW_COLUMNS))
    )
    # aggregate to od
    R = assert_flow_schema(R.groupby(["origin", "destination", "partition", "date"], dropna=False, as_index=False)["flow"].sum())
    if len(S):
        S = assert_flow_schema(S.groupby(["origin", "destination", "partition", "date"], dropna=False, as_index=False)["flow"].sum())
    return R, S
