"""Regione Lombardia Matrice OD2016 passeggeri adapter.

Reference R: lavoro in the morning time band → HBW.
studio is always held out. No calendar day stack → Condition-S day bootstrap undefined.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

from .flow_table import FLOW_COLUMNS, assert_flow_schema

# Motives as released
MOTIVE_WORK = {"lavoro", "work"}
MOTIVE_STUDY = {"studio", "study"}
MOTIVE_HOME = {"rientri a casa", "rientri_a_casa", "home"}
MOTIVE_OTHER = {"occasionali", "affari", "occasional", "business"}


class LombardyAdapter:
    def __init__(self, morning_bands: Sequence[str] = ("mattina", "AM", "7-10", "07-10")):
        self.morning_bands = {b.lower() for b in morning_bands}

    def to_long(
        self,
        raw: pd.DataFrame,
        *,
        partition: str,
        zone_ids: Optional[Iterable[str]] = None,
        flow_col: str = "flow",
    ) -> pd.DataFrame:
        df = raw.copy()
        for a, b in [("ORIGINE", "origin"), ("DESTINAZIONE", "destination"), ("origin_id", "origin")]:
            if a in df.columns and b not in df.columns:
                df = df.rename(columns={a: b})
        if flow_col not in df.columns:
            # try common names
            for c in ("TRIPS", "n_trips", "VALORE", "flow"):
                if c in df.columns:
                    flow_col = c
                    break
        out = pd.DataFrame(
            {
                "origin": df["origin"].astype(str),
                "destination": df["destination"].astype(str),
                "flow": pd.to_numeric(df[flow_col], errors="coerce").fillna(0.0),
                "partition": partition,
                "date": pd.NaT,  # no calendar stack
            }
        )
        if zone_ids is not None:
            z = set(map(str, zone_ids))
            out = out.loc[out["origin"].isin(z) & out["destination"].isin(z)]
        return assert_flow_schema(out)


def build_reference_R_lombardy(
    raw: pd.DataFrame,
    *,
    morning_bands: Sequence[str] = ("mattina", "AM", "7-10", "07-10"),
    zone_ids: Optional[Iterable[str]] = None,
    motive_col: str = "motive",
    time_col: str = "FASCIA_ORARIA",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (R_long, study_held_out_long)."""
    df = raw.copy()
    for a, b in [
        ("MOTIVO", "motive"),
        ("motivo", "motive"),
        ("FASCIA_ORARIA", "FASCIA_ORARIA"),
        ("fascia_oraria", "FASCIA_ORARIA"),
    ]:
        if a in df.columns and (b not in df.columns or a == b):
            df = df.rename(columns={a: b})
    if motive_col not in df.columns:
        motive_col = "motive"
    if time_col not in df.columns and "FASCIA_ORARIA" in df.columns:
        time_col = "FASCIA_ORARIA"
    morning = {b.lower() for b in morning_bands}
    tvals = df[time_col].astype(str).str.lower() if time_col in df.columns else pd.Series("am", index=df.index)
    mvals = df[motive_col].astype(str).str.lower()
    is_morning = tvals.isin(morning) | tvals.str.contains("matt|am|07|7-")
    work = mvals.isin(MOTIVE_WORK) & is_morning
    study = mvals.isin(MOTIVE_STUDY)
    adapter = LombardyAdapter(morning_bands=morning_bands)
    R = adapter.to_long(df.loc[work], partition="R_HBW_AM", zone_ids=zone_ids)
    S = adapter.to_long(df.loc[study], partition="study_heldout", zone_ids=zone_ids)
    R = assert_flow_schema(R.groupby(["origin", "destination", "partition", "date"], dropna=False, as_index=False)["flow"].sum())
    S = assert_flow_schema(S.groupby(["origin", "destination", "partition", "date"], dropna=False, as_index=False)["flow"].sum())
    return R, S
