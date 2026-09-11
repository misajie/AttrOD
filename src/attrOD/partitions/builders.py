"""Partition alphabet and helpers.

Purpose × time coarse: {work, home, other} × {AM, PM, night, overnight}.
Primary targets vs R: other-purpose AM; home-bound PM; home-bound AM.
Length bands never re-cut; cells outside band are zero (change of support).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PERIOD_BINS_MITMA: Dict[str, Tuple[int, ...]] = {
    "AM": (7, 8, 9),
    "PM": (16, 17, 18, 19),
    "night": (20, 21, 22, 23),
    "overnight": (0, 1, 2, 3, 4, 5, 6),
}
# Hours 10–16 are inter-peak — not merged into AM/PM

PURPOSE_COARSE = ("work", "home", "other")

MITMA_LENGTH_BANDS = ("0.5-2", "2-10", "10-50", ">50")


def collapse_activity_other(activity: str) -> str:
    """Table 2 collapse: frequent + infrequent → other; keep distinct in Table 5."""
    a = str(activity).lower()
    if a in {"home", "casa"}:
        return "home"
    if a in {"work_or_study", "trabajo_estudio", "work", "lavoro"}:
        return "work"
    if a in {"frequent_activity", "frecuente", "infrequent_activity", "no_frecuente", "other"}:
        return "other"
    return "other"


def partition_labels_purpose_time(
    purpose: str,
    period: str,
) -> str:
    return f"{purpose}|{period}"


def length_band_mask(
    distances: np.ndarray,
    band: str,
) -> np.ndarray:
    """Boolean mask for cells whose Euclidean d_ij falls in the vendor band.

    Bands (km): 0.5–2, 2–10, 10–50, >50. Trips <0.5 km absent in MITMA;
    intra-zonal cells zeroed if d_ii outside band.
    """
    d = np.asarray(distances, dtype=float)
    if band == "0.5-2":
        return (d >= 0.5) & (d < 2.0)
    if band == "2-10":
        return (d >= 2.0) & (d < 10.0)
    if band == "10-50":
        return (d >= 10.0) & (d < 50.0)
    if band == ">50":
        return d >= 50.0
    raise ValueError(f"unknown length band: {band}")


def apply_length_band(mat: np.ndarray, distances: np.ndarray, band: str) -> np.ndarray:
    m = mat.copy()
    m[~length_band_mask(distances, band)] = 0.0
    return m


def period_from_hour(hour: int, endpoints: Optional[Dict[str, Sequence[int]]] = None) -> Optional[str]:
    bins = endpoints or PERIOD_BINS_MITMA
    for name, hours in bins.items():
        if int(hour) in set(hours):
            return name
    return "interpeak"  # 10–16
