"""Flow long table: origin, destination, flow, partition, date(nullable).

Equivalent in spirit to a scikit-mobility FlowDataFrame (draft2 Implementation notes).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

FLOW_COLUMNS = ("origin", "destination", "flow", "partition", "date")


class FlowTable:
    """Thin wrapper around a validated long-format OD DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = assert_flow_schema(df)

    def __len__(self) -> int:
        return len(self.df)

    def filter_partition(self, partition: str) -> "FlowTable":
        return FlowTable(self.df.loc[self.df["partition"] == partition].copy())

    def totals(self) -> float:
        return float(self.df["flow"].sum())


def assert_flow_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FLOW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Flow table missing columns: {missing}")
    out = df.loc[:, list(FLOW_COLUMNS)].copy()
    out["origin"] = out["origin"].astype(str)
    out["destination"] = out["destination"].astype(str)
    out["partition"] = out["partition"].astype(str)
    out["flow"] = pd.to_numeric(out["flow"], errors="coerce").fillna(0.0).astype(float)
    # MITMA fractional n_trips below 1e-6 treated as zero (draft2)
    out.loc[out["flow"].abs() < 1e-6, "flow"] = 0.0
    if "date" not in out.columns:
        out["date"] = pd.NaT
    return out.reset_index(drop=True)


def long_to_matrix(
    df: pd.DataFrame,
    zones: Sequence[str],
    partition: Optional[str] = None,
    date: Optional[object] = None,
) -> np.ndarray:
    """Aggregate long table onto an N×N dense matrix ordered by `zones`."""
    d = assert_flow_schema(df)
    if partition is not None:
        d = d.loc[d["partition"] == partition]
    if date is not None:
        d = d.loc[d["date"] == date]
    n = len(zones)
    idx = {z: i for i, z in enumerate(zones)}
    mat = np.zeros((n, n), dtype=float)
    if d.empty:
        return mat
    g = d.groupby(["origin", "destination"], as_index=False)["flow"].sum()
    for o, dest, f in g.itertuples(index=False):
        i = idx.get(str(o))
        j = idx.get(str(dest))
        if i is not None and j is not None:
            mat[i, j] += float(f)
    return mat


def matrix_to_long(
    mat: np.ndarray,
    zones: Sequence[str],
    partition: str,
    date: Optional[object] = None,
    drop_zeros: bool = False,
) -> pd.DataFrame:
    n = len(zones)
    if mat.shape != (n, n):
        raise ValueError(f"matrix shape {mat.shape} != ({n},{n})")
    rows = []
    for i in range(n):
        for j in range(n):
            f = float(mat[i, j])
            if drop_zeros and f == 0.0:
                continue
            rows.append(
                {
                    "origin": str(zones[i]),
                    "destination": str(zones[j]),
                    "flow": f,
                    "partition": partition,
                    "date": date,
                }
            )
    return assert_flow_schema(pd.DataFrame(rows))


def row_sums(mat: np.ndarray) -> np.ndarray:
    return mat.sum(axis=1)


def col_sums(mat: np.ndarray) -> np.ndarray:
    return mat.sum(axis=0)


def h_intra(mat: np.ndarray) -> float:
    total = float(mat.sum())
    if total <= 0:
        return float("nan")
    return float(np.trace(mat) / total)
