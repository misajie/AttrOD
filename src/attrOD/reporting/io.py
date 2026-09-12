"""JSON / parquet writers with stable schemas."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Union

import pandas as pd

PathLike = Union[str, Path]


def write_json(path: PathLike, obj: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    return p


def write_frame(path: PathLike, df: pd.DataFrame, *, prefer_parquet: bool = True) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if prefer_parquet and p.suffix in {".parquet", ""}:
        if p.suffix == "":
            p = p.with_suffix(".parquet")
        try:
            df.to_parquet(p, index=False)
            return p
        except Exception:
            p = p.with_suffix(".csv")
            df.to_csv(p, index=False)
            return p
    if p.suffix == ".csv":
        df.to_csv(p, index=False)
    else:
        df.to_parquet(p, index=False)
    return p
