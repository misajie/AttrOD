#!/usr/bin/env python3
"""AttrOD CLI — see README for Server Bot recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from attrOD.data.flow_table import assert_flow_schema, long_to_matrix, matrix_to_long
from attrOD.data.mitma import build_reference_R_mitma
from attrOD.data.lombardy import build_reference_R_lombardy


def main():
    p = argparse.ArgumentParser(description="Build long-table partitions + reference R")
    p.add_argument("--config", required=True, help="study-area YAML")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--raw", default=None, help="optional path to pre-extracted parquet/csv flows")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.txt").write_text(
        "Partitions stub. Provide --raw vendor extract to materialise long tables.\n"
        f"study_area_id={cfg.get('study_area_id')}\n"
        f"product={cfg.get('product')}\n"
        "No data is downloaded by this script.\n"
    )
    # If raw provided, attempt R build
    if args.raw:
        path = Path(args.raw)
        if path.suffix == ".parquet":
            raw = pd.read_parquet(path)
        else:
            raw = pd.read_csv(path)
        product = cfg.get("product", "")
        hours = tuple(cfg.get("hour_endpoints", {}).get("AM", [7, 8, 9]))
        zones = cfg.get("zone_ids") or None
        if product.startswith("mitma"):
            R, study = build_reference_R_mitma(raw, hour_am=hours, zone_ids=zones)
        elif product.startswith("lombardy"):
            R, study = build_reference_R_lombardy(raw, zone_ids=zones)
        else:
            raise SystemExit(f"unknown product: {product}")
        R.to_parquet(out / "R_HBW_AM.parquet", index=False)
        study.to_parquet(out / "study_partition.parquet", index=False)
        print(f"wrote R rows={len(R)} study rows={len(study)}")
    else:
        print(f"wrote stub under {out} (no --raw)")


if __name__ == "__main__":
    main()
