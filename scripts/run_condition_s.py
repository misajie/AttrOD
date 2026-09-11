#!/usr/bin/env python3
"""AttrOD CLI — see README for Server Bot recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from attrOD.condition_s import score_condition_s, bootstrap_days
from attrOD.reporting.schemas import empty_table


def main():
    p = argparse.ArgumentParser(description="Run Condition S")
    p.add_argument("--config", required=True)
    p.add_argument("--partitions", required=True, help="dir with T_*.npy / R.npy or parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--T", default=None, help="optional path to T.npy")
    p.add_argument("--R", default=None, help="optional path to R.npy")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    part = Path(args.partitions)
    # Prefer explicit matrices for smoke / server runs
    if args.T and args.R:
        T = np.load(args.T)
        R = np.load(args.R)
        dist = None
        dpath = part / "distances.npy"
        if dpath.exists():
            dist = np.load(dpath)
        scores = score_condition_s(T, R, dist)
        pd.DataFrame([scores]).to_csv(out / "condition_s_scores.csv", index=False)
        print(json.dumps(scores, indent=2))
        return
    # Stub empty tables matching schema
    for name in ("table2_purpose_time", "table3_day_sex_age", "table4_length_bands", "table5_additional"):
        empty_table(name).to_csv(out / f"{name}.csv", index=False)
    meta = {
        "study_area_id": cfg.get("study_area_id"),
        "bootstrap_resamples": cfg.get("condition_s", {}).get("bootstrap_resamples", 1000),
        "note": "No matrices found; wrote empty table schemas. Pass --T/--R for scoring.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
