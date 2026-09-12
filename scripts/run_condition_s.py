#!/usr/bin/env python3
"""AttrOD CLI — Condition S exact engine (Ticket 2)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from attrOD.condition_s import run_condition_s_full
from attrOD.data.manifest import build_run_manifest, write_json
from attrOD.metrics.condition_s import day_bootstrap_frame
from attrOD.partitions.condition_s import audit_partition_disjointness, build_partitions
from attrOD.reporting.io import write_frame
from attrOD.reporting.schemas import empty_table


def main():
    p = argparse.ArgumentParser(description="Run Condition S")
    p.add_argument("--config", required=True)
    p.add_argument("--partitions", required=True, help="dir with T_*.npy / R.npy or parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--T", default=None, help="optional path to T.npy")
    p.add_argument("--R", default=None, help="optional path to R.npy")
    p.add_argument("--distances", default=None)
    p.add_argument("--data-root", default="~/AttrOD/data")
    p.add_argument("--allow-backup", action="store_true")
    p.add_argument("--run-id", default="condition_s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap-resamples", type=int, default=None)
    p.add_argument("--verification-only", action="store_true")
    p.add_argument("--paper-mainline", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    part = Path(args.partitions)
    n_boot = args.bootstrap_resamples or cfg.get("condition_s", {}).get("bootstrap_resamples", 1000)

    verification_only = args.verification_only or (not args.paper_mainline)
    paper_mainline = bool(args.paper_mainline) and not verification_only

    if args.T and args.R:
        T = np.load(args.T)
        R = np.load(args.R)
        dist = np.load(args.distances) if args.distances else None
        if dist is None:
            dpath = part / "distances.npy"
            if dpath.exists():
                dist = np.load(dpath)
        payload = run_condition_s_full(
            T, R,
            distances=dist,
            n_bootstrap=n_boot,
            seed=args.seed,
            study_area=cfg.get("study_area_id", ""),
            partition="full",
        )
        write_json(out / "metrics.json", {
            **payload["metrics"],
            "verification_only": verification_only,
            "paper_mainline": paper_mainline,
        })
        write_frame(out / "partition_metrics.parquet", pd.DataFrame([payload["metrics_row"]]))
        if payload.get("bootstrap"):
            write_frame(out / "day_bootstrap.parquet", day_bootstrap_frame(payload["bootstrap"]))
        write_json(out / "qa.json", {
            "independence_null": payload.get("independence_null"),
            "verification_only": verification_only,
            "paper_mainline": paper_mainline,
        })
        pindex = build_partitions(n_zones=T.shape[0], matrices={"full": T})
        write_json(out / "partition_index.json", pindex)
        write_json(out / "partition_audit.json", audit_partition_disjointness(pindex))
        try:
            man = build_run_manifest(
                run_id=args.run_id,
                dataset=cfg.get("study_area_id", "unknown"),
                data_root=args.data_root,
                output_root=out,
                config=cfg,
                code_root=_ROOT,
                paper_mainline=paper_mainline,
                verification_only=verification_only,
                allow_backup_data_root=args.allow_backup,
            )
            write_json(out / "run_manifest.json", man)
        except ValueError as e:
            write_json(out / "run_manifest_error.json", {"error": str(e)})
        print(json.dumps(payload["metrics"], indent=2))
        return

    for name in ("table2_purpose_time", "table3_day_sex_age", "table4_length_bands", "table5_additional"):
        empty_table(name).to_csv(out / f"{name}.csv", index=False)
    meta = {
        "study_area_id": cfg.get("study_area_id"),
        "bootstrap_resamples": n_boot,
        "note": "No matrices found; wrote empty table schemas. Pass --T/--R for scoring.",
        "verification_only": verification_only,
        "paper_mainline": paper_mainline,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
