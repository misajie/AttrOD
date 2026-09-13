#!/usr/bin/env python3
"""P5a: Lombardy Condition S (draft2) — no day bootstrap.

Same scoring contract as MITMA Condition S; calendar day bootstrap is undefined
for Lombardy (no day stack).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from attrOD.condition_s import run_condition_s_full
from attrOD.data.manifest import build_run_manifest, write_json
from attrOD.data.readiness import METRIC_DISTANCE
from attrOD.partitions.condition_s import audit_partition_disjointness, build_partitions
from attrOD.reporting.io import write_frame
from attrOD.runs.ledger import new_run_id


def _safe_name(label: str) -> str:
    return str(label).replace("|", "__").replace("/", "_").replace(" ", "_")


def main() -> None:
    p = argparse.ArgumentParser(description="P5a Lombardy Condition S (no day bootstrap)")
    p.add_argument("--materialised", required=True)
    p.add_argument("--partitions-yaml", default="configs/condition_s/draft2_partitions.yaml")
    p.add_argument("--out", default="outputs/mainline/lombardy/condition_s")
    p.add_argument("--data-root", default="~/AttrOD/data")
    p.add_argument("--allow-backup", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--study-area", default="lombardy")
    p.add_argument("--paper-mainline", action="store_true", default=True)
    p.add_argument("--verification-only", action="store_true")
    args = p.parse_args()

    mat = Path(args.materialised)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    part_dir = out / "partitions"
    part_dir.mkdir(parents=True, exist_ok=True)

    R = np.load(mat / "R.npy")
    dist_path = mat / "distances.npy" if (mat / "distances.npy").exists() else mat / "distance.npy"
    distances = np.load(dist_path)
    dist_qa = {}
    for qa_name in ("distance_qa.json", "distances_qa.json"):
        if (mat / qa_name).exists():
            dist_qa = json.loads((mat / qa_name).read_text(encoding="utf-8"))
            break
    distance_status = dist_qa.get("distance_status", METRIC_DISTANCE)
    metric_ok = distance_status == METRIC_DISTANCE
    verification_only = bool(args.verification_only) or (not metric_ok)
    paper_mainline = bool(args.paper_mainline) and metric_ok and not verification_only

    cfg = {}
    py = Path(args.partitions_yaml)
    if py.exists():
        cfg = yaml.safe_load(py.read_text(encoding="utf-8")) or {}
    # Lombardy: no length-band defaults required; purpose/time may differ — use config as-is
    labels: List[str] = []
    if cfg.get("include_full", True):
        labels.append("full")
    labels.extend([str(x) for x in cfg.get("purpose_time", [])])
    labels.extend([f"length|{b}" for b in cfg.get("length_bands", [])])
    labels.extend([str(x) for x in cfg.get("extra", [])])
    if not any(l != "full" for l in labels):
        # minimal lombardy defaults if yaml empty of purpose_time
        labels = ["full"]

    T_paths: Dict[str, Path] = {}
    for lab in labels:
        for c in (mat / f"T_{_safe_name(lab)}.npy", mat / f"T_{lab}.npy"):
            if c.exists():
                T_paths[lab] = c
                break
    for alt in ("total.npy", "T.npy", "T_full.npy"):
        if "full" in labels and "full" not in T_paths and (mat / alt).exists():
            T_paths["full"] = mat / alt
    for pth in sorted(mat.glob("T_*.npy")):
        stem = pth.stem[2:].replace("__", "|")
        if stem not in T_paths:
            T_paths[stem] = pth

    if not T_paths:
        raise SystemExit(f"no T matrices under {mat}")

    zone_ids = None
    if (mat / "zone_ids.json").exists():
        zone_ids = json.loads((mat / "zone_ids.json").read_text(encoding="utf-8"))

    run_id = args.run_id or new_run_id(prefix="lombardy_cs")
    rows = []
    null_summaries = {}
    matrices = {}

    for lab, tpath in T_paths.items():
        T = np.load(tpath)
        matrices[lab] = T
        # Explicitly NO day bootstrap for Lombardy
        payload = run_condition_s_full(
            T,
            R,
            distances=distances,
            T_by_day=None,
            R_by_day=None,
            n_bootstrap=0,
            seed=args.seed,
            study_area=args.study_area,
            partition=lab,
        )
        metrics = {
            **payload["metrics"],
            "partition": lab,
            "verification_only": verification_only,
            "paper_mainline": paper_mainline,
            "distance_status": distance_status,
            "day_bootstrap": "undefined_no_calendar_stack",
        }
        row = dict(payload["metrics_row"])
        row["day_bootstrap"] = "undefined_no_calendar_stack"
        rows.append(row)
        sub = part_dir / _safe_name(lab)
        sub.mkdir(parents=True, exist_ok=True)
        write_json(sub / "metrics.json", metrics)
        write_json(
            sub / "qa.json",
            {
                "partition": lab,
                "independence_null": payload.get("independence_null"),
                "day_bootstrap": None,
                "note": "Lombardy has no calendar stack — day bootstrap omitted",
            },
        )
        null_summaries[lab] = payload.get("independence_null")

    pindex = build_partitions(
        n_zones=R.shape[0],
        partition_config=cfg,
        zone_ids=zone_ids,
        matrices=matrices,
        distances=distances,
    )
    write_frame(out / "partition_metrics.parquet", pd.DataFrame(rows))
    write_json(
        out / "metrics.json",
        {
            "study_area": args.study_area,
            "run_id": run_id,
            "n_partitions": len(rows),
            "partitions": [r.get("partition") for r in rows],
            "paper_mainline": paper_mainline,
            "day_bootstrap": "undefined_no_calendar_stack",
            "by_partition": {r.get("partition"): r for r in rows},
        },
    )
    write_json(
        out / "qa.json",
        {
            "ticket": "P5a",
            "independence_null": null_summaries,
            "day_bootstrap": None,
            "distance_qa": dist_qa,
            "note": "Condition S only; no day bootstrap for Lombardy",
        },
    )
    write_json(out / "partition_index.json", pindex)
    write_json(out / "partition_audit.json", audit_partition_disjointness(pindex))
    try:
        man = build_run_manifest(
            run_id=run_id,
            dataset=args.study_area,
            data_root=args.data_root,
            output_root=out,
            config={"seed": args.seed, "day_bootstrap": False},
            code_root=_ROOT,
            paper_mainline=paper_mainline,
            verification_only=verification_only,
            distance_status=distance_status,
            allow_backup_data_root=args.allow_backup,
            extra={"ticket": "P5a", "condition": "S"},
        )
        write_json(out / "run_manifest.json", man)
    except ValueError as e:
        write_json(out / "run_manifest_error.json", {"error": str(e)})

    print(
        json.dumps(
            {
                "run_id": run_id,
                "out": str(out),
                "n_partitions": len(rows),
                "partitions": [r.get("partition") for r in rows],
                "day_bootstrap": "undefined_no_calendar_stack",
                "paper_mainline": paper_mainline,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
