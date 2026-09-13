#!/usr/bin/env python3
"""P2: full MITMA Condition S (draft2) — T partitions vs morning HBW R.

Expects materialised matrices (Server Sync produces these on the host):
  <materialised>/R.npy
  <materialised>/distances.npy          # metric_projected D
  <materialised>/distance_qa.json       # optional; should show metric_projected
  <materialised>/T_<partition>.npy      # one per published cut (see --partitions-yaml)
  <materialised>/T_by_day/*.npy         # optional day stack for T (weekday dates)
  <materialised>/R_by_day/*.npy         # optional day stack for R
  <materialised>/zone_ids.json          # optional

Outputs under outputs/mainline/mitma/condition_s/ (default):
  metrics.json, partition_metrics.parquet, day_bootstrap.parquet (if D>=2),
  qa.json, partition_index.json, partition_audit.json, run_manifest.json,
  per-partition/*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from attrOD.condition_s import run_condition_s_full
from attrOD.data.manifest import build_run_manifest, write_json
from attrOD.data.readiness import METRIC_DISTANCE, PROVISIONAL_DISTANCE
from attrOD.metrics.condition_s import day_bootstrap_frame
from attrOD.partitions.condition_s import (
    PRIMARY_PURPOSE_TIME,
    audit_partition_disjointness,
    build_partitions,
)
from attrOD.reporting.io import write_frame


def _safe_name(label: str) -> str:
    return (
        str(label)
        .replace("|", "__")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _load_day_stack(folder: Path) -> Optional[List[np.ndarray]]:
    if not folder.exists():
        return None
    files = sorted(folder.glob("*.npy"))
    if len(files) < 2:
        return None
    return [np.load(f) for f in files]


def _discover_T_matrices(mat_dir: Path, labels: Sequence[str]) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for lab in labels:
        candidates = [
            mat_dir / f"T_{_safe_name(lab)}.npy",
            mat_dir / f"T_{lab}.npy",
            mat_dir / lab / "T.npy",
        ]
        for c in candidates:
            if c.exists():
                found[lab] = c
                break
    # also accept any T_*.npy not in list as extra partitions
    for p in sorted(mat_dir.glob("T_*.npy")):
        stem = p.stem[2:]  # strip T_
        lab = stem.replace("__", "|")
        if lab not in found and stem not in found:
            # prefer mapped label if stem matches safe name of a known label
            matched = None
            for known in labels:
                if _safe_name(known) == stem:
                    matched = known
                    break
            found[matched or lab] = p
    return found


def main() -> None:
    p = argparse.ArgumentParser(description="P2 MITMA Condition S (draft2)")
    p.add_argument("--materialised", required=True, help="dir with R.npy, distances.npy, T_*.npy")
    p.add_argument("--partitions-yaml", default="configs/condition_s/draft2_partitions.yaml")
    p.add_argument("--out", default="outputs/mainline/mitma/condition_s")
    p.add_argument("--data-root", default="~/AttrOD/data")
    p.add_argument("--allow-backup", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="CPU processes for day-bootstrap (0 = all cores); partitions stay sequential",
    )
    p.add_argument("--study-area", default="mitma")
    p.add_argument(
        "--require-metric-distance",
        action="store_true",
        default=True,
        help="refuse paper_mainline if distance_qa is not metric_projected",
    )
    p.add_argument("--paper-mainline", action="store_true", default=True)
    p.add_argument("--verification-only", action="store_true")
    args = p.parse_args()

    mat_dir = Path(args.materialised)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    part_dir = out / "partitions"
    part_dir.mkdir(parents=True, exist_ok=True)

    R = np.load(mat_dir / "R.npy")
    dist_path = mat_dir / "distances.npy"
    if not dist_path.exists():
        dist_path = mat_dir / "distance.npy"
    distances = np.load(dist_path)

    dist_qa: Dict[str, Any] = {}
    for qa_name in ("distance_qa.json", "distances_qa.json"):
        qap = mat_dir / qa_name
        if qap.exists():
            dist_qa = json.loads(qap.read_text(encoding="utf-8"))
            break
    distance_status = dist_qa.get("distance_status") or (
        METRIC_DISTANCE
        if args.require_metric_distance
        else PROVISIONAL_DISTANCE
    )
    if args.require_metric_distance and distance_status != METRIC_DISTANCE:
        # still allow run as verification_only
        pass

    metric_ok = distance_status == METRIC_DISTANCE
    verification_only = bool(args.verification_only) or (not metric_ok)
    paper_mainline = bool(args.paper_mainline) and metric_ok and not verification_only

    cfg = {}
    part_yaml = Path(args.partitions_yaml)
    if part_yaml.exists():
        cfg = yaml.safe_load(part_yaml.read_text(encoding="utf-8")) or {}
    labels: List[str] = []
    if cfg.get("include_full", True):
        labels.append("full")
    labels.extend([str(x) for x in cfg.get("purpose_time", list(PRIMARY_PURPOSE_TIME))])
    labels.extend([f"length|{b}" for b in cfg.get("length_bands", [])])
    labels.extend([str(x) for x in cfg.get("extra", [])])
    # de-dupe
    seen = set()
    ordered = []
    for lab in labels:
        if lab not in seen:
            seen.add(lab)
            ordered.append(lab)

    T_paths = _discover_T_matrices(mat_dir, ordered)
    if "full" in ordered and "full" not in T_paths:
        # convention: total.npy or T.npy as full
        for alt in ("total.npy", "T.npy", "T_full.npy"):
            if (mat_dir / alt).exists():
                T_paths["full"] = mat_dir / alt
                break

    if not T_paths:
        raise SystemExit(
            f"no T_*.npy matrices found under {mat_dir}; "
            f"expected partitions {ordered}"
        )

    zone_ids = None
    zpath = mat_dir / "zone_ids.json"
    if zpath.exists():
        zone_ids = json.loads(zpath.read_text(encoding="utf-8"))

    T_by_day = _load_day_stack(mat_dir / "T_by_day")
    R_by_day = _load_day_stack(mat_dir / "R_by_day")
    # only use bootstrap when both stacks present and aligned length
    if T_by_day is not None and R_by_day is not None and len(T_by_day) != len(R_by_day):
        T_by_day = R_by_day = None

    from attrOD.runs.ledger import new_run_id

    run_id = args.run_id or new_run_id(prefix="mitma_cs")

    rows = []
    null_summaries = {}
    boot_summaries = {}
    matrices_for_index = {}

    for lab, tpath in sorted(T_paths.items(), key=lambda kv: ordered.index(kv[0]) if kv[0] in ordered else 10_000):
        T = np.load(tpath)
        if T.shape != R.shape:
            raise SystemExit(f"shape mismatch {lab}: T{T.shape} vs R{R.shape}")
        matrices_for_index[lab] = T
        # day bootstrap only meaningful for full / aggregate T unless per-partition day stacks exist
        use_days = T_by_day if lab in ("full", "total") else None
        use_r_days = R_by_day if use_days is not None else None
        payload = run_condition_s_full(
            T,
            R,
            distances=distances,
            T_by_day=use_days,
            R_by_day=use_r_days,
            n_bootstrap=args.bootstrap_resamples,
            seed=args.seed,
            n_jobs=args.n_jobs,
            study_area=args.study_area,
            partition=lab,
        )
        metrics = {
            **payload["metrics"],
            "partition": lab,
            "verification_only": verification_only,
            "paper_mainline": paper_mainline,
            "distance_status": distance_status,
        }
        row = dict(payload["metrics_row"])
        row["verification_only"] = verification_only
        row["paper_mainline"] = paper_mainline
        rows.append(row)

        sub = part_dir / _safe_name(lab)
        sub.mkdir(parents=True, exist_ok=True)
        write_json(sub / "metrics.json", metrics)
        write_json(
            sub / "qa.json",
            {
                "partition": lab,
                "independence_null": payload.get("independence_null"),
                "verification_only": verification_only,
                "paper_mainline": paper_mainline,
                "distance_status": distance_status,
            },
        )
        null_summaries[lab] = payload.get("independence_null")
        if payload.get("bootstrap"):
            write_frame(sub / "day_bootstrap.parquet", day_bootstrap_frame(payload["bootstrap"]))
            boot_summaries[lab] = payload["bootstrap"].get("bootstrap_summary")

    pindex = build_partitions(
        n_zones=R.shape[0],
        partition_config=cfg,
        zone_ids=zone_ids,
        matrices=matrices_for_index,
        distances=distances,
    )
    audit = audit_partition_disjointness(pindex)

    write_json(out / "metrics.json", {
        "study_area": args.study_area,
        "run_id": run_id,
        "n_partitions": len(rows),
        "partitions": [r.get("partition") for r in rows],
        "verification_only": verification_only,
        "paper_mainline": paper_mainline,
        "distance_status": distance_status,
        "by_partition": {r.get("partition"): r for r in rows},
    })
    write_frame(out / "partition_metrics.parquet", pd.DataFrame(rows))
    write_json(out / "qa.json", {
        "independence_null": null_summaries,
        "bootstrap_summary": boot_summaries,
        "distance_qa": dist_qa,
        "verification_only": verification_only,
        "paper_mainline": paper_mainline,
        "note": "Condition S only — no generators in P2",
    })
    write_json(out / "partition_index.json", pindex)
    write_json(out / "partition_audit.json", audit)

    # optional top-level bootstrap from full partition
    if "full" in boot_summaries:
        # already written per-partition; also mirror summary
        write_json(out / "day_bootstrap_summary.json", boot_summaries["full"])

    try:
        man = build_run_manifest(
            run_id=run_id,
            dataset=args.study_area,
            data_root=args.data_root,
            output_root=out,
            config={
                "partitions_yaml": str(part_yaml),
                "materialised": str(mat_dir),
                "bootstrap_resamples": args.bootstrap_resamples,
        "n_jobs": args.n_jobs,
                "seed": args.seed,
            },
            code_root=_ROOT,
            paper_mainline=paper_mainline,
            verification_only=verification_only,
            distance_status=distance_status,
            allow_backup_data_root=args.allow_backup,
            extra={"ticket": "P2", "condition": "S"},
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
                "paper_mainline": paper_mainline,
                "verification_only": verification_only,
                "distance_status": distance_status,
                "bootstrap_partitions": list(boot_summaries),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
