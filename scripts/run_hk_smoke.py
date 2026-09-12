#!/usr/bin/env python3
"""HK vertical smoke: Condition S + one classic Condition G + O_i^T QA (Ticket 1).

HK outputs ALWAYS set verification_only=true, paper_mainline=false.
Provisional distances and FID 9 quarantine are recorded; no paper conclusions.
"""
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

from attrOD.condition_g.constraints import audit_constraints
from attrOD.condition_g.generators import predictions_to_frame, run_registered_generators
from attrOD.condition_s import run_condition_s_full
from attrOD.data.manifest import build_run_manifest, write_json
from attrOD.data.readiness import (
    PROVISIONAL_DISTANCE,
    quarantine_hk_fid9,
    validate_metric_distance,
)
from attrOD.metrics.condition_s import day_bootstrap_frame
from attrOD.models.gravity import GravityModel
from attrOD.partitions.condition_s import audit_partition_disjointness, build_partitions
from attrOD.reporting.io import write_frame


def _load_hk(materialised: Path):
    R = np.load(materialised / "R_HBW_proxy.npy")
    T = np.load(materialised / "total.npy")
    d = np.load(materialised / "distance_provisional.npy")
    zones = pd.read_csv(materialised / "zones.csv")
    qa = json.loads((materialised / "qa.json").read_text())
    return R, T, d, zones, qa


def main() -> None:
    p = argparse.ArgumentParser(description="HK AttrOD smoke (verification-only)")
    p.add_argument("--config", default="configs/study_areas/hk_teralytics.yaml")
    p.add_argument("--data-root", default="/workspace/AttrOD/data",
                   help="use ~/AttrOD/data on myserver; /workspace/... is backup")
    p.add_argument("--hk-materialised", default="outputs/hk_teralytics")
    p.add_argument("--out", default="outputs/hk_smoke")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap-resamples", type=int, default=50,
                   help="use 1000 on server; default 50 for quick smoke")
    p.add_argument("--allow-backup", action="store_true", default=True)
    p.add_argument("--run-id", default="hk_smoke")
    p.add_argument("--classic-model", default="Gravity_power",
                   help="one classic Condition G generator for the vertical slice")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) if Path(args.config).exists() else {}
    out = Path(args.out)
    cs_out = out / "condition_s"
    cg_out = out / "condition_g" / args.classic_model
    cs_out.mkdir(parents=True, exist_ok=True)
    cg_out.mkdir(parents=True, exist_ok=True)

    mat_dir = Path(args.hk_materialised)
    R, T, distances, zones, qa = _load_hk(mat_dir)
    # Provisional WGS84 distances have NaN diagonal (d_ii deferred). For smoke
    # numeric stability only, fill NaN/Inf with a tiny positive length.
    distances = np.asarray(distances, dtype=float).copy()
    distances[~np.isfinite(distances)] = 1.0
    np.fill_diagonal(distances, np.maximum(np.diag(distances), 1.0))
    zone_ids = [str(z) for z in zones["fid"].tolist()] if "fid" in zones.columns else [str(i) for i in range(R.shape[0])]

    missing = qa.get("fid_missing_in_shapefile") or ["9"]
    fid_qa = quarantine_hk_fid9(zone_ids, missing_fids=missing, matrix=R)
    dist_gate = validate_metric_distance(
        distance_status=qa.get("distance_label", PROVISIONAL_DISTANCE),
        distances=distances,
        crs="EPSG:4326",
    )

    # Synthetic day stack from analysis_dates length if only aggregate matrices exist:
    # split total mass into pseudo-days for interface test when D from qa >= 2 dates.
    analysis_dates = qa.get("analysis_dates") or []
    T_by_day = R_by_day = None
    if len(analysis_dates) >= 2:
        # Interface-only: replicate aggregate with tiny noise so bootstrap path runs;
        # NOT a claim about true day OD (verification_only).
        rng = np.random.default_rng(args.seed)
        D = len(analysis_dates)
        T_by_day = [np.maximum(T / D + rng.normal(0, 1e-6, T.shape), 0.0) for _ in range(D)]
        R_by_day = [np.maximum(R / D + rng.normal(0, 1e-6, R.shape), 0.0) for _ in range(D)]
        # renormalize so sums match
        for i in range(D):
            if T_by_day[i].sum() > 0:
                T_by_day[i] *= T.sum() / D / T_by_day[i].sum()
            if R_by_day[i].sum() > 0:
                R_by_day[i] *= R.sum() / D / R_by_day[i].sum()

    part_index = build_partitions(
        n_zones=R.shape[0],
        partition_config={"include_full": True, "purpose_time": ["full_proxy"]},
        zone_ids=zone_ids,
        matrices={"full": T},
    )
    part_audit = audit_partition_disjointness(part_index)

    cs = run_condition_s_full(
        T, R,
        distances=distances,
        T_by_day=T_by_day,
        R_by_day=R_by_day,
        n_bootstrap=args.bootstrap_resamples,
        seed=args.seed,
        study_area="hk_teralytics",
        partition="total_vs_R_HBW_proxy",
    )
    write_json(cs_out / "metrics.json", {
        **cs["metrics"],
        "verification_only": True,
        "paper_mainline": False,
        "distance_status": dist_gate["distance_status"],
        "fid_quarantine": fid_qa,
    })
    write_frame(cs_out / "partition_metrics.parquet", pd.DataFrame([cs["metrics_row"]]))
    if cs.get("bootstrap"):
        write_frame(cs_out / "day_bootstrap.parquet", day_bootstrap_frame(cs["bootstrap"]))
        write_json(cs_out / "day_bootstrap_summary.json", cs["bootstrap"]["bootstrap_summary"])
    write_json(cs_out / "qa.json", {
        "verification_only": True,
        "paper_mainline": False,
        "independence_null": {k: v for k, v in (cs.get("independence_null") or {}).items() if k != "perm_null"},
        "partition_audit": part_audit,
        "fid_quarantine": fid_qa,
        "distance_gate": dist_gate,
        "note": "HK smoke — schema/partition/numeric stability only; not paper evidence",
    })

    # One classic Condition G generator (Gravity_power) + constraint QA
    attractiveness = np.maximum(R.sum(0), 1e-6)  # destination mass proxy
    train = list(range(R.shape[0]))
    g = GravityModel(deterrence="power")
    g.fit(R, distances, attractiveness, train_origins=train)
    O_T = T.sum(1)
    T_hat = g.emit(O_T)
    cqa = audit_constraints(T_hat, O_T, zone_ids=zone_ids)
    from attrOD.metrics.core import score_pair
    scores = score_pair(T_hat, T, distances)
    write_frame(cg_out / "predictions.parquet", predictions_to_frame(T_hat, zone_ids))
    write_json(cg_out / "metrics.json", {
        **scores,
        "model": args.classic_model,
        "verification_only": True,
        "paper_mainline": False,
    })
    write_json(cg_out / "constraint_qa.json", {
        **cqa,
        "verification_only": True,
        "paper_mainline": False,
        "hbw_only_fit": True,
    })

    manifest = build_run_manifest(
        run_id=args.run_id,
        dataset="hk",
        data_root=args.data_root,
        output_root=out,
        config=cfg,
        code_root=_ROOT,
        input_files=[
            mat_dir / "R_HBW_proxy.npy",
            mat_dir / "total.npy",
            mat_dir / "qa.json",
        ],
        paper_mainline=False,
        verification_only=True,
        distance_status=dist_gate["distance_status"],
        missing_fids=missing,
        allow_backup_data_root=args.allow_backup,
        extra={
            "hk_materialised": str(mat_dir),
            "classic_model": args.classic_model,
            "partition_index": part_index,
        },
    )
    write_json(out / "run_manifest.json", manifest)
    print(json.dumps({
        "out": str(out),
        "verification_only": True,
        "paper_mainline": False,
        "CPC_R": cs["metrics"].get("CPC_R"),
        "Delta": cs["metrics"].get("Delta"),
        "constraint_ok": cqa.get("ok"),
        "fid_quarantine_ok": fid_qa.get("ok"),
    }, indent=2))


if __name__ == "__main__":
    main()
