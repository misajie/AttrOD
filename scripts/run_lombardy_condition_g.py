#!/usr/bin/env python3
"""P5b: Lombardy Condition G zero-shot (draft2) — fit only on HBW R."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from attrOD.condition_g.generators import (
    NOW2_CLASSIC_SIX,
    predictions_to_frame,
    run_registered_generators,
)
from attrOD.data.manifest import build_run_manifest, write_json
from attrOD.data.readiness import METRIC_DISTANCE
from attrOD.reporting.io import write_frame
from attrOD.runs.ledger import new_run_id


def main() -> None:
    p = argparse.ArgumentParser(description="P5b Lombardy Condition G zero-shot")
    p.add_argument("--materialised", required=True)
    p.add_argument("--T", default=None)
    p.add_argument("--out", default="outputs/mainline/lombardy/condition_g")
    p.add_argument("--models", default=None)
    p.add_argument("--include-deep", action="store_true")
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

    R = np.load(mat / "R.npy")
    dist_path = mat / "distances.npy" if (mat / "distances.npy").exists() else mat / "distance.npy"
    distances = np.load(dist_path)
    if args.T:
        T = np.load(args.T)
        t_label = Path(args.T).stem
    else:
        T = None
        t_label = "full"
        for name in ("T_full.npy", "total.npy", "T.npy"):
            if (mat / name).exists():
                T = np.load(mat / name)
                t_label = name.replace(".npy", "")
                break
        if T is None:
            raise SystemExit("need --T or T_full.npy under materialised")

    attr = np.load(mat / "attractiveness.npy") if (mat / "attractiveness.npy").exists() else np.maximum(R.sum(0), 1e-6)
    osm = np.load(mat / "osm_features.npy") if (mat / "osm_features.npy").exists() else None
    pop = np.load(mat / "population.npy") if (mat / "population.npy").exists() else attr

    dist_qa = {}
    for qa_name in ("distance_qa.json", "distances_qa.json"):
        if (mat / qa_name).exists():
            dist_qa = json.loads((mat / qa_name).read_text(encoding="utf-8"))
            break
    distance_status = dist_qa.get("distance_status", METRIC_DISTANCE)
    metric_ok = distance_status == METRIC_DISTANCE
    verification_only = bool(args.verification_only) or (not metric_ok)
    paper_mainline = bool(args.paper_mainline) and metric_ok and not verification_only

    zone_ids = None
    if (mat / "zone_ids.json").exists():
        zone_ids = json.loads((mat / "zone_ids.json").read_text(encoding="utf-8"))

    names = [x.strip() for x in args.models.split(",")] if args.models else list(NOW2_CLASSIC_SIX)
    if args.include_deep and "DeepGravity" not in names:
        names.append("DeepGravity")

    run_id = args.run_id or new_run_id(prefix="lombardy_cg")
    results = run_registered_generators(
        R,
        T,
        distances,
        attr,
        list(range(R.shape[0])),
        model_names=names,
        populations=pop,
        osm_features=osm,
        zone_ids=zone_ids,
        seed=args.seed,
    )

    rows = []
    for model, pack in results.items():
        model_dir = out / model
        model_dir.mkdir(parents=True, exist_ok=True)
        write_frame(model_dir / "predictions.parquet", predictions_to_frame(pack["T_hat"], zone_ids))
        meta = dict(pack.get("meta") or {})
        metrics = {
            **pack["metrics"],
            "model": model,
            "target_partition": t_label,
            "verification_only": verification_only,
            "paper_mainline": paper_mainline and not meta.get("is_stub", False),
            "distance_status": distance_status,
        }
        write_json(model_dir / "metrics.json", metrics)
        cqa = dict(pack["constraint_qa"])
        cqa.update({"verification_only": verification_only, "paper_mainline": metrics["paper_mainline"]})
        write_json(model_dir / "constraint_qa.json", cqa)
        rows.append({"law_model": model, **pack["metrics"], **meta})

    pd.DataFrame(rows).to_csv(out / "condition_g_scores.csv", index=False)
    write_json(
        out / "qa.json",
        {
            "ticket": "P5b",
            "fit_on": "HBW_R_only",
            "models": list(results),
            "distance_status": distance_status,
            "paper_mainline": paper_mainline,
        },
    )
    try:
        man = build_run_manifest(
            run_id=run_id,
            dataset=args.study_area,
            data_root=args.data_root,
            output_root=out,
            config={"models": names, "seed": args.seed},
            code_root=_ROOT,
            paper_mainline=paper_mainline,
            verification_only=verification_only,
            distance_status=distance_status,
            allow_backup_data_root=args.allow_backup,
            extra={"ticket": "P5b", "condition": "G_zeroshot"},
        )
        write_json(out / "run_manifest.json", man)
    except ValueError as e:
        write_json(out / "run_manifest_error.json", {"error": str(e)})

    print(json.dumps({"run_id": run_id, "out": str(out), "models": list(results)}, indent=2))


if __name__ == "__main__":
    main()
