#!/usr/bin/env python3
"""P3: MITMA Condition G zero-shot (draft2) — fit only on HBW R.

Reads frozen materialised inputs (same layout as P2, plus optional attractiveness /
OSM features). Does not read P2 scores. Writes independent outputs under
outputs/mainline/mitma/condition_g/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# Zero-shot science set: classics + DeepGravity (pin) when --include-deep
ZERO_SHOT_CLASSIC = list(NOW2_CLASSIC_SIX)


def _load_optional(path: Path) -> Optional[np.ndarray]:
    return np.load(path) if path.exists() else None


def main() -> None:
    p = argparse.ArgumentParser(description="P3 MITMA Condition G zero-shot")
    p.add_argument("--materialised", required=True, help="frozen R.npy, distances.npy, T_*.npy")
    p.add_argument("--T", default=None, help="override target T.npy (default: T_full/total)")
    p.add_argument("--out", default="outputs/mainline/mitma/condition_g")
    p.add_argument("--models", default=None, help="comma-separated registry names")
    p.add_argument("--include-deep", action="store_true", help="also run DeepGravity registry model")
    p.add_argument("--adapters-dry-run", action="store_true", help="DG/NG provenance only; no main table")
    p.add_argument("--data-root", default="~/AttrOD/data")
    p.add_argument("--allow-backup", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--study-area", default="mitma")
    p.add_argument("--paper-mainline", action="store_true", default=True)
    p.add_argument("--verification-only", action="store_true")
    args = p.parse_args()

    mat = Path(args.materialised)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    R = np.load(mat / "R.npy")
    dist_path = mat / "distances.npy"
    if not dist_path.exists():
        dist_path = mat / "distance.npy"
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
            raise SystemExit("need --T or T_full.npy/total.npy under materialised")

    attr = _load_optional(mat / "attractiveness.npy")
    if attr is None:
        attr = np.maximum(R.sum(0), 1e-6)
    osm = _load_optional(mat / "osm_features.npy")
    pop = _load_optional(mat / "population.npy")
    if pop is None:
        pop = attr

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

    run_id = args.run_id or new_run_id(prefix="mitma_cg")
    train = list(range(R.shape[0]))

    if args.adapters_dry_run:
        from attrOD.models.adapters.deep_gravity import DeepGravityAdapter
        from attrOD.models.adapters.neuro_gravity import NeuroGravityAdapter
        from attrOD.data.manifest import record_third_party_provenance

        tp = _ROOT / "third_party"
        tp_out = out / "third_party"
        tp_out.mkdir(parents=True, exist_ok=True)
        dg = DeepGravityAdapter(tp, seed=args.seed)
        ng = NeuroGravityAdapter(tp, backend="stub", seed=args.seed)
        write_json(tp_out / "deepgravity_manifest.json", dg.provenance())
        write_json(tp_out / "neurogravity_manifest.json", ng.provenance())
        write_json(
            tp_out / "adapter_qa.json",
            {
                "neurogravity": {"backend": "stub", "formal_main_results": False, "is_stub": True},
                "third_party": record_third_party_provenance(tp),
                "note": "dry-run only — not Condition G main table",
            },
        )
        print(json.dumps({"adapters_dry_run": True, "out": str(tp_out)}, indent=2))
        return

    names = [x.strip() for x in args.models.split(",")] if args.models else list(ZERO_SHOT_CLASSIC)
    if args.include_deep and "DeepGravity" not in names:
        names.append("DeepGravity")

    results = run_registered_generators(
        R,
        T,
        distances,
        attr,
        train,
        model_names=names,
        populations=pop,
        osm_features=osm,
        zone_ids=zone_ids,
        include_deep_adapters=False,
        third_party_root=str(_ROOT / "third_party"),
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
            "hbw_only_fit": meta.get("hbw_only_fit", True),
            "is_oracle": meta.get("is_oracle", False),
            "is_stub": meta.get("is_stub", False),
            "distance_status": distance_status,
        }
        write_json(model_dir / "metrics.json", metrics)
        cqa = dict(pack["constraint_qa"])
        cqa.update(
            {
                "verification_only": verification_only,
                "paper_mainline": metrics["paper_mainline"],
                "hbw_only_fit": bool(meta.get("hbw_only_fit", True)),
            }
        )
        write_json(model_dir / "constraint_qa.json", cqa)
        rows.append({"law_model": model, **pack["metrics"], **meta})

    pd.DataFrame(rows).to_csv(out / "condition_g_scores.csv", index=False)
    write_json(
        out / "qa.json",
        {
            "ticket": "P3",
            "fit_on": "HBW_R_only",
            "target_partition": t_label,
            "models": list(results),
            "distance_status": distance_status,
            "verification_only": verification_only,
            "paper_mainline": paper_mainline,
            "note": "zero-shot Condition G; NG few-shot is a separate runner",
        },
    )
    try:
        man = build_run_manifest(
            run_id=run_id,
            dataset=args.study_area,
            data_root=args.data_root,
            output_root=out,
            config={"models": names, "seed": args.seed, "target": t_label},
            code_root=_ROOT,
            paper_mainline=paper_mainline,
            verification_only=verification_only,
            distance_status=distance_status,
            allow_backup_data_root=args.allow_backup,
            extra={"ticket": "P3", "condition": "G_zeroshot"},
        )
        write_json(out / "run_manifest.json", man)
    except ValueError as e:
        write_json(out / "run_manifest_error.json", {"error": str(e)})

    print(
        json.dumps(
            {
                "run_id": run_id,
                "out": str(out),
                "models": list(results),
                "paper_mainline": paper_mainline,
                "distance_status": distance_status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
