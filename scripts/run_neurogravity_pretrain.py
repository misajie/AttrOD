#!/usr/bin/env python3
"""P4a: neuroGravity HBW pretrain contract runner (stub-safe).

Records checkpoint/provenance pathing. Full edge-enhanced training only when the
official backend is enabled; default uses the stub adapter and marks
formal_main_results=false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from attrOD.data.manifest import build_run_manifest, write_json
from attrOD.data.readiness import METRIC_DISTANCE
from attrOD.models.adapters.neuro_gravity import NeuroGravityAdapter
from attrOD.runs.ledger import new_run_id


def main() -> None:
    p = argparse.ArgumentParser(description="P4a neuroGravity HBW pretrain (contract)")
    p.add_argument("--materialised", required=True)
    p.add_argument("--out", default="outputs/mainline/mitma/neurogravity/pretrain")
    p.add_argument("--backend", default="stub", choices=["stub", "official"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="~/AttrOD/data")
    p.add_argument("--allow-backup", action="store_true")
    p.add_argument("--study-area", default="mitma")
    args = p.parse_args()

    mat = Path(args.materialised)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = np.load(mat / "R.npy")
    distances = np.load(mat / "distances.npy" if (mat / "distances.npy").exists() else mat / "distance.npy")
    pop = np.load(mat / "population.npy") if (mat / "population.npy").exists() else np.maximum(R.sum(0), 1e-6)
    feats = np.load(mat / "osm_features.npy") if (mat / "osm_features.npy").exists() else np.column_stack([pop, pop])

    ng = NeuroGravityAdapter(_ROOT / "third_party", backend=args.backend, seed=args.seed)
    # contract: zeroshot fit on HBW R (stub path)
    train = list(range(R.shape[0]))
    try:
        ng.fit_zeroshot(R, distances, pop, feats, train)
        fit_ok = True
        err = None
    except Exception as exc:  # noqa: BLE001 — contract runner records failure
        fit_ok = False
        err = f"{type(exc).__name__}: {exc}"

    run_id = new_run_id(prefix="ng_pretrain")
    prov = ng.provenance() if hasattr(ng, "provenance") else {}
    payload = {
        "ticket": "P4a",
        "backend": args.backend,
        "formal_main_results": args.backend == "official",
        "is_stub": args.backend == "stub",
        "fit_ok": fit_ok,
        "error": err,
        "provenance": prov,
        "seed": args.seed,
        "n_zones": int(R.shape[0]),
        "note": "HBW pretrain contract; few-shot is a separate runner",
    }
    write_json(out / "pretrain_qa.json", payload)
    write_json(out / "neurogravity_manifest.json", prov)
    # checkpoint placeholder path (stub may not write weights)
    write_json(
        out / "checkpoint_manifest.json",
        {
            "run_id": run_id,
            "backend": args.backend,
            "checkpoint_path": None if args.backend == "stub" else str(out / "checkpoint.pt"),
            "seed": args.seed,
        },
    )
    dist_qa = {}
    if (mat / "distance_qa.json").exists():
        dist_qa = json.loads((mat / "distance_qa.json").read_text(encoding="utf-8"))
    distance_status = dist_qa.get("distance_status", METRIC_DISTANCE)
    try:
        man = build_run_manifest(
            run_id=run_id,
            dataset=args.study_area,
            data_root=args.data_root,
            output_root=out,
            config={"backend": args.backend, "seed": args.seed},
            code_root=_ROOT,
            paper_mainline=False,
            verification_only=True,
            distance_status=distance_status,
            allow_backup_data_root=args.allow_backup,
            extra={"ticket": "P4a"},
        )
        write_json(out / "run_manifest.json", man)
    except ValueError as e:
        write_json(out / "run_manifest_error.json", {"error": str(e)})

    print(json.dumps({"run_id": run_id, "fit_ok": fit_ok, "backend": args.backend, "out": str(out)}, indent=2))
    if not fit_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
