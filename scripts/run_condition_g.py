#!/usr/bin/env python3
"""AttrOD CLI — Condition G registry + adapters (Ticket 3)."""
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

from attrOD.condition_g.generators import predictions_to_frame, run_registered_generators
from attrOD.condition_g.masks import build_few_shot_masks
from attrOD.condition_g.pipeline import run_generators
from attrOD.data.manifest import build_run_manifest, record_third_party_provenance, write_json
from attrOD.models.adapters.deep_gravity import DeepGravityAdapter
from attrOD.models.adapters.neuro_gravity import NeuroGravityAdapter
from attrOD.models.registry import list_models
from attrOD.reporting.io import write_frame
from attrOD.reporting.schemas import empty_table
from attrOD.spatial.blocks import folds_from_blocks


def main():
    p = argparse.ArgumentParser(description="Run Condition G")
    p.add_argument("--config", required=True)
    p.add_argument("--partitions", required=True)
    p.add_argument("--blocks", default=None, help="blocks.json from make_spatial_blocks")
    p.add_argument("--out", required=True)
    p.add_argument("--R", default=None)
    p.add_argument("--T", default=None)
    p.add_argument("--distances", default=None)
    p.add_argument("--attractiveness", default=None)
    p.add_argument("--mechanism", action="store_true")
    p.add_argument("--fewshot", action="store_true")
    p.add_argument("--registry", action="store_true", help="use Ticket-3 registry path")
    p.add_argument("--models", default=None, help="comma-separated registry names")
    p.add_argument("--adapters-dry-run", action="store_true",
                   help="DeepGravity/NeuroGravity import + schema round-trip only")
    p.add_argument("--data-root", default="~/AttrOD/data")
    p.add_argument("--allow-backup", action="store_true")
    p.add_argument("--run-id", default="condition_g")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verification-only", action="store_true", default=True)
    p.add_argument("--paper-mainline", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    verification_only = True if not args.paper_mainline else False
    paper_mainline = bool(args.paper_mainline)

    tp_root = _ROOT / "third_party"
    if args.adapters_dry_run:
        dg = DeepGravityAdapter(tp_root, seed=args.seed)
        ng = NeuroGravityAdapter(tp_root, backend="stub", seed=args.seed)
        # tiny synthetic for round-trip
        n = 4
        rng = np.random.default_rng(args.seed)
        R = rng.random((n, n))
        d = rng.random((n, n)) + 0.1
        pop = rng.random(n) + 1
        feats = rng.random((n, 3))
        rt = dg.schema_roundtrip(R, feats, pop, d)
        # Prefer repo outputs/third_party
        tp_out = _ROOT / "outputs" / "third_party"
        tp_out.mkdir(parents=True, exist_ok=True)
        dg_prov = dg.provenance()
        write_json(tp_out / "deepgravity_manifest.json", dg_prov)
        write_json(tp_out / "neurogravity_manifest.json", ng.provenance())
        write_json(tp_out / "adapter_qa.json", {
            "deepgravity_roundtrip": rt,
            "neurogravity": {
                "backend": "stub",
                "formal_main_results": False,
                "is_stub": True,
                "pin": ng.PIN,
            },
            "third_party_commits": record_third_party_provenance(tp_root),
        })
        print(json.dumps({"adapters_dry_run": True, "dg_importable": rt["official_importable"]}, indent=2))
        return

    if args.R and args.T and args.distances and args.attractiveness:
        R = np.load(args.R)
        T = np.load(args.T)
        d = np.load(args.distances)
        m = np.load(args.attractiveness)
        n = R.shape[0]
        if args.blocks and Path(args.blocks).exists():
            blocks = json.loads(Path(args.blocks).read_text())["blocks"]
        else:
            blocks = [list(range(n))]

        if args.registry or args.models:
            names = args.models.split(",") if args.models else None
            train = list(range(n))
            if len(blocks) > 1:
                fold0 = folds_from_blocks(blocks)[0]
                train = fold0["train_origins"]
            results = run_registered_generators(
                R, T, d, m, train,
                model_names=names,
                include_deep_adapters=args.mechanism,
                third_party_root=str(tp_root),
                seed=args.seed,
            )
            rows = []
            for model, pack in results.items():
                model_dir = out / model
                model_dir.mkdir(parents=True, exist_ok=True)
                write_frame(model_dir / "predictions.parquet", predictions_to_frame(pack["T_hat"]))
                write_json(model_dir / "metrics.json", {
                    **pack["metrics"],
                    "verification_only": verification_only,
                    "paper_mainline": paper_mainline and not pack["meta"].get("is_stub", False),
                    **{k: pack["meta"].get(k) for k in ("is_stub", "is_oracle", "formal_main_results") if k in pack["meta"] or True},
                })
                write_json(model_dir / "constraint_qa.json", pack["constraint_qa"])
                if "provenance" in pack:
                    write_json(model_dir / "provenance.json", pack["provenance"])
                rows.append({"law_model": model, **pack["metrics"], **pack["meta"]})
            pd.DataFrame(rows).to_csv(out / "condition_g_scores.csv", index=False)
            # few-shot mask stub from draft2
            mask_doc = build_few_shot_masks(n, seed=args.seed)
            write_json(out / "few_shot_masks.json", {
                **{k: v for k, v in mask_doc.items() if k != "masks"},
                "masks": {mk: {**mv, "edges": mv["edges"][:20], "n_edges_total": mv["n_edges"]} for mk, mv in mask_doc["masks"].items()},
            })
            print(f"wrote registry results models={list(results)} → {out}")
            return

        # legacy path
        rows = []
        for fold in folds_from_blocks(blocks) if len(blocks) > 1 else [
            {"fold": 0, "train_origins": list(range(n)), "test_origins": []}
        ]:
            train = fold["train_origins"]
            res = run_generators(
                R, T, d, m, train,
                mechanism=args.mechanism or bool(cfg.get("mechanism_subset")),
                fewshot=args.fewshot,
            )
            for model, scores in res.items():
                rows.append({"fold": fold["fold"], "law_model": model, **scores})
        pd.DataFrame(rows).to_csv(out / "condition_g_scores.csv", index=False)
        print(f"wrote {out / 'condition_g_scores.csv'} rows={len(rows)}")
        return

    empty_table("table6_condition_g_blocks").to_csv(out / "table6_condition_g_blocks.csv", index=False)
    empty_table("table7_panel_no_holdout").to_csv(out / "table7_panel_no_holdout.csv", index=False)
    meta = {
        "study_area_id": cfg.get("study_area_id"),
        "registry_models": list_models(),
        "note": "Pass --R --T --distances --attractiveness to run generators. Use --registry for Ticket-3 path.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
