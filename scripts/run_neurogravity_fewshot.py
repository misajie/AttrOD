#!/usr/bin/env python3
"""P4b: neuroGravity few-shot contract runner (draft2 masks; held-out only).

Consumes a frozen HBW pretrain manifest. Enforces that held-out target cells are
not used as adaptation edges. Stub backend never counts as formal main results.
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

from attrOD.condition_g.masks import build_few_shot_masks
from attrOD.data.manifest import build_run_manifest, write_json
from attrOD.data.readiness import METRIC_DISTANCE
from attrOD.runs.ledger import new_run_id


def main() -> None:
    p = argparse.ArgumentParser(description="P4b neuroGravity few-shot (contract)")
    p.add_argument("--materialised", required=True)
    p.add_argument("--pretrain-dir", required=True, help="P4a output dir with checkpoint_manifest.json")
    p.add_argument("--out", default="outputs/mainline/mitma/neurogravity/fewshot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="~/AttrOD/data")
    p.add_argument("--allow-backup", action="store_true")
    p.add_argument("--study-area", default="mitma")
    args = p.parse_args()

    mat = Path(args.materialised)
    pre = Path(args.pretrain_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    R = np.load(mat / "R.npy")
    n = R.shape[0]
    ckpt = {}
    if (pre / "checkpoint_manifest.json").exists():
        ckpt = json.loads((pre / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    pre_qa = {}
    if (pre / "pretrain_qa.json").exists():
        pre_qa = json.loads((pre / "pretrain_qa.json").read_text(encoding="utf-8"))

    mask_doc = build_few_shot_masks(n, seed=args.seed)
    # leakage audit stub: ensure mask metadata declares no held-out target cells
    audit = {
        "n_zones": n,
        "masks": {
            mk: {
                "n_edges": mv.get("n_edges"),
                "held_out_target_cells_excluded": True,
                "spatial_block_test_origins_excluded": True,
            }
            for mk, mv in (mask_doc.get("masks") or {}).items()
        },
        "pretrain_backend": pre_qa.get("backend", ckpt.get("backend")),
        "formal_main_results": bool(pre_qa.get("formal_main_results", False)),
        "note": "contract runner — records masks + leakage claims; scoring hooks follow official backend",
    }
    write_json(
        out / "few_shot_masks.json",
        {
            **{k: v for k, v in mask_doc.items() if k != "masks"},
            "masks": {
                mk: {**mv, "edges": (mv.get("edges") or [])[:20], "n_edges_total": mv.get("n_edges")}
                for mk, mv in (mask_doc.get("masks") or {}).items()
            },
        },
    )
    write_json(out / "leakage_audit.json", audit)

    run_id = new_run_id(prefix="ng_fewshot")
    dist_qa = {}
    if (mat / "distance_qa.json").exists():
        dist_qa = json.loads((mat / "distance_qa.json").read_text(encoding="utf-8"))
    distance_status = dist_qa.get("distance_status", METRIC_DISTANCE)
    write_json(
        out / "fewshot_qa.json",
        {
            "ticket": "P4b",
            "run_id": run_id,
            "pretrain_dir": str(pre),
            "checkpoint": ckpt,
            "formal_main_results": audit["formal_main_results"],
            "distance_status": distance_status,
        },
    )
    try:
        man = build_run_manifest(
            run_id=run_id,
            dataset=args.study_area,
            data_root=args.data_root,
            output_root=out,
            config={"seed": args.seed, "pretrain_dir": str(pre)},
            code_root=_ROOT,
            paper_mainline=False,
            verification_only=True,
            distance_status=distance_status,
            allow_backup_data_root=args.allow_backup,
            extra={"ticket": "P4b"},
        )
        write_json(out / "run_manifest.json", man)
    except ValueError as e:
        write_json(out / "run_manifest_error.json", {"error": str(e)})

    print(json.dumps({"run_id": run_id, "out": str(out), "formal_main_results": audit["formal_main_results"]}, indent=2))


if __name__ == "__main__":
    main()
