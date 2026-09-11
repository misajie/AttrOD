#!/usr/bin/env python3
"""AttrOD CLI — see README for Server Bot recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from attrOD.condition_g import run_generators
from attrOD.reporting.schemas import empty_table
from attrOD.spatial.blocks import folds_from_blocks, make_queen_blocks


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
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.R and args.T and args.distances and args.attractiveness:
        R = np.load(args.R)
        T = np.load(args.T)
        d = np.load(args.distances)
        m = np.load(args.attractiveness)
        n = R.shape[0]
        if args.blocks and Path(args.blocks).exists():
            blocks = json.loads(Path(args.blocks).read_text())["blocks"]
        else:
            # full-tessellation (Table 7 style): all origins train
            blocks = [list(range(n))]
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
        "note": "Pass --R --T --distances --attractiveness to run generators.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
