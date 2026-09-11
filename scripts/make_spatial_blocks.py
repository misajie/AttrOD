#!/usr/bin/env python3
"""AttrOD CLI — see README for Server Bot recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from attrOD.spatial.blocks import folds_from_blocks, make_queen_blocks, random_origin_holdout


def main():
    p = argparse.ArgumentParser(description="Queen contiguous 5-block origin split")
    p.add_argument("--config", required=True)
    p.add_argument("--adjacency", default=None, help="JSON {zone_idx: [nbr_idx,...]}")
    p.add_argument("--out", required=True)
    p.add_argument("--n-blocks", type=int, default=5)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    min_N = cfg.get("spatial", {}).get("min_N_mechanism", 80)
    n_zones = len(cfg.get("zone_ids") or [])
    if args.adjacency:
        raw = json.loads(Path(args.adjacency).read_text())
        adjacency = {int(k): list(map(int, v)) for k, v in raw.items()}
        n = max(adjacency.keys()) + 1 if adjacency else 0
    else:
        # stub empty adjacency — writes schema only
        adjacency = {}
        n = n_zones
    if n and n < min_N:
        print(f"warning: N={n} < mechanism threshold {min_N}; blocks still computed if adjacency given")
    if adjacency:
        blocks = make_queen_blocks(adjacency, n_blocks=args.n_blocks, n_nodes=n)
        folds = folds_from_blocks(blocks)
        hold_size = len(blocks[0]) if blocks else 0
        train_r, hold_r = random_origin_holdout(n, hold_size)
        payload = {
            "n": n,
            "n_blocks": args.n_blocks,
            "blocks": blocks,
            "folds": folds,
            "random_holdout": {"train": train_r, "test": hold_r},
        }
    else:
        payload = {
            "n": n,
            "n_blocks": args.n_blocks,
            "blocks": [],
            "note": "Provide --adjacency JSON to compute blocks.",
        }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
