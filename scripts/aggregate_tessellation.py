#!/usr/bin/env python3
"""AttrOD CLI — see README for Server Bot recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from attrOD.scale.experiment import scale_spearman_and_delta_sign
from attrOD.tessellation.core import official_dissolve, random_contiguous_merge


def main():
    p = argparse.ArgumentParser(description="Coarser tessellations for Table 8")
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["official_dissolve", "random_contiguous"], required=True)
    p.add_argument("--mat", default=None, help="native OD .npy")
    p.add_argument("--zones", default=None, help="JSON list of zone ids")
    p.add_argument("--mapping", default=None, help="JSON zone->group for official dissolve")
    p.add_argument("--adjacency", default=None, help="JSON adjacency for random merge")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.mat:
        (out / "README.txt").write_text(
            "Scale aggregation stub. Pass --mat and mapping/adjacency to materialise.\n"
        )
        print(f"wrote stub {out}")
        return
    mat = np.load(args.mat)
    zones = json.loads(Path(args.zones).read_text()) if args.zones else [str(i) for i in range(mat.shape[0])]
    if args.mode == "official_dissolve":
        if not args.mapping:
            raise SystemExit("--mapping required for official_dissolve")
        mapping = json.loads(Path(args.mapping).read_text())
        coarse, groups = official_dissolve(mat, mapping, zones)
        np.save(out / "coarse.npy", coarse)
        (out / "groups.json").write_text(json.dumps(groups, indent=2))
    else:
        if not args.adjacency:
            raise SystemExit("--adjacency required for random_contiguous")
        adj = json.loads(Path(args.adjacency).read_text())
        rng = np.random.default_rng(args.seed)
        coarse, groups, z2g = random_contiguous_merge(adj, zones, mat, rng=rng)
        np.save(out / "coarse.npy", coarse)
        (out / "groups.json").write_text(json.dumps(groups, indent=2))
        (out / "zone_to_group.json").write_text(json.dumps(z2g, indent=2))
    print(f"coarse N={coarse.shape[0]} -> {out}")


if __name__ == "__main__":
    main()
