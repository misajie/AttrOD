#!/usr/bin/env python3
"""Materialise Hong Kong Teralytics OD artefacts for AttrOD S1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attrOD.data.teralytics import materialise_hk_teralytics


def main() -> None:
    p = argparse.ArgumentParser(description="HK Teralytics S1 materialisation")
    p.add_argument(
        "--csv",
        required=True,
        help="pipe-delimited Teralytics OD matrix CSV",
    )
    p.add_argument(
        "--shapefile-dir",
        required=True,
        help="directory containing CHKU_Shapes.dbf/.shp/.prj",
    )
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument(
        "--no-partitions",
        action="store_true",
        help="skip purpose×period partition long tables",
    )
    args = p.parse_args()
    result = materialise_hk_teralytics(
        args.csv,
        args.shapefile_dir,
        args.out,
        chunksize=args.chunksize,
        write_purpose_period_partitions=not args.no_partitions,
    )
    qa = result["qa"]
    print(
        json.dumps(
            {
                "n_zones": qa["n_zones_shapefile"],
                "unmatched": qa["csv_ids_unmatched"],
                "count_column": qa["count_column"],
                "total_sum": qa["flow_sum_total_matrix"],
                "proxy_sum": qa["flow_sum_proxy_matrix"],
                "proxy_definition": qa["proxy_definition"],
                "out": str(Path(args.out).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
