#!/usr/bin/env python3
"""Audit server data readiness → outputs/readiness/*.json (Ticket 1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Ensure src on path when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from attrOD.data.manifest import (
    build_run_manifest,
    record_third_party_provenance,
    validate_server_data_root,
    write_json,
)
from attrOD.data.readiness import build_dataset_readiness


def main() -> None:
    p = argparse.ArgumentParser(description="AttrOD data readiness audit")
    p.add_argument("--data-root", default="~/AttrOD/data", help="canonical server data root")
    p.add_argument("--out", default="outputs/readiness")
    p.add_argument("--dataset", default="all", choices=["all", "mitma", "hk", "lombardy"])
    p.add_argument("--allow-backup", action="store_true",
                   help="permit /workspace/AttrOD/data as backup root")
    p.add_argument("--hk-qa", default=None, help="optional path to hk_teralytics/qa.json")
    p.add_argument("--config", default=None)
    p.add_argument("--run-id", default="readiness")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        root_info = validate_server_data_root(args.data_root, allow_backup=args.allow_backup)
    except ValueError as e:
        err = {"ok": False, "error": str(e)}
        write_json(out / "server_manifest.json", err)
        print(json.dumps(err, indent=2))
        raise SystemExit(2) from e

    hk_qa = None
    hk_zones = None
    if args.hk_qa and Path(args.hk_qa).exists():
        hk_qa = json.loads(Path(args.hk_qa).read_text())
        hk_zones = hk_qa.get("zone_ids")

    readiness = build_dataset_readiness(
        data_root=root_info["resolved"],
        dataset=args.dataset if args.dataset != "lombardy" else "all",
        hk_zone_ids=hk_zones,
        hk_qa=hk_qa,
    )

    cfg = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    tp = record_third_party_provenance(_ROOT / "third_party")
    server_manifest = {
        "run_id": args.run_id,
        "data_root": root_info,
        "third_party": tp,
        "readiness_summary": {
            "paper_mainline_allowed": readiness.get("paper_mainline_allowed"),
            "verification_only": readiness.get("verification_only"),
        },
    }
    if not args.dry_run:
        write_json(out / "server_manifest.json", server_manifest)
        write_json(out / "dataset_readiness.json", readiness)
        # also a run-style manifest for ledger continuity
        try:
            man = build_run_manifest(
                run_id=args.run_id,
                dataset=args.dataset,
                data_root=args.data_root,
                output_root=out,
                config=cfg,
                code_root=_ROOT,
                paper_mainline=False,
                verification_only=True,
                allow_backup_data_root=args.allow_backup,
            )
            write_json(out / "run_manifest.json", man)
        except ValueError as e:
            write_json(out / "run_manifest_error.json", {"error": str(e)})

    print(json.dumps({"server_manifest": server_manifest["data_root"], "readiness_ok_keys": list(readiness.get("checks", {}))}, indent=2))


if __name__ == "__main__":
    main()
