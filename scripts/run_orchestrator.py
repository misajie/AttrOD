#!/usr/bin/env python3
"""NOW-4 run orchestrator CLI — outputs/runs/<run_id>/{run_manifest,status,logs}.

Uses existing build_run_manifest (single schema). No paper-mainline under
provisional distance. Sample mode needs no dataset pull.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from attrOD.runs.ledger import RunLedger, load_manifest, load_status, new_run_id
from attrOD.condition_g.generators import NOW2_CLASSIC_SIX


def main() -> None:
    p = argparse.ArgumentParser(
        description="AttrOD NOW-4 run ledger (outputs/runs/<run_id>/)"
    )
    p.add_argument("--run-id", default=None, help="unique id; auto if omitted")
    p.add_argument("--dataset", default="ledger")
    p.add_argument("--data-root", default="~/AttrOD/data")
    p.add_argument("--output-root", default="outputs", help="parent of runs/")
    p.add_argument("--config", default=None, help="optional yaml config path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--distance-status",
        default="provisional_wgs84_not_metric",
        help="recorded in gates; provisional forces verification_only",
    )
    p.add_argument("--missing-fids", default="9", help="comma-separated; empty to clear")
    p.add_argument("--allow-backup", action="store_true")
    p.add_argument(
        "--command",
        default=None,
        help="optional shell stage; cwd=run work/; env ATTROD_RUN_* set",
    )
    p.add_argument(
        "--sample",
        action="store_true",
        help="write deterministic sample artifact (default if no --command)",
    )
    p.add_argument(
        "--replay",
        default=None,
        metavar="RUN_ID",
        help="re-run using seed/config/hash context from an existing run_manifest",
    )
    p.add_argument(
        "--fail-demo",
        action="store_true",
        help="force a failure after init to verify status retains stage+hash+error",
    )
    p.add_argument("--dry-run", action="store_true", help="print planned run_id and exit")
    p.add_argument("--pipeline", choices=["hk_smoke_classic"], default=None,
                   help="ledger-wired draft2 HK smoke→S→G classic matrix (NOW-5)")
    p.add_argument("--hk-materialised", default="outputs/hk_teralytics")
    p.add_argument("--hk-out", default="outputs/hk_smoke")
    p.add_argument("--classic-models", default=None,
                   help="override classic registry csv for hk_smoke_classic")
    args = p.parse_args()

    cfg = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.pipeline:
        cfg = {
            **cfg,
            "pipeline": args.pipeline,
            "hk_materialised": args.hk_materialised,
            "hk_out": args.hk_out,
            "classic_models": args.classic_models or ",".join(NOW2_CLASSIC_SIX),
        }

    missing = (
        [x.strip() for x in args.missing_fids.split(",") if x.strip()]
        if args.missing_fids is not None
        else []
    )

    if args.replay:
        prev = load_manifest(args.output_root, args.replay)
        prev_cfg = dict(prev.get("config") or {})
        prev_seed = int(prev_cfg.get("seed", args.seed))
        prev_flags = prev.get("flags") or {}
        run_id = new_run_id(prefix="replay")
        ledger = RunLedger(
            run_id=run_id,
            dataset=str(prev.get("dataset") or args.dataset),
            data_root=args.data_root,
            output_root=args.output_root,
            seed=prev_seed,
            config={**prev_cfg, "replay_of": args.replay},
            code_root=_ROOT,
            paper_mainline=False,
            verification_only=True,
            distance_status=prev_flags.get("distance_status", args.distance_status),
            missing_fids=prev_flags.get("missing_fids", missing),
            allow_backup_data_root=args.allow_backup,
            stage_command=args.command,
            extra={"replay_of": args.replay, "replay_of_input_hash": prev_cfg.get("input_hash")},
        )
    else:
        run_id = args.run_id or new_run_id()
        ledger = RunLedger(
            run_id=run_id,
            dataset=args.dataset,
            data_root=args.data_root,
            output_root=args.output_root,
            seed=args.seed,
            config=cfg,
            code_root=_ROOT,
            paper_mainline=False,
            verification_only=True,
            distance_status=args.distance_status,
            missing_fids=missing,
            allow_backup_data_root=args.allow_backup,
            stage_command=args.command,
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": ledger.run_id,
                    "input_hash": ledger.input_hash,
                    "run_dir": str(ledger.paths["base"]),
                },
                indent=2,
            )
        )
        return

    sample = args.sample or (not args.command and not args.pipeline)
    try:
        if args.fail_demo:
            try:
                ledger.start()
                raise RuntimeError(
                    "fail-demo: intentional NOW-4 failure retention check"
                )
            except Exception as exc:
                ledger.fail(ledger.status.get("stage") or "init", exc)
                raise
        elif args.pipeline == "hk_smoke_classic":
            models = args.classic_models or ",".join(NOW2_CLASSIC_SIX)
            allow = " --allow-backup" if args.allow_backup else ""
            smoke_py = _ROOT / "scripts" / "run_hk_smoke.py"
            cmd = (
                f'"{sys.executable}" "{smoke_py}"'
                f" --data-root {args.data_root}"
                f" --hk-materialised {args.hk_materialised}"
                f" --out {args.hk_out}"
                f" --seed {args.seed}"
                f" --run-id {ledger.run_id}"
                f" --classic-models {models}"
                f"{allow}"
            )
            ledger.start()
            ledger.set_stage("hk_smoke_classic", state="running")
            # run from repo root so relative outputs/ paths resolve; work/ stays private
            ledger.run_shell_stage(cmd, cwd=_ROOT)
            # record artifact pointer under run dir
            from attrOD.data.manifest import write_json as _wj
            _wj(
                ledger.paths["base"] / "pipeline_pointer.json",
                {
                    "pipeline": "hk_smoke_classic",
                    "hk_out": str(Path(args.hk_out)),
                    "classic_models": models.split(","),
                    "verification_only": True,
                    "paper_mainline": False,
                },
            )
            ledger.succeed()
            status = load_status(args.output_root, ledger.run_id)
        else:
            status = ledger.execute(sample=sample)
    except Exception:
        status = load_status(args.output_root, ledger.run_id)
        print(json.dumps(status, indent=2))
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "run_id": status["run_id"],
                "state": status["state"],
                "stage": status["stage"],
                "input_hash": status["input_hash"],
                "run_dir": status["paths"]["run_dir"],
                "gates": status["gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
