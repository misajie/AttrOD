"""Run ledger: isolated outputs/runs/<run_id>/ orchestration (NOW-4).

Extends the single build_run_manifest schema — no second manifest shape.
Each run owns its directory; parallel jobs must not share mutable paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Union

from attrOD.data.manifest import (
    DEEPGRAVITY_PIN,
    NEUROGRAVITY_PIN,
    build_run_manifest,
    write_json,
)

PathLike = Union[str, Path]

STATUS_SCHEMA = "attrOD.run_status.v1"
FIXED_FILENAMES = {
    "run_manifest": "run_manifest.json",
    "status": "status.json",
    "logs_dir": "logs",
    "work_dir": "work",
}


def new_run_id(*, prefix: str = "run") -> str:
    """UTC timestamp + short uuid — unique and sortable."""
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def stable_input_hash(payload: Mapping[str, Any]) -> str:
    """Deterministic sha256 over canonical JSON of inputs/config/seeds."""
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def runs_root(output_root: PathLike) -> Path:
    return Path(output_root) / "runs"


def run_dir(output_root: PathLike, run_id: str) -> Path:
    return runs_root(output_root) / run_id


def ensure_run_layout(output_root: PathLike, run_id: str) -> Dict[str, Path]:
    """Create the fixed per-run layout. Never reuses another run's paths."""
    base = run_dir(output_root, run_id)
    logs = base / FIXED_FILENAMES["logs_dir"]
    work = base / FIXED_FILENAMES["work_dir"]
    logs.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "logs": logs,
        "work": work,
        "run_manifest": base / FIXED_FILENAMES["run_manifest"],
        "status": base / FIXED_FILENAMES["status"],
    }


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def write_status(path: PathLike, status: Mapping[str, Any]) -> Path:
    return write_json(path, status)


def append_log(logs_dir: PathLike, stage: str, line: str) -> Path:
    p = Path(logs_dir) / f"{stage}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(f"{_utc_now()} {line.rstrip()}\n")
    return p


class RunLedger:
    """Gate-aware run orchestrator with failure retention and replay hooks."""

    def __init__(
        self,
        *,
        run_id: str,
        dataset: str,
        data_root: PathLike,
        output_root: PathLike,
        seed: int = 0,
        config: Optional[Mapping[str, Any]] = None,
        code_root: Optional[PathLike] = None,
        paper_mainline: bool = False,
        verification_only: bool = True,
        distance_status: Optional[str] = "provisional_wgs84_not_metric",
        missing_fids: Optional[Sequence[Any]] = None,
        allow_backup_data_root: bool = False,
        stage_command: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # Hard gate: never claim paper mainline while provisional / verification.
        if distance_status == "provisional_wgs84_not_metric":
            paper_mainline = False
            verification_only = True
        if paper_mainline and verification_only:
            paper_mainline = False

        self.run_id = run_id
        self.dataset = dataset
        self.data_root = data_root
        self.output_root = Path(output_root)
        self.seed = int(seed)
        self.config = dict(config or {})
        self.code_root = code_root
        self.paper_mainline = bool(paper_mainline)
        self.verification_only = bool(verification_only)
        self.distance_status = distance_status
        self.missing_fids = list(missing_fids) if missing_fids is not None else []
        self.allow_backup_data_root = allow_backup_data_root
        self.stage_command = stage_command
        self.extra = dict(extra or {})
        self.paths = ensure_run_layout(self.output_root, self.run_id)
        self.input_hash = stable_input_hash(
            {
                "dataset": self.dataset,
                "data_root": str(self.data_root),
                "config": self.config,
                "seed": self.seed,
                "stage_command": self.stage_command,
                "distance_status": self.distance_status,
                "missing_fids": self.missing_fids,
                "paper_mainline": self.paper_mainline,
                "verification_only": self.verification_only,
                "pins": {
                    "deepgravity": DEEPGRAVITY_PIN,
                    "neurogravity": NEUROGRAVITY_PIN,
                },
            }
        )
        self.status: Dict[str, Any] = {
            "schema": STATUS_SCHEMA,
            "run_id": self.run_id,
            "state": "pending",
            "stage": "init",
            "input_hash": self.input_hash,
            "seed": self.seed,
            "started_utc": None,
            "finished_utc": None,
            "error": None,
            "gates": {
                "paper_mainline": self.paper_mainline,
                "verification_only": self.verification_only,
                "distance_status": self.distance_status,
                "missing_fids": self.missing_fids,
                "ok_mainline": False,
            },
            "paths": {
                "run_dir": str(self.paths["base"]),
                "work_dir": str(self.paths["work"]),
                "logs_dir": str(self.paths["logs"]),
            },
            "filenames": dict(FIXED_FILENAMES),
        }

    def _persist_status(self) -> None:
        write_status(self.paths["status"], self.status)

    def _write_manifest(self) -> Dict[str, Any]:
        man = build_run_manifest(
            run_id=self.run_id,
            dataset=self.dataset,
            data_root=self.data_root,
            output_root=self.paths["base"],
            config={**self.config, "seed": self.seed, "input_hash": self.input_hash},
            code_root=self.code_root,
            paper_mainline=self.paper_mainline,
            verification_only=self.verification_only,
            distance_status=self.distance_status,
            missing_fids=self.missing_fids,
            allow_backup_data_root=self.allow_backup_data_root,
            extra={
                **self.extra,
                "ledger": {
                    "schema": STATUS_SCHEMA,
                    "input_hash": self.input_hash,
                    "seed": self.seed,
                    "stage_command": self.stage_command,
                    "work_dir": str(self.paths["work"]),
                },
            },
        )
        write_json(self.paths["run_manifest"], man)
        return man

    def set_stage(self, stage: str, *, state: Optional[str] = None) -> None:
        self.status["stage"] = stage
        if state:
            self.status["state"] = state
        append_log(
            self.paths["logs"],
            stage,
            f"enter stage={stage} state={self.status['state']}",
        )
        self._persist_status()

    def fail(self, stage: str, exc: BaseException) -> None:
        """Retain stage + input_hash + full error; never wipe prior artifacts."""
        self.status["stage"] = stage
        self.status["state"] = "failed"
        self.status["finished_utc"] = _utc_now()
        self.status["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        append_log(self.paths["logs"], stage, f"FAILED {type(exc).__name__}: {exc}")
        self._persist_status()

    def succeed(self) -> None:
        self.status["state"] = "ok"
        self.status["stage"] = "done"
        self.status["finished_utc"] = _utc_now()
        self.status["error"] = None
        append_log(self.paths["logs"], "done", "run ok")
        self._persist_status()

    @contextmanager
    def isolated_workdir(self) -> Iterator[Path]:
        """Private work dir for this run only — no shared mutable intermediates."""
        work = self.paths["work"]
        work.mkdir(parents=True, exist_ok=True)
        prev = os.environ.get("ATTROD_RUN_WORK")
        os.environ["ATTROD_RUN_WORK"] = str(work)
        try:
            yield work
        finally:
            if prev is None:
                os.environ.pop("ATTROD_RUN_WORK", None)
            else:
                os.environ["ATTROD_RUN_WORK"] = prev

    def start(self) -> Dict[str, Any]:
        self.status["started_utc"] = _utc_now()
        self.status["state"] = "running"
        self.set_stage("init", state="running")
        man = self._write_manifest()
        append_log(
            self.paths["logs"],
            "init",
            f"run_id={self.run_id} input_hash={self.input_hash}",
        )
        return man

    def run_sample_stage(self) -> Path:
        """Deterministic sample artifact for acceptance without pulling datasets."""
        self.set_stage("sample")
        with self.isolated_workdir() as work:
            artifact = work / "sample_marker.json"
            payload = {
                "run_id": self.run_id,
                "input_hash": self.input_hash,
                "seed": self.seed,
                "note": "NOW-4 sample stage — verification_only, not paper mainline",
                "gates": self.status["gates"],
            }
            write_json(artifact, payload)
            append_log(self.paths["logs"], "sample", f"wrote {artifact}")
            return artifact

    def run_shell_stage(self, command: str) -> None:
        """Execute an external stage with cwd=work and env isolation."""
        import subprocess

        self.set_stage("command")
        with self.isolated_workdir() as work:
            append_log(self.paths["logs"], "command", f"$ {command}")
            log_path = self.paths["logs"] / "command.out"
            with log_path.open("w", encoding="utf-8") as out:
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(work),
                    env={
                        **os.environ,
                        "ATTROD_RUN_ID": self.run_id,
                        "ATTROD_RUN_WORK": str(work),
                        "ATTROD_INPUT_HASH": self.input_hash,
                        "ATTROD_SEED": str(self.seed),
                    },
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"stage command exited {proc.returncode}; see {log_path}"
                )
            append_log(self.paths["logs"], "command", f"exit={proc.returncode}")

    def execute(self, *, sample: bool = False) -> Dict[str, Any]:
        try:
            self.start()
            if sample or not self.stage_command:
                self.run_sample_stage()
            if self.stage_command:
                self.run_shell_stage(self.stage_command)
            self.succeed()
        except BaseException as exc:
            self.fail(self.status.get("stage") or "unknown", exc)
            raise
        return self.status


def load_status(output_root: PathLike, run_id: str) -> Dict[str, Any]:
    path = run_dir(output_root, run_id) / FIXED_FILENAMES["status"]
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifest(output_root: PathLike, run_id: str) -> Dict[str, Any]:
    path = run_dir(output_root, run_id) / FIXED_FILENAMES["run_manifest"]
    return json.loads(Path(path).read_text(encoding="utf-8"))
