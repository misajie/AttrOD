"""Run manifest + server data-root validators (Ticket 1 / NOW-0).

Canonical data root on myserver is ``~/AttrOD/data``. Mac paths and undeclared
backup roots are rejected. ``/workspace/AttrOD/data`` may be recorded only as
backup and must not be mixed with the canonical root in one run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

PathLike = Union[str, Path]

# Commit pins required by EXPERIMENT_PLAN / draft2 integration notes.
DEEPGRAVITY_PIN = "8693536"
NEUROGRAVITY_PIN = "7e29ef0"

_MAC_MARKERS = (
    "/Users/",
    "/Volumes/",
    "Library/CloudStorage",
    "OneDrive-",
)


def _expand(p: PathLike) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


def is_mac_path(path: PathLike) -> bool:
    """True if *path* looks like a user Mac / OneDrive path (must be rejected)."""
    s = str(path)
    expanded = str(Path(os.path.expanduser(s)))
    for m in _MAC_MARKERS:
        if m in s or m in expanded:
            return True
    # bare macOS volume roots
    if expanded.startswith("/Users/") or s.startswith("/Users/"):
        return True
    return False


def validate_server_data_root(
    data_root: PathLike,
    *,
    allow_backup: bool = False,
    backup_roots: Optional[Sequence[PathLike]] = None,
) -> Dict[str, Any]:
    """Validate that *data_root* is a server-side AttrOD data directory.

    Returns a dict with ``ok``, ``resolved``, ``role`` (``canonical``|``backup``),
    and ``violations`` (list of strings). Raises ``ValueError`` on Mac paths or
    other hard failures when used as the primary root.
    """
    violations: List[str] = []
    raw = str(data_root)
    if is_mac_path(raw):
        msg = f"Mac / OneDrive path rejected for data_root: {raw}"
        violations.append(msg)
        raise ValueError(msg)

    resolved = _expand(data_root)
    if is_mac_path(resolved):
        msg = f"Mac / OneDrive path rejected after resolve: {resolved}"
        violations.append(msg)
        raise ValueError(msg)

    home_canonical = _expand("~/AttrOD/data")
    workspace_backup = Path("/workspace/AttrOD/data")
    backup_set = {_expand(b) for b in (backup_roots or [])}
    if workspace_backup.exists():
        backup_set.add(workspace_backup.resolve())

    role = "unknown"
    resolved_s = str(resolved)
    is_workspace_backup = resolved_s.startswith("/workspace/AttrOD/data") or resolved in backup_set
    is_attr_od_data = resolved.name == "data" and resolved.parent.name == "AttrOD"
    if is_workspace_backup:
        role = "backup"
    elif resolved == home_canonical or (is_attr_od_data and "/workspace/" not in resolved_s):
        role = "canonical"
    elif is_attr_od_data:
        role = "backup"
    else:
        violations.append(
            f"data_root not recognised as AttrOD data directory: {resolved}"
        )

    if role == "backup" and not allow_backup:
        violations.append(
            f"backup data_root {resolved} used without allow_backup=True "
            "(canonical is ~/AttrOD/data; /workspace/AttrOD/data is backup only)"
        )

    ok = len(violations) == 0
    result = {
        "ok": ok,
        "resolved": str(resolved),
        "role": role,
        "violations": violations,
        "home_canonical": str(home_canonical),
    }
    if not ok:
        raise ValueError("; ".join(violations))
    return result


def file_fingerprint(path: PathLike) -> Dict[str, Any]:
    """Path, size, mtime, sha256 for an existing file (or missing marker)."""
    p = Path(path)
    if not p.exists():
        return {
            "path": str(p),
            "exists": False,
            "size": None,
            "mtime": None,
            "sha256": None,
        }
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size": int(st.st_size),
        "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": h.hexdigest(),
    }


def _git_rev(repo: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return None


def _git_short(repo: Path, n: int = 7) -> Optional[str]:
    rev = _git_rev(repo)
    return rev[:n] if rev else None


def record_third_party_provenance(
    third_party_root: PathLike,
    *,
    deepgravity_pin: str = DEEPGRAVITY_PIN,
    neurogravity_pin: str = NEUROGRAVITY_PIN,
) -> Dict[str, Any]:
    """Record DeepGravity / NeuroGravity commit pins and observed HEADs."""
    root = Path(third_party_root)
    dg = root / "DeepGravity"
    ng = root / "NeuroGravity"
    dg_head = _git_rev(dg) if dg.exists() else None
    ng_head = _git_rev(ng) if ng.exists() else None
    dg_short = (dg_head or "")[:7]
    ng_short = (ng_head or "")[:7]
    return {
        "deepgravity": {
            "path": str(dg) if dg.exists() else None,
            "pin": deepgravity_pin,
            "head": dg_head,
            "pin_matched": bool(dg_short.startswith(deepgravity_pin[:7])) if dg_head else False,
            "read_only": True,
        },
        "neurogravity": {
            "path": str(ng) if ng.exists() else None,
            "pin": neurogravity_pin,
            "head": ng_head,
            "pin_matched": bool(ng_short.startswith(neurogravity_pin[:7])) if ng_head else False,
            "read_only": True,
        },
    }


def build_run_manifest(
    *,
    run_id: str,
    dataset: str,
    data_root: PathLike,
    output_root: PathLike,
    config: Optional[Mapping[str, Any]] = None,
    code_root: Optional[PathLike] = None,
    third_party_root: Optional[PathLike] = None,
    input_files: Optional[Iterable[PathLike]] = None,
    paper_mainline: bool = False,
    verification_only: bool = True,
    distance_status: Optional[str] = None,
    missing_fids: Optional[Sequence[Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    allow_backup_data_root: bool = False,
) -> Dict[str, Any]:
    """Build an immutable-style run manifest dict (caller writes to disk)."""
    root_info = validate_server_data_root(
        data_root, allow_backup=allow_backup_data_root
    )
    code_root_p = Path(code_root) if code_root else Path(__file__).resolve().parents[3]
    code_commit = _git_rev(code_root_p)

    tp_root = Path(third_party_root) if third_party_root else code_root_p / "third_party"
    tp = record_third_party_provenance(tp_root) if tp_root.exists() else {}

    fingerprints = [file_fingerprint(p) for p in (input_files or [])]

    # HK / provisional gates
    if distance_status == "provisional_wgs84_not_metric" and paper_mainline:
        raise ValueError(
            "paper_mainline=True forbidden while distance_status="
            "provisional_wgs84_not_metric"
        )
    if str(dataset).lower().startswith("hk"):
        if not verification_only or paper_mainline:
            raise ValueError(
                "HK runs must set verification_only=True and paper_mainline=False"
            )

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "dataset": dataset,
        "data_root": root_info,
        "output_root": str(_expand(output_root)),
        "config": dict(config) if config else {},
        "code": {
            "root": str(code_root_p),
            "commit": code_commit,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "third_party": tp,
        "inputs": fingerprints,
        "flags": {
            "paper_mainline": bool(paper_mainline),
            "verification_only": bool(verification_only),
            "distance_status": distance_status,
            "missing_fids": list(missing_fids) if missing_fids is not None else [],
        },
        "pins": {
            "deepgravity": DEEPGRAVITY_PIN,
            "neurogravity": NEUROGRAVITY_PIN,
        },
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def write_json(path: PathLike, obj: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    return p
