#!/usr/bin/env python3
"""P2-EQUIV-FIX: serial (--n-jobs 1) day-bootstrap vs frozen Condition S ref.

Uses public ``day_bootstrap`` only (no private aggregate helpers). Compares
day indices + λ / CPC_R / CPC_RT / Δ summaries to the reference run under
``outputs/mainline/mitma/condition_s/`` and writes
``outputs/qa/mitma_cs_equiv_report.json``.

Example (compute node):

  python scripts/check_p2_njobs_equiv.py \\
    --materialised outputs/materialised/mitma_cs \\
    --ref-dir outputs/mainline/mitma/condition_s \\
    --out outputs/qa/mitma_cs_equiv_report.json \\
    --n-resamples 1000 --seed 0 --n-jobs 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from attrOD.metrics.condition_s import day_bootstrap


def _load_day_stack(folder: Path) -> Optional[List[np.ndarray]]:
    if not folder.exists():
        return None
    files = sorted(folder.glob("*.npy"))
    if len(files) < 2:
        return None
    return [np.load(f) for f in files]


def _load_distances(mat: Path) -> np.ndarray:
    for name in ("distances.npy", "distance.npy"):
        p = mat / name
        if p.exists():
            return np.load(p)
    raise SystemExit(f"no distances under {mat}")


def _find_ref_bootstrap(ref_dir: Path) -> tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Return (samples_df, summary_dict) from reference Condition S outputs."""
    samples = None
    summary = None

    for path in (
        ref_dir / "day_bootstrap.parquet",
        ref_dir / "partitions" / "full" / "day_bootstrap.parquet",
        ref_dir / "partitions" / "total" / "day_bootstrap.parquet",
    ):
        if path.exists():
            samples = pd.read_parquet(path)
            break

    for path in (
        ref_dir / "day_bootstrap_summary.json",
        ref_dir / "partitions" / "full" / "qa.json",
        ref_dir / "partitions" / "full" / "metrics.json",
    ):
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            if "bootstrap_summary" in obj:
                summary = obj["bootstrap_summary"]
                break
            if "bootstrap" in obj and isinstance(obj["bootstrap"], dict):
                summary = obj["bootstrap"].get("bootstrap_summary") or obj["bootstrap"]
                break
            # flat mean_/p2.5_ keys
            if any(k.startswith("mean_") for k in obj):
                summary = obj
                break
    return samples, summary


def _summary_from_samples(df: pd.DataFrame, keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in keys:
        if k not in df.columns:
            continue
        arr = pd.to_numeric(df[k], errors="coerce").to_numpy(dtype=float)
        out[f"mean_{k}"] = float(np.nanmean(arr))
        out[f"p2.5_{k}"] = float(np.nanpercentile(arr, 2.5))
        out[f"p97.5_{k}"] = float(np.nanpercentile(arr, 97.5))
    return out


def _compare_summaries(
    a: Dict[str, float], b: Dict[str, float], *, atol: float, rtol: float
) -> Dict[str, Any]:
    keys = sorted(set(a) & set(b))
    diffs = {}
    ok = True
    for k in keys:
        va, vb = float(a[k]), float(b[k])
        if not (np.isfinite(va) and np.isfinite(vb)):
            match = bool(np.isnan(va) and np.isnan(vb))
        else:
            match = abs(va - vb) <= (atol + rtol * max(abs(va), abs(vb)))
        diffs[k] = {"a": va, "b": vb, "abs_diff": abs(va - vb) if np.isfinite(va) and np.isfinite(vb) else None, "match": match}
        ok = ok and match
    return {"ok": ok, "n_keys": len(keys), "per_key": diffs}


def main() -> None:
    p = argparse.ArgumentParser(description="P2 serial vs ref day-bootstrap equivalence")
    p.add_argument("--materialised", required=True)
    p.add_argument("--ref-dir", required=True, help="frozen Condition S scores dir")
    p.add_argument("--out", default="outputs/qa/mitma_cs_equiv_report.json")
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=1, help="usually 1 for serial check")
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--rtol", type=float, default=1e-7)
    p.add_argument(
        "--also-parallel-jobs",
        type=int,
        default=0,
        help="if >1, also compare live n_jobs=1 vs this n_jobs on same draws",
    )
    args = p.parse_args()

    mat = Path(args.materialised)
    ref_dir = Path(args.ref_dir)
    T_by_day = _load_day_stack(mat / "T_by_day")
    R_by_day = _load_day_stack(mat / "R_by_day")
    if T_by_day is None or R_by_day is None:
        raise SystemExit("need T_by_day and R_by_day under materialised (≥2 days)")
    if len(T_by_day) != len(R_by_day):
        raise SystemExit("T_by_day / R_by_day length mismatch")
    distances = _load_distances(mat)

    serial = day_bootstrap(
        T_by_day,
        R_by_day,
        distances=distances,
        n_resamples=args.n_resamples,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    ref_samples, ref_summary = _find_ref_bootstrap(ref_dir)
    keys = ("lambda", "CPC_R", "CPC_RT", "Delta")
    serial_summary = {k: float(serial["bootstrap_summary"][k]) for k in serial["bootstrap_summary"]}

    report: Dict[str, Any] = {
        "ticket": "P2-EQUIV-FIX",
        "materialised": str(mat),
        "ref_dir": str(ref_dir),
        "n_days": int(serial["n_days"]),
        "n_resamples": int(serial["n_resamples"]),
        "seed": int(serial["seed"]),
        "n_jobs": int(serial.get("n_jobs", args.n_jobs)),
        "pass": False,
        "checks": {},
    }

    # day-index identity vs ref samples if present
    if ref_samples is not None and "day_indices" in ref_samples.columns:
        # day_indices may be list-like
        ref_sorted = ref_samples.sort_values("resample") if "resample" in ref_samples.columns else ref_samples
        n = min(len(ref_sorted), len(serial["bootstrap_samples"]))
        mismatches = 0
        for i in range(n):
            a = list(serial["bootstrap_samples"][i]["day_indices"])
            raw = ref_sorted.iloc[i]["day_indices"]
            if isinstance(raw, str):
                try:
                    b = list(json.loads(raw))
                except Exception:
                    b = list(raw)
            else:
                b = list(raw)
            if list(map(int, a)) != list(map(int, b)):
                mismatches += 1
        report["checks"]["day_indices_vs_ref"] = {
            "ok": mismatches == 0,
            "n_compared": n,
            "n_mismatch": mismatches,
        }
    else:
        report["checks"]["day_indices_vs_ref"] = {
            "ok": None,
            "note": "ref day_indices not available; skipped",
        }

    # summary vs ref
    if ref_summary is None and ref_samples is not None:
        # map possible parquet schema names
        rs = ref_samples.rename(
            columns={
                "CPC_T_lambdaR": "CPC_R",
                "CPC_T_lambdaRT": "CPC_RT",
            }
        )
        ref_summary = _summary_from_samples(rs, keys)

    if ref_summary is not None:
        # normalise keys
        norm = {}
        for k, v in ref_summary.items():
            lk = str(k)
            lk = lk.replace("CPC_T_lambdaR", "CPC_R").replace("CPC_T_lambdaRT", "CPC_RT")
            norm[lk] = float(v) if np.isscalar(v) else v
        report["checks"]["summary_vs_ref"] = _compare_summaries(
            serial_summary, {k: float(norm[k]) for k in norm if isinstance(norm[k], (int, float, np.floating)) and k in serial_summary},
            atol=args.atol,
            rtol=args.rtol,
        )
    else:
        report["checks"]["summary_vs_ref"] = {
            "ok": None,
            "note": "no ref bootstrap summary/samples found",
        }

    # optional live 1 vs N
    if args.also_parallel_jobs and args.also_parallel_jobs > 1:
        parallel = day_bootstrap(
            T_by_day,
            R_by_day,
            distances=distances,
            n_resamples=args.n_resamples,
            seed=args.seed,
            n_jobs=args.also_parallel_jobs,
        )
        idx_ok = all(
            list(serial["bootstrap_samples"][i]["day_indices"])
            == list(parallel["bootstrap_samples"][i]["day_indices"])
            for i in range(args.n_resamples)
        )
        sum_cmp = _compare_summaries(
            serial_summary,
            {k: float(parallel["bootstrap_summary"][k]) for k in parallel["bootstrap_summary"]},
            atol=args.atol,
            rtol=args.rtol,
        )
        report["checks"]["live_serial_vs_parallel"] = {
            "ok": bool(idx_ok and sum_cmp["ok"]),
            "day_indices_match": bool(idx_ok),
            "summary": sum_cmp,
            "parallel_n_jobs": int(args.also_parallel_jobs),
        }

    # overall pass: require summary_vs_ref ok when present; else live check
    oks = [c.get("ok") for c in report["checks"].values() if c.get("ok") is not None]
    report["pass"] = bool(oks) and all(oks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
