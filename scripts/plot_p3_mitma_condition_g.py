#!/usr/bin/env python3
"""Plot MITMA P3 Condition G (PLOT-AUDIT-FIX).

Horizontal CPC bars sorted desc; optional R2/pearson companions.
Single basename p3_mitma_condition_g.png/.pdf. Oracle diagnostic.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Generate MITMA P3 Condition G figures.")
    p.add_argument("--condition-g-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"", "0", "false", "no", "n", "off", "none", "null"}:
        return False
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    return False

def _is_oracle(name: str) -> bool:
    return str(name).strip().lower() == "oracle"

def _looks_like_score_row(obj: dict[str, Any]) -> bool:
    keys = {str(k).lower() for k in obj}
    return bool(keys & {"cpc", "law_model", "model", "name"}) and (
        "cpc" in keys or "law_model" in keys or "model" in keys
    )

def _walk_json(obj: Any, inherited_model: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(obj, list):
        for item in obj:
            rows.extend(_walk_json(item, inherited_model))
        return rows
    if not isinstance(obj, dict):
        return rows
    rec = dict(obj)
    if inherited_model and "law_model" not in rec and "model" not in rec:
        rec["law_model"] = inherited_model
    if _looks_like_score_row(rec):
        rows.append(rec)
    for key, value in obj.items():
        low = str(key).lower()
        if low in {"by_model", "models", "results", "scores", "metrics"}:
            if isinstance(value, dict) and low == "by_model":
                for model, payload in value.items():
                    rows.extend(_walk_json(payload, str(model)))
            else:
                rows.extend(_walk_json(value, inherited_model))
        elif isinstance(value, (dict, list)):
            rows.extend(_walk_json(value, inherited_model))
    return rows

def _load_rows(root: Path) -> pd.DataFrame:
    for name in (
        "model_metrics.parquet", "metrics.parquet", "condition_g_metrics.parquet",
        "scores.parquet", "partition_metrics.parquet", "summary.parquet",
    ):
        path = root / name
        if path.exists():
            return _normalise(pd.read_parquet(path))
    for path in sorted(root.glob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        cols = {str(c).lower() for c in frame.columns}
        if "cpc" in cols and ({"law_model", "model", "name"} & cols):
            return _normalise(frame)
    rows: list[dict[str, Any]] = []
    for name in ("metrics.json", "qa.json", "summary.json", "model_metrics.json"):
        path = root / name
        if path.exists():
            rows.extend(_walk_json(json.loads(path.read_text(encoding="utf-8"))))
    for path in sorted(root.glob("**/*.json")):
        if path.name in {"run_manifest.json", "figure_manifest.json"}:
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        inherited = None if path.parent == root else path.parent.name
        for rec in _walk_json(obj, inherited):
            if inherited and "law_model" not in rec and "model" not in rec:
                rec["law_model"] = inherited
            rows.append(rec)
    if not rows:
        raise SystemExit(f"No Condition G metrics found under {root}")
    return _normalise(pd.DataFrame(rows))

def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    colmap = {str(c).lower(): c for c in frame.columns}
    def take(*names: str):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None
    cols = [take("law_model", "model", "name")]
    # coalesce all name-like columns (mixed long/wide exports)
    name_cols = [c for c in ("law_model", "model", "name") if c in frame.columns or take(c)]
    name_cols = []
    for n in ("law_model", "model", "name"):
        c = take(n)
        if c is not None and c not in name_cols:
            name_cols.append(c)
    if not name_cols:
        raise SystemExit(f"missing law_model column; have {list(frame.columns)}")
    series = frame[name_cols[0]]
    for c in name_cols[1:]:
        series = series.fillna(frame[c])
    frame["law_model"] = series.astype(str)
    rename = {}
    for col in frame.columns:
        low = str(col).lower()
        if low == "cpc":
            rename[col] = "CPC"
        elif low in {"r2", "r_squared"}:
            rename[col] = "R2"
        elif low in {"pearson", "pearson_r", "corr", "correlation"}:
            rename[col] = "pearson"
        elif low == "diagnostic":
            rename[col] = "diagnostic"
    frame = frame.rename(columns=rename)
    for col in ("CPC", "R2", "pearson"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "CPC" not in frame.columns:
        raise SystemExit("missing CPC column")
    frame = frame.drop_duplicates(subset=["law_model"], keep="first").reset_index(drop=True)
    if "diagnostic" in frame.columns:
        diag_flag = frame["diagnostic"].map(_as_bool)
    else:
        diag_flag = pd.Series(False, index=frame.index)
    frame["diagnostic"] = diag_flag.to_numpy() | frame["law_model"].map(_is_oracle).to_numpy()
    return frame

def plot_composite(frame: pd.DataFrame, out_dir: Path, seed: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = frame.sort_values("CPC", ascending=False).reset_index(drop=True)
    companions = [c for c in ("R2", "pearson") if c in order.columns and order[c].notna().any()]
    n_panels = 1 + (1 if companions else 0)
    fig_h = max(4.5, 0.35 * len(order) + 1.5)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, fig_h), sharey=True, squeeze=False)
    ax0 = axes[0, 0]
    colors = ["#9e9e9e" if d else "#1976d2" for d in order["diagnostic"]]
    y = np.arange(len(order))
    ax0.barh(y, order["CPC"], color=colors)
    ax0.set_yticks(y)
    ax0.set_yticklabels(order["law_model"])
    ax0.invert_yaxis()
    ax0.set_xlabel("CPC")
    ax0.set_title("CPC (grey = diagnostic / Oracle)")
    ax0.set_xlim(0, max(1.0, float(np.nanmax(order["CPC"])) * 1.05))
    if companions:
        ax1 = axes[0, 1]
        height = 0.35 if len(companions) > 1 else 0.6
        for i, col in enumerate(companions):
            offset = (i - (len(companions) - 1) / 2) * height
            ax1.barh(y + offset, order[col], height=height * 0.9, label=col)
        ax1.set_xlabel("score")
        ax1.set_title("Companions")
        ax1.legend(loc="lower right")
        ax1.set_xlim(left=min(0.0, float(np.nanmin(order[companions].to_numpy()))))
    fig.suptitle(f"MITMA P3 Condition G (seed={seed})", y=1.02)
    fig.tight_layout()
    png = out_dir / "p3_mitma_condition_g.png"
    pdf = out_dir / "p3_mitma_condition_g.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]

def main() -> None:
    args = parse_args()
    frame = _load_rows(args.condition_g_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "p3_mitma_condition_g_metrics_used.csv", index=False)
    paths = plot_composite(frame, args.out_dir, args.seed)
    meta = {
        "ticket": "PLOT-AUDIT-FIX",
        "n_models": int(len(frame)),
        "models": frame["law_model"].tolist(),
        "outputs": [str(p) for p in paths],
        "seed": args.seed,
        "basename": "p3_mitma_condition_g",
    }
    (args.out_dir / "figure_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
