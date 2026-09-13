#!/usr/bin/env python3
"""Plot MITMA P3 Condition G zero-shot metrics (CPC bars; Oracle diagnostic)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIAGNOSTIC_MODELS = {"Oracle", "oracle"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MITMA P3 Condition G figures.")
    p.add_argument("--condition-g-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_rows(root: Path) -> pd.DataFrame:
    candidates = [
        root / "model_metrics.parquet",
        root / "metrics.parquet",
        root / "partition_metrics.parquet",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_parquet(path)
            return _normalise(frame)

    rows: list[dict[str, Any]] = []
    agg = root / "metrics.json"
    if agg.exists():
        obj = json.loads(agg.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            rows.extend(obj)
        elif isinstance(obj, dict):
            if "by_model" in obj and isinstance(obj["by_model"], dict):
                for model, payload in obj["by_model"].items():
                    rec = dict(payload) if isinstance(payload, dict) else {}
                    rec.setdefault("law_model", model)
                    rows.append(rec)
            elif "models" in obj and isinstance(obj["models"], list):
                rows.extend(obj["models"])
            elif "CPC" in obj or "law_model" in obj:
                rows.append(obj)

    models_dir = root / "models"
    if models_dir.exists():
        for path in sorted(models_dir.glob("*/metrics.json")):
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                obj.setdefault("law_model", path.parent.name)
                rows.append(obj)

    # also flat */metrics.json under root
    for path in sorted(root.glob("*/metrics.json")):
        if path.parent.name in {"partitions", "models"}:
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            obj.setdefault("law_model", path.parent.name)
            rows.append(obj)

    if not rows:
        raise SystemExit(f"No Condition G metrics found under {root}")
    return _normalise(pd.DataFrame(rows))


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    aliases = {
        "model": "law_model",
        "name": "law_model",
        "cpc": "CPC",
        "r2": "R2",
        "r_squared": "R2",
        "pearson_r": "pearson",
        "corr": "pearson",
        "correlation": "pearson",
    }
    rename = {}
    for col in frame.columns:
        low = str(col).lower()
        if low in aliases:
            rename[col] = aliases[low]
    frame = frame.rename(columns=rename)
    if "law_model" not in frame.columns:
        raise SystemExit(f"missing law_model column; have {list(frame.columns)}")
    for col in ("CPC", "R2", "pearson"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "CPC" not in frame.columns:
        raise SystemExit("missing CPC column")
    frame["law_model"] = frame["law_model"].astype(str)
    # dedupe by model keeping first
    frame = frame.drop_duplicates(subset=["law_model"], keep="first").reset_index(drop=True)
    if "diagnostic" in frame.columns:
        diag_flag = frame["diagnostic"].astype(bool)
    else:
        diag_flag = pd.Series(False, index=frame.index)
    frame["diagnostic"] = diag_flag | frame["law_model"].isin(DIAGNOSTIC_MODELS)
    return frame


def plot_cpc_bars(frame: pd.DataFrame, out_dir: Path, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = frame.sort_values("CPC", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#9e9e9e" if d else "#1976d2" for d in order["diagnostic"]]
    ax.bar(order["law_model"], order["CPC"], color=colors)
    ax.set_ylabel("CPC")
    ax.set_title(f"MITMA P3 Condition G CPC (seed={seed}; grey=diagnostic)")
    ax.set_ylim(0, max(1.0, float(np.nanmax(order["CPC"])) * 1.1))
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    png = out_dir / "p3_condition_g_cpc_bars.png"
    pdf = out_dir / "p3_condition_g_cpc_bars.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def plot_companion_bars(frame: pd.DataFrame, out_dir: Path, seed: int) -> Path | None:
    cols = [c for c in ("CPC", "R2", "pearson") if c in frame.columns]
    if len(cols) < 2:
        return None
    order = frame.sort_values("CPC", ascending=False)
    x = np.arange(len(order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, col in enumerate(cols):
        ax.bar(x + (i - 1) * width, order[col], width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(order["law_model"], rotation=35, ha="right")
    ax.set_ylabel("score")
    ax.set_title(f"MITMA P3 Condition G companions (seed={seed})")
    ax.legend()
    fig.tight_layout()
    png = out_dir / "p3_condition_g_companions.png"
    fig.savefig(png, dpi=150)
    fig.savefig(out_dir / "p3_condition_g_companions.pdf")
    plt.close(fig)
    return png


def main() -> None:
    args = parse_args()
    frame = _load_rows(args.condition_g_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "p3_condition_g_metrics_used.csv", index=False)
    paths = [plot_cpc_bars(frame, args.out_dir, args.seed)]
    companion = plot_companion_bars(frame, args.out_dir, args.seed)
    if companion is not None:
        paths.append(companion)
    meta = {
        "ticket": "P2-FIG-FIX",
        "n_models": int(len(frame)),
        "models": frame["law_model"].tolist(),
        "outputs": [str(p) for p in paths],
        "seed": args.seed,
    }
    (args.out_dir / "figure_manifest.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
