#!/usr/bin/env python3
"""Plot MITMA P2 Condition S metrics and optional day-bootstrap uncertainty."""

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


METRIC_COLUMNS = [
    "partition",
    "lambda",
    "CPC_R",
    "CPC_RT",
    "Delta",
    "CPL",
    "CPCd",
    "R2",
    "NRMSE",
    "JSD",
    "CPC_indep",
]

BOOTSTRAP_COLUMNS = ["partition", "resample", "lambda", "CPC_R", "CPC_RT", "Delta"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MITMA P2 Condition S figures."
    )
    parser.add_argument(
        "--condition-s-dir",
        type=Path,
        required=True,
        help="Condition S output directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for PNG and PDF outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed recorded in the figure metadata/title (default: 0).",
    )
    return parser.parse_args()


def _records_from_json(obj: Any, inherited_partition: str | None = None) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        records: list[dict[str, Any]] = []
        for item in obj:
            records.extend(_records_from_json(item, inherited_partition))
        return records

    if not isinstance(obj, dict):
        return []

    partition = obj.get("partition", inherited_partition)
    if any(key in obj for key in ("CPC_R", "CPC_RT", "CPC_T_lambdaR", "CPC_T_lambdaRT", "Delta", "CPC_indep")):
        record = dict(obj)
        if partition is not None:
            record["partition"] = partition
        return [record]

    records = []
    for key, value in obj.items():
        if isinstance(value, (dict, list)):
            child_partition = inherited_partition
            if isinstance(key, str) and key.lower() not in {
                "metrics",
                "partitions",
                "partition_metrics",
                "results",
            }:
                child_partition = key
            records.extend(_records_from_json(value, child_partition))
    return records


def _normalise_columns(
    frame: pd.DataFrame,
    required: list[str],
    default_partition: str | None = None,
) -> pd.DataFrame:
    frame = frame.copy()

    aliases = {
        "cpc_r": "CPC_R",
        "cpc_rt": "CPC_RT",
        # parquet / metrics_to_row schema (draft2 table names)
        "cpc_t_lambdar": "CPC_R",
        "cpc_t_lambdart": "CPC_RT",
        "cpc_t_lambda_r": "CPC_R",
        "cpc_t_lambda_rt": "CPC_RT",
        "delta": "Delta",
        "cpc_independence": "CPC_indep",
        "cpc_indep": "CPC_indep",
        "r2": "R2",
        "r_squared": "R2",
        "resample_id": "resample",
    }
    rename = {}
    for column in frame.columns:
        key = str(column)
        lower = key.lower()
        if lower in aliases:
            rename[column] = aliases[lower]
    frame = frame.rename(columns=rename)

    if "partition" not in frame.columns:
        frame["partition"] = default_partition

    if frame["partition"].isna().all() and default_partition is not None:
        frame["partition"] = default_partition

    core = [c for c in ("partition", "CPC_R", "CPC_RT", "Delta", "CPC_indep", "lambda") if c in required]
    # companions optional (fill if absent)
    for column in required:
        if column not in frame.columns:
            if column in core and column != "partition":
                raise ValueError(f"Missing required columns: {column}")
            if column != "partition":
                frame[column] = np.nan

    missing_core = [c for c in core if c not in frame.columns]
    if missing_core:
        raise ValueError(f"Missing required columns: {', '.join(missing_core)}")

    for column in required:
        if column != "partition" and column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["partition"] = frame["partition"].astype(str)
    return frame


def load_metrics(root: Path) -> pd.DataFrame:
    parquet_path = root / "partition_metrics.parquet"
    if parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
        return _normalise_columns(frame, METRIC_COLUMNS)

    records: list[dict[str, Any]] = []
    aggregate = root / "metrics.json"
    if aggregate.exists():
        records.extend(_records_from_json(json.loads(aggregate.read_text(encoding="utf-8"))))

    for path in sorted((root / "partitions").glob("*/metrics.json")):
        partition = path.parent.name
        records.extend(
            _records_from_json(
                json.loads(path.read_text(encoding="utf-8")),
                inherited_partition=partition,
            )
        )

    if not records:
        raise FileNotFoundError(
            f"No partition_metrics.parquet or partition metrics JSON found under {root}"
        )

    frame = pd.DataFrame.from_records(records)
    frame = _normalise_columns(frame, METRIC_COLUMNS)

    # Prefer one row per partition when aggregate and per-partition files overlap.
    frame = frame.dropna(subset=["CPC_R", "CPC_RT", "CPC_indep"], how="all")
    frame = frame.drop_duplicates(subset=["partition"], keep="last")
    if frame.empty:
        raise ValueError("No usable Condition S metric rows were found.")
    return frame.reset_index(drop=True)


def load_bootstrap(root: Path) -> pd.DataFrame | None:
    candidates = []
    root_file = root / "day_bootstrap.parquet"
    if root_file.exists():
        candidates.append(root_file)
    candidates.extend(sorted(root.glob("*/day_bootstrap.parquet")))
    candidates.extend(sorted(root.glob("partitions/*/day_bootstrap.parquet")))

    if not candidates:
        return None

    frames = []
    for path in candidates:
        frame = pd.read_parquet(path)
        default_partition = path.parent.name
        frame = _normalise_columns(
            frame,
            BOOTSTRAP_COLUMNS,
            default_partition=default_partition,
        )
        frames.append(frame)

    bootstrap = pd.concat(frames, ignore_index=True)
    bootstrap = bootstrap.dropna(subset=["Delta"])
    return bootstrap if not bootstrap.empty else None


def partition_order(metrics: pd.DataFrame) -> list[str]:
    return metrics["partition"].astype(str).tolist()


def choose_full_partition(bootstrap: pd.DataFrame) -> str:
    preferred = {
        "full",
        "all",
        "total",
        "unsliced",
        "weekday_total",
        "weekday total",
    }
    for partition in bootstrap["partition"].astype(str).drop_duplicates():
        if partition.strip().lower() in preferred:
            return partition
    return str(bootstrap["partition"].iloc[0])


def bootstrap_intervals(
    bootstrap: pd.DataFrame,
    partitions: list[str],
) -> dict[str, tuple[float, float]]:
    intervals: dict[str, tuple[float, float]] = {}
    for partition in partitions:
        values = bootstrap.loc[
            bootstrap["partition"].astype(str) == partition,
            "Delta",
        ].dropna()
        if len(values):
            intervals[partition] = (
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            )
    return intervals


def annotate_bars(
    ax: plt.Axes,
    bars: list[Any],
    deltas: pd.Series,
) -> None:
    finite_deltas = pd.to_numeric(deltas, errors="coerce")
    for bar, delta in zip(bars, finite_deltas):
        if not np.isfinite(delta):
            continue
        height = max(float(bar.get_height()), 0.0)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"Delta={delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
            clip_on=False,
        )


def make_figure(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame | None,
    seed: int,
) -> plt.Figure:
    partitions = partition_order(metrics)
    x = np.arange(len(partitions))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"MITMA P2 Condition S | Seed {seed}",
        fontsize=16,
        fontweight="bold",
    )

    # Panel 1: CPC comparison.
    ax = axes[0, 0]
    bars_r = ax.bar(
        x - width,
        metrics["CPC_R"],
        width,
        label="CPC(T, lambda R)",
        color="#4C78A8",
    )
    bars_rt = ax.bar(
        x,
        metrics["CPC_RT"],
        width,
        label="CPC(T, lambda R^T)",
        color="#F58518",
    )
    bars_indep = ax.bar(
        x + width,
        metrics["CPC_indep"],
        width,
        label="CPC independence",
        color="#54A24B",
    )
    annotate_bars(ax, list(bars_r), metrics["Delta"])
    ax.set_ylabel("CPC")
    ax.set_title("Condition S overlap by partition")
    ax.set_xticks(x)
    ax.set_xticklabels(partitions, rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)

    # Panel 2: Delta forest/point plot.
    ax = axes[0, 1]
    intervals = (
        bootstrap_intervals(bootstrap, partitions)
        if bootstrap is not None
        else {}
    )
    y = np.arange(len(partitions))
    delta_values = metrics["Delta"].to_numpy(dtype=float)

    lower_errors = []
    upper_errors = []
    has_ci = []
    for partition, value in zip(partitions, delta_values):
        if partition in intervals and np.isfinite(value):
            low, high = intervals[partition]
            lower_errors.append(value - low)
            upper_errors.append(high - value)
            has_ci.append(True)
        else:
            lower_errors.append(0.0)
            upper_errors.append(0.0)
            has_ci.append(False)

    error = np.vstack([lower_errors, upper_errors])
    ax.errorbar(
        delta_values,
        y,
        xerr=error,
        fmt="o",
        color="#2F4B7C",
        ecolor="#7A7A7A",
        elinewidth=2,
        capsize=4,
        markersize=6,
    )
    ax.axvline(0, color="black", linewidth=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(partitions)
    ax.invert_yaxis()
    ax.set_xlabel("Delta = CPC(T, lambda R^T) - CPC(T, lambda R)")
    ax.set_title("Delta by partition")
    if any(has_ci):
        ax.text(
            0.02,
            0.02,
            "Error bars: bootstrap 2.5th-97.5th percentiles",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )
    ax.grid(axis="x", alpha=0.25)

    # Panel 3: companion metrics.
    ax = axes[1, 0]
    companion = ["CPL", "CPCd", "R2", "NRMSE", "JSD"]
    available = [column for column in companion if column in metrics.columns]
    if available:
        metric_matrix = metrics[available].to_numpy(dtype=float)
        offsets = np.linspace(-0.35, 0.35, len(available))
        for offset, column_index, name in zip(offsets, range(len(available)), available):
            ax.scatter(
                x + offset,
                metric_matrix[:, column_index],
                s=35,
                label=name,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(partitions, rotation=45, ha="right")
        ax.set_ylabel("Metric value")
        ax.set_title("Companion Condition S metrics")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No companion metrics available", ha="center", va="center")

    # Panel 4: optional full-partition bootstrap histogram.
    ax = axes[1, 1]
    if bootstrap is None:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No day-bootstrap data available",
            ha="center",
            va="center",
        )
    else:
        full_partition = choose_full_partition(bootstrap)
        values = bootstrap.loc[
            bootstrap["partition"].astype(str) == full_partition,
            "Delta",
        ].dropna()
        if values.empty:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No full-partition Delta bootstrap data",
                ha="center",
                va="center",
            )
        else:
            ax.hist(
                values,
                bins=min(40, max(10, int(np.sqrt(len(values))))),
                color="#937860",
                alpha=0.8,
                edgecolor="white",
            )
            ax.axvline(0, color="black", linewidth=1)
            ax.axvline(
                float(values.mean()),
                color="#D62728",
                linestyle="--",
                linewidth=1.5,
                label=f"mean={values.mean():+.3f}",
            )
            ax.set_xlabel("Bootstrap Delta")
            ax.set_ylabel("Count")
            ax.set_title(f"Delta bootstrap: {full_partition}")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.condition_s_dir)
    bootstrap = load_bootstrap(args.condition_s_dir)
    figure = make_figure(metrics, bootstrap, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / "p2_mitma_condition_s.png"
    pdf_path = args.out_dir / "p2_mitma_condition_s.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
