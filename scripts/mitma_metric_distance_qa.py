#!/usr/bin/env python3
"""Generate the P1 MITMA metric-distance QA figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator


FIGURE_BASENAME = "p1_mitma_metric_distance_qa"
KDE_MAX_POINTS = 100_000
KDE_GRID_POINTS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot MITMA metric-distance QA: off-diagonal distribution, "
            "diagonal distances, and a reproducible 80x80 sub-block."
        )
    )
    parser.add_argument(
        "--distance-npy",
        type=Path,
        required=True,
        help="Path to the square distance matrix D in metres (.npy).",
    )
    parser.add_argument(
        "--areas-npy",
        type=Path,
        default=None,
        help="Optional length-N polygon-area vector in square metres (.npy).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory in which the PNG and PDF are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for KDE subsampling and the 80x80 sub-block (default: 0).",
    )
    return parser.parse_args()


def load_distance(path: Path) -> np.ndarray:
    try:
        distance = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not load distance matrix {path}: {exc}") from exc

    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise SystemExit(
            f"Distance matrix must be square 2-D; got shape {distance.shape}."
        )
    if distance.shape[0] < 80:
        raise SystemExit(
            f"Distance matrix has N={distance.shape[0]}; N >= 80 is required."
        )
    if not np.issubdtype(distance.dtype, np.number):
        raise SystemExit(f"Distance matrix must be numeric; got dtype {distance.dtype}.")
    if not np.isfinite(distance).all():
        raise SystemExit("Distance matrix contains non-finite values.")
    if np.any(distance < 0):
        raise SystemExit("Distance matrix contains negative distances.")
    return distance


def load_areas(path: Path, n: int) -> np.ndarray:
    try:
        areas = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not load area vector {path}: {exc}") from exc

    areas = np.asarray(areas)
    if areas.ndim != 1 or areas.shape[0] != n:
        raise SystemExit(f"Areas must have shape ({n},); got shape {areas.shape}.")
    if not np.issubdtype(areas.dtype, np.number):
        raise SystemExit(f"Areas must be numeric; got dtype {areas.dtype}.")
    if not np.isfinite(areas).all() or np.any(areas < 0):
        raise SystemExit("Areas must be finite and non-negative.")
    return areas.astype(np.float64, copy=False)


def off_diagonal_values(distance: np.ndarray) -> np.ndarray:
    n = distance.shape[0]
    mask = ~np.eye(n, dtype=bool)
    values = np.asarray(distance[mask], dtype=np.float64)
    invalid = values <= 0
    if np.any(invalid):
        raise SystemExit(
            "Off-diagonal distances must be strictly positive for a log-scale plot; "
            f"found {int(invalid.sum()):,} non-positive values."
        )
    return values


def log_kde(
    values: np.ndarray,
    rng: np.random.Generator,
    max_points: int = KDE_MAX_POINTS,
    grid_points: int = KDE_GRID_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate density in log10-space and transform it back to metres."""
    log_values = np.log10(values)
    if log_values.size > max_points:
        sample = log_values[rng.choice(log_values.size, size=max_points, replace=False)]
    else:
        sample = log_values

    lo = float(np.min(log_values))
    hi = float(np.max(log_values))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    grid = np.linspace(lo, hi, grid_points)

    if sample.size > 1:
        spread = float(np.std(sample, ddof=1))
        bandwidth = 1.06 * spread * sample.size ** (-1.0 / 5.0)
    else:
        bandwidth = 0.0
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = max((hi - lo) * 0.02, 0.01)

    kernel_sum = np.zeros_like(grid)
    normalizer = np.sqrt(2.0 * np.pi) * bandwidth * sample.size
    chunk_size = 8_192
    for start in range(0, sample.size, chunk_size):
        chunk = sample[start : start + chunk_size]
        z = (grid[:, None] - chunk[None, :]) / bandwidth
        kernel_sum += np.exp(-0.5 * z * z).sum(axis=1)
    density_log10 = kernel_sum / normalizer

    x = np.power(10.0, grid)
    density_metres = density_log10 / (x * np.log(10.0))
    return x, density_metres


def meter_formatter(value: float, _position: float) -> str:
    if value == 0:
        return "0"
    return f"{value:,.0f}"


def make_figure(
    distance: np.ndarray,
    areas: np.ndarray | None,
    seed: int,
) -> plt.Figure:
    n = distance.shape[0]
    seed_sequence = np.random.SeedSequence(seed)
    kde_rng, block_rng = [np.random.default_rng(s) for s in seed_sequence.spawn(2)]

    offdiag = off_diagonal_values(distance)
    quantiles = {
        "p50": float(np.quantile(offdiag, 0.50)),
        "p90": float(np.quantile(offdiag, 0.90)),
        "p99": float(np.quantile(offdiag, 0.99)),
    }

    matrix_diagonal = np.asarray(distance.diagonal(), dtype=np.float64)
    if np.any(matrix_diagonal <= 0):
        raise SystemExit("Diagonal distances must be strictly positive.")

    area_diagonal = None
    if areas is not None:
        area_diagonal = np.sqrt(areas / np.pi)
        discrepancy = np.abs(area_diagonal - matrix_diagonal)
        print(
            "Area-derived vs matrix diagonal absolute difference: "
            f"median={np.median(discrepancy):.6g} m, "
            f"max={np.max(discrepancy):.6g} m"
        )

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), constrained_layout=True)
    fig.suptitle(
        "MITMA metric-distance QA | CRS EPSG:25830 | distances in metres",
        fontsize=15,
        fontweight="bold",
    )

    # Panel A: off-diagonal distances on a logarithmic x-axis.
    ax = axes[0]
    bins = np.logspace(np.log10(offdiag.min()), np.log10(offdiag.max()), 80)
    ax.hist(
        offdiag,
        bins=bins,
        density=True,
        color="#4C78A8",
        alpha=0.45,
        edgecolor="white",
        linewidth=0.25,
        label="off-diagonal histogram",
    )
    kde_x, kde_y = log_kde(offdiag, kde_rng)
    ax.plot(kde_x, kde_y, color="#D62728", linewidth=2.0, label="KDE (log10 space)")
    line_styles = {"p50": ("#2CA02C", "--"), "p90": ("#FF7F0E", "--"), "p99": ("#9467BD", "--")}
    for name, value in quantiles.items():
        color, linestyle = line_styles[name]
        ax.axvline(value, color=color, linestyle=linestyle, linewidth=1.5, label=f"{name} = {value:,.0f} m")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(FuncFormatter(meter_formatter))
    ax.set_xlabel("Off-diagonal distance $d_{ij}$ (m)")
    ax.set_ylabel("Density")
    ax.set_title("Off-diagonal distances")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="best")

    # Panel B: diagonal distances, with optional area-formula QA overlay.
    ax = axes[1]
    diag_min = float(matrix_diagonal.min())
    diag_max = float(matrix_diagonal.max())
    diag_bins = np.linspace(diag_min, diag_max, 41) if diag_min < diag_max else 20
    ax.hist(
        matrix_diagonal,
        bins=diag_bins,
        color="#59A14F",
        alpha=0.65,
        edgecolor="white",
        linewidth=0.35,
        label="$D_{ii}$",
    )
    if area_diagonal is not None:
        ax.hist(
            area_diagonal,
            bins=diag_bins,
            histtype="step",
            color="#B279A2",
            linewidth=1.8,
            label="$\\sqrt{A_i/\\pi}$",
        )
    ax.xaxis.set_major_formatter(FuncFormatter(meter_formatter))
    ax.set_xlabel("Diagonal distance $d_{ii}$ (m)")
    ax.set_ylabel("Count")
    ax.set_title("Intra-zonal distances")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=9, loc="best")

    # Panel C: reproducible random 80x80 sub-block.
    ax = axes[2]
    indices = np.sort(block_rng.choice(n, size=80, replace=False))
    block = np.asarray(distance[np.ix_(indices, indices)], dtype=np.float64)
    positive = block[block > 0]
    heatmap = plt.get_cmap("magma").copy()
    heatmap.set_bad("#E6E6E6")
    masked_block = np.ma.masked_less_equal(block, 0)
    image = ax.imshow(
        masked_block,
        cmap=heatmap,
        norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())),
        interpolation="nearest",
        aspect="equal",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Distance (m)")
    ax.set_xlabel("Sampled destination index")
    ax.set_ylabel("Sampled origin index")
    ax.set_title("Random 80x80 distance sub-block")
    ax.text(
        0.02,
        0.98,
        f"seed={seed}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.5},
    )

    return fig


def main() -> None:
    args = parse_args()
    distance = load_distance(args.distance_npy)
    areas = load_areas(args.areas_npy, distance.shape[0]) if args.areas_npy else None
    figure = make_figure(distance, areas, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / f"{FIGURE_BASENAME}.png"
    pdf_path = args.out_dir / f"{FIGURE_BASENAME}.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
