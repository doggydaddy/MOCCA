#!/usr/bin/env python3
"""Support figure: percolation calibration curve for a bundle-FWER cluster-forming
threshold. Reads the output of 02_cudaPerm/percolation_calibration.py and renders
the null-only percolation order parameter (giant-bundle voxel fraction, primary/
gating) alongside the giant-bundle edge fraction (diagnostic only, retained to show
why it was rejected as the gating metric -- see 02_cudaPerm/README.md).

Usage:
    .venv/bin/python 03_prepResultsForVisualization/plot_percolation_calibration.py \\
        /mnt/storage/MOCCA_UCLA/percolation_calibration_controlsVSpatients_200 \\
        03_prepResultsForVisualization/figures/percolation_calibration_controlsVSpatients.png \\
        --dataset-label "Controls vs. patients"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dataviz reference palette (references/palette.md) -- light-mode chart chrome.
BLUE = "#2a78d6"      # categorical slot 1 -- primary/gating series
ORANGE = "#eb6834"    # categorical slot 2 -- diagnostic series
RED = "#e34948"       # categorical slot 8 -- recommended threshold marker
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def load_run(result_dir: Path):
    summary = pd.read_csv(result_dir / "percolation_calibration_summary.csv")
    curve = pd.read_csv(result_dir / "percolation_calibration_curve.csv")
    with open(result_dir / "percolation_calibration_results.json") as f:
        results = json.load(f)
    return summary, curve, results


def jittered_x(p_values: np.ndarray, rng: np.random.Generator, spread: float = 0.06):
    """Log-space jitter so per-permutation points at one threshold don't overplot."""
    return p_values * 10 ** rng.uniform(-spread, spread, size=p_values.shape)


def fmt_p(v: float) -> str:
    exp = int(np.floor(np.log10(v)))
    mant = v / 10**exp
    mant_str = f"{mant:g}"
    return f"{mant_str}×10$^{{{exp}}}$"


def draw_panel(ax, summary, curve, results, *, value_col, pct_col, mean_col,
                ylabel, title, show_epsilon, show_thresholds, point_color):
    rng = np.random.default_rng(0)
    x_jit = jittered_x(curve["cluster_forming_p"].to_numpy(), rng)
    ax.scatter(
        x_jit, curve[value_col], s=5, color=point_color, alpha=0.15,
        linewidths=0, zorder=1, label="individual null permutations",
    )

    ax.plot(
        summary["cluster_forming_p"], summary[mean_col],
        color=INK_SECONDARY, linestyle="--", linewidth=1.5, marker="none",
        zorder=2, label="mean",
    )
    ax.plot(
        summary["cluster_forming_p"], summary[pct_col],
        color=BLUE, linestyle="-", linewidth=2, marker="o", markersize=4,
        zorder=3, label=f"{results['percentile']:.0f}th percentile",
    )

    if show_epsilon:
        ax.axhline(
            results["epsilon"], color=INK_MUTED, linestyle=":", linewidth=1.2,
            zorder=1,
        )
        ax.text(
            summary["cluster_forming_p"].min() * 0.85, results["epsilon"],
            f"ε = {results['epsilon']:g} (sub-critical criterion)",
            color=INK_MUTED, fontsize=8, va="bottom", ha="right",
        )

    if show_thresholds:
        transition = results["transition_p_cf"]
        recommended = results["recommended_p_cf"]
        ax.axvline(transition, color=ORANGE, linestyle="--", linewidth=1.2, zorder=1)
        ax.axvline(recommended, color=RED, linestyle="-", linewidth=1.5, zorder=1)
        # Labels extend away from each other (transition leftward, recommended
        # rightward) since the two thresholds sit only one grid step apart on
        # the log axis and would otherwise collide.
        ax.annotate(
            f"transition, p_CF={fmt_p(transition)}",
            xy=(transition, 1), xycoords=("data", "axes fraction"),
            xytext=(-4, -4), textcoords="offset points",
            rotation=90, color=ORANGE, fontsize=7.5, va="top", ha="right",
        )
        ax.annotate(
            f"recommended, p_CF={fmt_p(recommended)}",
            xy=(recommended, 1), xycoords=("data", "axes fraction"),
            xytext=(4, -4), textcoords="offset points",
            rotation=90, color=RED, fontsize=7.5, va="top", ha="left",
        )

    ax.set_xscale("log")
    ax.invert_xaxis()  # liberal (large p_CF) -> strict (small p_CF), left to right
    ax.set_xlabel("cluster-forming p-value (p_CF)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.set_title(title, color=INK_PRIMARY, fontsize=10, loc="left")
    ax.set_ylim(bottom=0)

    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def make_figure(summary, curve, results, dataset_label: str):
    fig, (ax_voxel, ax_edge) = plt.subplots(
        1, 2, figsize=(11, 4.5), facecolor=SURFACE,
    )

    draw_panel(
        ax_voxel, summary, curve, results,
        value_col="giant_voxel_fraction",
        pct_col="percentile_giant_voxel_fraction",
        mean_col="mean_giant_voxel_fraction",
        ylabel="largest bundle / mask voxels",
        title="A. Voxel-fraction order parameter (gating metric)",
        show_epsilon=True, show_thresholds=True, point_color=BLUE,
    )
    draw_panel(
        ax_edge, summary, curve, results,
        value_col="giant_edge_fraction",
        pct_col="percentile_giant_edge_fraction",
        mean_col="mean_giant_edge_fraction",
        ylabel="largest bundle / retained edges",
        title="B. Edge-fraction order parameter (diagnostic only)",
        show_epsilon=False, show_thresholds=True, point_color=ORANGE,
    )
    ax_edge.annotate(
        "denominator collapses as edges\nvanish — not used for gating",
        xy=(results["threshold_grid"][-1], curve.loc[
            curve["cluster_forming_p"] == results["threshold_grid"][-1],
            "giant_edge_fraction",
        ].mean()),
        xytext=(-70, -35), textcoords="offset points",
        fontsize=7.5, color=INK_MUTED,
        arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.8),
        ha="center",
    )

    handles, labels = ax_voxel.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False,
        fontsize=8, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.02),
    )

    n_perm = results["calibration_permutations"]
    fig.suptitle(
        f"Percolation calibration — {dataset_label} (N={n_perm} null permutations)",
        color=INK_PRIMARY, fontsize=12, y=1.03,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("result_dir", type=Path, help="percolation_calibration.py output directory")
    p.add_argument("output", type=Path, help="output image path (.png or .pdf)")
    p.add_argument("--dataset-label", required=True, help="human-readable dataset label for the figure title, e.g. 'Controls vs. patients'")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    summary, curve, results = load_run(args.result_dir)
    fig = make_figure(summary, curve, results, args.dataset_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.output}")
    print(
        f"recommended p_CF = {results['recommended_p_cf']:g} "
        f"(transition at {results['transition_p_cf']:g}, "
        f"{results['percentile']:.0f}th-percentile voxel-fraction epsilon = {results['epsilon']:g})"
    )


if __name__ == "__main__":
    main()
