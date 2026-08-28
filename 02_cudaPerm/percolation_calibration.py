#!/usr/bin/env python3
"""Calibrate a sub-percolation cluster-forming threshold from null permutations.

The historical strict (transitive/union-find) bundler undergoes a percolation
phase transition as the suprathreshold edge graph gets denser: past some
cluster-forming p-value, spatially local admissible merges chain across most
of the brain and one "bundle" swallows a large fraction of all surviving
edges. Any cluster-forming threshold on the super-critical side of that
transition produces a giant, anatomically meaningless component rather than a
localized bundle -- regardless of whether the resulting statistic happens to
be significant.

This script estimates that transition directly from an independent batch of
null label-permutations (never the observed grouping), so the calibrated
threshold is a property of the dataset's own null adjacency geometry and is
decided before any observed bundle statistic is inspected. It reuses the
existing v3 ("df-stored") sparse CUDA format: CUDA runs once at the most
liberal candidate threshold, and the C++ bundler cheaply re-thresholds the
same cached sparse edges at every stricter grid point.

Order parameter per (permutation, threshold): the fraction of all mask voxels
touched by the single largest strict bundle (summed across both signs,
matching how the production pipeline takes one joint two-sided maximum).
This voxel-coverage fraction is what actually flags a brain-spanning giant
component (the original symptom was "98.7% of voxels touched"), and its
denominator (total mask voxels) is fixed regardless of threshold. A companion
edge fraction (largest bundle edges / all retained edges) is also recorded
for diagnostics only: it is NOT used to find the transition, because its
denominator shrinks alongside its numerator at strict thresholds, and with
very few retained edges left, a handful of them land in one component by
chance -- producing a spurious *second* rise in edge-fraction at the sparse
end that has nothing to do with real percolation. A pilot run always showed
this reversal: edge-fraction fell as expected, then climbed back to 1.0 at
the strictest grid point, while voxel-fraction kept falling monotonically.

Calibration rule: the most liberal (largest) threshold in the grid at which a
chosen percentile of the null giant-voxel-fraction distribution stays at or
below --epsilon is the estimated transition point. The recommended operating
threshold is one grid step stricter than that point, as a safety margin
against the large fluctuations expected near a critical point.

This is a calibration tool, not the inference pipeline: it never touches row
0 (the observed grouping) and never computes an FWER p-value. Run it once per
dataset/neighbor_dist combination before choosing the grid for the real
run_bundle_fwer.py grid-FWER inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bundle_fwer import consecutive_batches, count_nonempty_lines


DEFAULT_THRESHOLD_GRID = (
    1e-3, 7e-4, 5e-4, 3e-4, 2e-4, 1e-4,
    7e-5, 5e-5, 3e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6,
)

GIANT_COMPONENT_COLUMNS = (
    "permutation", "observed", "retained_edges", "bundles",
    "largest_bundle_edges", "largest_bundle_voxels", "n_voxels",
)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filelist", type=Path, help="one subject ccmat path per line")
    parser.add_argument(
        "permutations", type=Path,
        help="row 0 is the observed grouping and is never used; rows 1.. are nulls",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--mask", type=Path, default=project / "templates/mask3mm.dump",
    )
    parser.add_argument(
        "--calibration-permutations", type=int, default=30,
        help="number of independent null rows to calibrate on (default: 30, a pilot size)",
    )
    parser.add_argument(
        "--first-null-row", type=int, default=1,
        help="first permutation-file row to use as a null (default: 1, i.e. skip row 0)",
    )
    parser.add_argument(
        "--threshold-grid", type=float, nargs="+", default=None,
        help="candidate cluster-forming p-values, most liberal first (default: a built-in log-spaced grid)",
    )
    parser.add_argument("--statistic", choices=("mass", "extent"), default="mass")
    parser.add_argument("--neighbor-dist", type=float, default=1.0)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--min-cluster-voxels", type=int, default=6)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=("rows per CUDA invocation; each invocation reloads the full "
              "subject connectivity data (tens to hundreds of GB), so the "
              "default is one batch covering all calibration permutations"),
    )
    parser.add_argument(
        "--capacity", type=int, default=20_000_000,
        help="maximum sparse edges per CUDA part (production runs needed 20M after a 10M overflow)",
    )
    parser.add_argument(
        "--backend", type=Path,
        default=Path(__file__).resolve().parent / "build/permutationTest_cuda_bundle",
    )
    parser.add_argument(
        "--cpp-backend", type=Path,
        default=Path(__file__).resolve().parent / "build/bundle_fwer_omp",
        help="strict-bundle C++ engine only; percolation calibration is meaningless for the bounded method",
    )
    parser.add_argument("--bundle-threads", type=int, default=16)
    parser.add_argument(
        "--epsilon", type=float, default=0.05,
        help="giant-component voxel-fraction ceiling that defines the transition (default: 0.05)",
    )
    parser.add_argument(
        "--percentile", type=float, default=95.0,
        help="percentile of the null giant-voxel-fraction distribution to calibrate against, "
             "not the mean, since the FWER null tail is what a percolating outlier would distort",
    )
    parser.add_argument("--keep-sparse", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    threshold_grid = sorted(
        set(args.threshold_grid) if args.threshold_grid else DEFAULT_THRESHOLD_GRID,
        reverse=True,
    )
    if len(threshold_grid) < 2:
        raise ValueError("--threshold-grid requires at least two distinct values.")
    if not (0.0 < args.epsilon < 1.0):
        raise ValueError("--epsilon must be between 0 and 1.")
    if not (0.0 < args.percentile < 100.0):
        raise ValueError("--percentile must be between 0 and 100.")
    if args.calibration_permutations < 1:
        raise ValueError("--calibration-permutations must be positive.")

    filelist = args.filelist.resolve()
    permutations = args.permutations.resolve()
    mask = args.mask.resolve()
    backend = args.backend.resolve()
    cpp_backend = args.cpp_backend.resolve()
    for path in (filelist, permutations, mask, backend, cpp_backend):
        if not path.exists():
            raise FileNotFoundError(path)

    total_rows = count_nonempty_lines(permutations)
    first_row = args.first_null_row
    last_row_exclusive = first_row + args.calibration_permutations
    if first_row < 1 or last_row_exclusive > total_rows:
        raise ValueError(
            f"Requested null rows [{first_row}, {last_row_exclusive}) but the "
            f"permutation file only has {total_rows} rows (row 0 is observed)."
        )
    null_rows = list(range(first_row, last_row_exclusive))

    coordinates = np.loadtxt(mask, usecols=(0, 1, 2), dtype=np.float64, ndmin=2)
    n_voxels = int(coordinates.shape[0])
    n_subjects = count_nonempty_lines(filelist)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = output_dir / "sparse_work"
    sparse_dir.mkdir(exist_ok=True)
    prefix = sparse_dir / "calib"

    config = {
        "filelist": str(filelist),
        "permutations": str(permutations),
        "mask": str(mask),
        "null_rows": [null_rows[0], null_rows[-1]],
        "calibration_permutations": len(null_rows),
        "subjects": n_subjects,
        "voxels": n_voxels,
        "threshold_grid": threshold_grid,
        "statistic": args.statistic,
        "neighbor_dist": args.neighbor_dist,
        "min_size": args.min_size,
        "min_cluster_voxels": args.min_cluster_voxels,
        "epsilon": args.epsilon,
        "percentile": args.percentile,
    }
    (output_dir / "percolation_calibration_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    most_liberal = threshold_grid[0]
    batch_size = args.batch_size if args.batch_size is not None else len(null_rows)
    cuda_seconds_total = 0.0
    for batch in consecutive_batches(null_rows, batch_size):
        print(f"[CUDA] rows {batch[0]}..{batch[-1]} at p={most_liberal:g}", flush=True)
        command = [
            str(backend), str(filelist), str(permutations), str(prefix), "0",
            "--start-perm", str(batch[0]), "--count", str(len(batch)),
            "--capacity", str(args.capacity),
            "--cluster-forming-p", str(most_liberal), "--store-df",
        ]
        started = perf_counter()
        subprocess.run(command, check=True)
        elapsed = perf_counter() - started
        cuda_seconds_total += elapsed
        print(f"[CUDA] batch done in {elapsed:.3f}s", flush=True)

    frames: list[pd.DataFrame] = []
    bundle_seconds_total = 0.0
    for threshold_index, threshold in enumerate(threshold_grid):
        report_path = output_dir / f"giant_component_{threshold_index:02d}.csv"
        is_last_threshold = threshold_index == len(threshold_grid) - 1
        command = [
            str(cpp_backend), str(mask), str(prefix), str(null_rows[0]),
            str(len(null_rows)), args.statistic, str(threshold),
            str(args.neighbor_dist), str(args.min_size),
            str(args.min_cluster_voxels),
            str(output_dir / f"maxima_{threshold_index:02d}.csv"),
            "--threads", str(args.bundle_threads),
            "--df-aware", "--records-contain-df", "--subjects", str(n_subjects),
            "--giant-component-report", str(report_path),
        ]
        if not args.keep_sparse and is_last_threshold:
            command.append("--delete-inputs")
        started = perf_counter()
        subprocess.run(command, check=True)
        elapsed = perf_counter() - started
        bundle_seconds_total += elapsed
        print(
            f"[bundle p={threshold:g}] rethresholded {len(null_rows)} nulls "
            f"in {elapsed:.3f}s",
            flush=True,
        )

        rows = pd.read_csv(report_path)
        if sorted(rows["permutation"].astype(int).tolist()) != null_rows:
            raise RuntimeError(
                f"Giant-component report at p={threshold:g} does not cover "
                "the requested calibration rows."
            )
        rows.insert(0, "cluster_forming_p", threshold)
        rows.insert(1, "threshold_index", threshold_index)
        frames.append(rows)

    curve = pd.concat(frames, ignore_index=True)
    curve["giant_edge_fraction"] = np.where(
        curve["retained_edges"] > 0,
        curve["largest_bundle_edges"] / curve["retained_edges"],
        0.0,
    )
    curve["giant_voxel_fraction"] = curve["largest_bundle_voxels"] / curve["n_voxels"]
    curve.to_csv(output_dir / "percolation_calibration_curve.csv", index=False)

    summary = (
        curve.groupby(["threshold_index", "cluster_forming_p"])
        .agg(
            n_permutations=("permutation", "count"),
            mean_retained_edges=("retained_edges", "mean"),
            mean_giant_edge_fraction=("giant_edge_fraction", "mean"),
            percentile_giant_edge_fraction=(
                "giant_edge_fraction",
                lambda values: float(np.percentile(values, args.percentile)),
            ),
            max_giant_edge_fraction=("giant_edge_fraction", "max"),
            mean_giant_voxel_fraction=("giant_voxel_fraction", "mean"),
            percentile_giant_voxel_fraction=(
                "giant_voxel_fraction",
                lambda values: float(np.percentile(values, args.percentile)),
            ),
        )
        .reset_index()
        .sort_values("cluster_forming_p", ascending=False)
    )
    summary.to_csv(output_dir / "percolation_calibration_summary.csv", index=False)

    sub_critical = summary[
        summary["percentile_giant_voxel_fraction"] <= args.epsilon
    ].sort_values("cluster_forming_p", ascending=False)

    print("\n=== Percolation calibration curve ===")
    print(summary.to_string(index=False))

    result = {
        **config,
        "cuda_seconds": cuda_seconds_total,
        "bundle_seconds": bundle_seconds_total,
    }
    if sub_critical.empty:
        result["transition_p_cf"] = None
        result["recommended_p_cf"] = None
        print(
            f"\nNo grid point kept the p{args.percentile:g} giant-voxel-fraction "
            f"at or below {args.epsilon:g}. Extend --threshold-grid further "
            "toward stricter (smaller) values."
        )
    else:
        transition_row = sub_critical.iloc[0]
        transition_p_cf = float(transition_row["cluster_forming_p"])
        transition_index = int(transition_row["threshold_index"])
        safety_index = min(transition_index + 1, len(threshold_grid) - 1)
        recommended_p_cf = float(threshold_grid[safety_index])
        result["transition_p_cf"] = transition_p_cf
        result["transition_threshold_index"] = transition_index
        result["recommended_p_cf"] = recommended_p_cf
        result["recommended_threshold_index"] = safety_index
        print(
            f"\nEstimated transition: p_CF={transition_p_cf:g} is the most "
            f"liberal grid point with p{args.percentile:g}(giant_voxel_fraction) "
            f"<= {args.epsilon:g}."
        )
        if safety_index == transition_index:
            print(
                "Warning: this is already the strictest grid point; extend "
                "--threshold-grid stricter to get a real safety margin."
            )
        print(
            f"Recommended operating threshold (one grid step stricter): "
            f"p_CF={recommended_p_cf:g}."
        )

    (output_dir / "percolation_calibration_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(f"\nComplete: {output_dir / 'percolation_calibration_results.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, FileNotFoundError,
            subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
