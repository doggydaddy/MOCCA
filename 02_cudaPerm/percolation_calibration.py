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

Calibration reads a *held-out* range of the master permutation file. Under the
default partition it uses rows 1..1000 only, while run_bundle_fwer.py uses row
0 plus rows 1001..11000; the two null subsets are disjoint, so a label
assignment that helped choose the threshold can never also contribute to the
FWER null distribution. See permutation_rows.py and
manuscript/ANALYSIS_DECISIONS.md (2026-09-02, "disjoint calibration and
inference permutations").

One thousand calibration rows give roughly 50 observations in the upper 5%
tail that a 95th-percentile rule depends on, against roughly 10 at 200 rows.
Selection stability is assessed by resampling and subdividing those
calibration rows only (--stability-replicates, --stability-subdivisions). If
the choice is unstable, increase --calibration-permutations prospectively or
adopt a stricter predeclared rule; never resolve it with inference rows.
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
from permutation_rows import (
    DEFAULT_CALIBRATION_PERMUTATIONS,
    add_partition_arguments,
    partition_from_args,
    validate_permutation_file,
)


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
    parser.add_argument(
        "--stability-replicates", type=int, default=1000,
        help="bootstrap resamples of the calibration rows used to assess how "
             "stable the threshold selection is (default: 1000; 0 disables)",
    )
    parser.add_argument(
        "--stability-subdivisions", type=int, default=4,
        help="disjoint subsets the calibration rows are split into, each of "
             "which re-selects a threshold independently (default: 4)",
    )
    parser.add_argument(
        "--stability-seed", type=int, default=20260902,
        help="seed for the stability bootstrap (recorded in the results)",
    )
    parser.add_argument("--keep-sparse", action="store_true")
    parser.add_argument(
        "--freedman-lane-plan", type=Path, default=None,
        help="freedman_lane_plan.flp from freedman_lane.py. Calibrates against "
             "the covariate-adjusted HC2 statistic instead of unadjusted "
             "Welch, and requires a full-index permutation file. The null "
             "adjacency geometry belongs to whichever statistic inference will "
             "actually use, so this must match the production run.",
    )
    add_partition_arguments(parser, stage="calibration")
    return parser.parse_args()


def select_threshold_index(
    fractions: pd.DataFrame, threshold_grid: list[float],
    epsilon: float, percentile: float,
) -> int | None:
    """Apply the calibration rule to one set of rows.

    ``fractions`` needs ``threshold_index`` and ``giant_voxel_fraction``
    columns.  Returns the index of the most liberal grid point whose
    ``percentile`` of the null giant-voxel-fraction stays at or below
    ``epsilon``, or ``None`` when no grid point qualifies.
    """
    qualifying = [
        index
        for index in range(len(threshold_grid))
        if float(
            np.percentile(
                fractions.loc[
                    fractions["threshold_index"] == index, "giant_voxel_fraction"
                ].to_numpy(float),
                percentile,
            )
        ) <= epsilon
    ]
    return min(qualifying) if qualifying else None


def assess_stability(
    curve: pd.DataFrame, threshold_grid: list[float], null_rows: list[int],
    epsilon: float, percentile: float, replicates: int, subdivisions: int,
    seed: int,
) -> dict[str, object]:
    """Resample and subdivide the calibration rows only.

    The decision log requires selection stability to be judged without ever
    touching an inference row: if the choice is unstable, the fix is a larger
    prospective calibration set or a stricter predeclared rule, never a peek
    at the held-out nulls.
    """
    report: dict[str, object] = {
        "epsilon": epsilon,
        "percentile": percentile,
        "seed": seed,
        "calibration_rows": len(null_rows),
    }

    if replicates > 0:
        generator = np.random.default_rng(seed)
        indexed = {
            index: curve.loc[curve["threshold_index"] == index]
            .set_index("permutation")["giant_voxel_fraction"]
            for index in range(len(threshold_grid))
        }
        counts: dict[int | None, int] = {}
        for _ in range(replicates):
            drawn = generator.choice(null_rows, size=len(null_rows), replace=True)
            resampled = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "threshold_index": index,
                            "giant_voxel_fraction": series.loc[drawn].to_numpy(float),
                        }
                    )
                    for index, series in indexed.items()
                ],
                ignore_index=True,
            )
            choice = select_threshold_index(
                resampled, threshold_grid, epsilon, percentile
            )
            counts[choice] = counts.get(choice, 0) + 1
        report["bootstrap_replicates"] = replicates
        report["bootstrap_selection_counts"] = {
            ("none" if key is None else f"{threshold_grid[key]:g}"): value
            for key, value in sorted(
                counts.items(), key=lambda item: (item[0] is None, item[0])
            )
        }
        modal = max(counts.items(), key=lambda item: item[1])
        report["bootstrap_modal_selection"] = (
            None if modal[0] is None else float(threshold_grid[modal[0]])
        )
        report["bootstrap_modal_fraction"] = modal[1] / replicates

    if subdivisions > 1 and len(null_rows) >= subdivisions:
        blocks = np.array_split(np.asarray(null_rows), subdivisions)
        selections = []
        for block in blocks:
            subset = curve.loc[curve["permutation"].isin(block.tolist())]
            choice = select_threshold_index(
                subset, threshold_grid, epsilon, percentile
            )
            selections.append(
                None if choice is None else float(threshold_grid[choice])
            )
        report["subdivision_count"] = subdivisions
        report["subdivision_rows_each"] = [int(len(block)) for block in blocks]
        report["subdivision_selections"] = selections
        report["subdivision_unanimous"] = len(set(map(str, selections))) == 1

    return report


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
    partition = partition_from_args(args)
    freedman_lane_plan = (
        args.freedman_lane_plan.resolve()
        if args.freedman_lane_plan is not None else None
    )
    if freedman_lane_plan is not None and not freedman_lane_plan.exists():
        raise FileNotFoundError(freedman_lane_plan)

    filelist = args.filelist.resolve()
    permutations = args.permutations.resolve()
    mask = args.mask.resolve()
    backend = args.backend.resolve()
    cpp_backend = args.cpp_backend.resolve()
    for path in (filelist, permutations, mask, backend, cpp_backend):
        if not path.exists():
            raise FileNotFoundError(path)

    # Rejects overlapping calibration/inference ranges, a non-observed row 0,
    # duplicate rows, and an incorrect total row count before any GPU work.
    partition_report = validate_permutation_file(
        permutations, partition,
        allow_extra_rows=args.allow_extra_permutation_rows,
        representation="full-index" if freedman_lane_plan else "group-a",
        n_subjects=count_nonempty_lines(filelist),
    )
    null_rows = partition.calibration_rows
    print(f"partition: {partition.describe()}", flush=True)
    print(
        f"calibrating on rows {null_rows[0]}..{null_rows[-1]} only; "
        f"inference rows {partition.inference_start}.."
        f"{partition.inference_stop - 1} are held out and never read here",
        flush=True,
    )

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
        **partition_report,
        "subjects": n_subjects,
        "voxels": n_voxels,
        "threshold_grid": threshold_grid,
        "statistic": args.statistic,
        "neighbor_dist": args.neighbor_dist,
        "min_size": args.min_size,
        "min_cluster_voxels": args.min_cluster_voxels,
        "epsilon": args.epsilon,
        "percentile": args.percentile,
        "edge_statistic": (
            "hc2_freedman_lane_adjusted" if freedman_lane_plan
            else "welch_unadjusted"
        ),
        "freedman_lane_plan": (
            str(freedman_lane_plan) if freedman_lane_plan else None
        ),
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
        if freedman_lane_plan is not None:
            command.extend(["--freedman-lane", str(freedman_lane_plan)])
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

    stability = assess_stability(
        curve, threshold_grid, null_rows, args.epsilon, args.percentile,
        args.stability_replicates, args.stability_subdivisions,
        args.stability_seed,
    )
    result["selection_stability"] = stability
    print("\n=== Selection stability (calibration rows only) ===")
    if "bootstrap_modal_fraction" in stability:
        modal = stability["bootstrap_modal_selection"]
        print(
            f"Bootstrap ({stability['bootstrap_replicates']} resamples): "
            f"modal transition p_CF={modal if modal is None else f'{modal:g}'} "
            f"selected in {stability['bootstrap_modal_fraction']:.1%} of replicates"
        )
        print(f"  selection counts: {stability['bootstrap_selection_counts']}")
    if "subdivision_selections" in stability:
        print(
            f"Subdivision into {stability['subdivision_count']} disjoint blocks "
            f"of {stability['subdivision_rows_each']} rows selected: "
            f"{stability['subdivision_selections']}"
        )
        if not stability["subdivision_unanimous"]:
            print(
                "  Warning: disjoint calibration blocks disagree. Increase "
                "--calibration-permutations prospectively or adopt a stricter "
                "predeclared rule BEFORE running inference. Do not resolve an "
                "unstable choice with inference rows."
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
