"""Null-only calibration of the TFCE exponents E and H.

The single-threshold pipeline calibrates `p_CF` from null permutations alone;
TFCE removes that threshold but introduces two shape parameters in its place.
This driver calibrates those the same way, and under the same discipline: only
the held-out calibration rows are read, never the observed row and never an
inference row, so the choice cannot be tuned on the result it will later judge.

Criterion
---------
FWER is decided by the extreme upper tail of the null maximum distribution, so
a good (E, H) is one whose null max-TFCE tail does not run away from its own
bulk -- when it does, a handful of percolating permutations set the bar and no
focal effect can clear it, which is precisely the failure the percolation
calibration documented for the mass statistic. The primary, gating metric is
therefore the scale-free tail ratio

    tail_ratio = q99 / q50

minimised over the candidate grid. `q999 / q50`, `max / q50` and the rank
correlation between the null maximum and the largest bundle's voxel footprint
are reported as diagnostics but do not gate, exactly as edge-fraction is
retained as a diagnostic in `percolation_calibration.py`.

Cost
----
CUDA runs once, at the sparse storage threshold, with `--store-df`; every
candidate (E, H) then re-integrates the cached v3 sparse edges on CPU. Note
that TFCE cost grows sharply as `--tfce-z-min` falls: the lowest height fixes
how many edges are retained, and every height re-bundles them. Check the
reported per-candidate bundling time against your permutation budget before
committing to a production run.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tfce as tfce_reference
from permutation_rows import (
    add_partition_arguments,
    partition_from_args,
    validate_permutation_file,
)
from run_bundle_fwer import consecutive_batches, count_nonempty_lines

DEFAULT_EXTENT_EXPONENTS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_HEIGHT_EXPONENTS = (1.0, 2.0, 3.0)


def tail_metrics(maxima: np.ndarray) -> dict[str, float]:
    """Scale-free descriptions of how far the upper tail runs above the bulk."""

    positive = maxima[maxima > 0]
    if positive.size == 0:
        return {
            "median": 0.0, "q95": 0.0, "q99": 0.0, "q999": 0.0, "maximum": 0.0,
            "tail_ratio": float("inf"), "tail_ratio_999": float("inf"),
            "max_ratio": float("inf"), "nonzero_fraction": 0.0,
        }
    median = float(np.quantile(positive, 0.50))
    quantiles = {
        "median": median,
        "q95": float(np.quantile(positive, 0.95)),
        "q99": float(np.quantile(positive, 0.99)),
        "q999": float(np.quantile(positive, 0.999)),
        "maximum": float(positive.max()),
        "nonzero_fraction": float(positive.size / maxima.size),
    }
    scale = median if median > 0 else float("nan")
    quantiles["tail_ratio"] = quantiles["q99"] / scale
    quantiles["tail_ratio_999"] = quantiles["q999"] / scale
    quantiles["max_ratio"] = quantiles["maximum"] / scale
    return quantiles


def parse_args() -> argparse.Namespace:
    project = HERE.parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filelist", type=Path)
    parser.add_argument("permutations", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mask", type=Path, default=project / "templates/mask3mm.dump")
    parser.add_argument(
        "--cluster-forming-p", type=float, required=True,
        help=("how liberally the sparse edges are stored by CUDA; must reach "
              "at least the two-sided p of the lowest integration height"),
    )
    parser.add_argument(
        "--extent-exponents", type=float, nargs="+",
        default=list(DEFAULT_EXTENT_EXPONENTS), metavar="E",
    )
    parser.add_argument(
        "--height-exponents", type=float, nargs="+",
        default=list(DEFAULT_HEIGHT_EXPONENTS), metavar="H",
    )
    parser.add_argument("--tfce-z-min", type=float, default=4.0)
    parser.add_argument("--tfce-z-max", type=float, default=7.0)
    parser.add_argument("--tfce-z-step", type=float, default=0.1)
    parser.add_argument("--neighbor-dist", type=float, default=1.0)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--min-cluster-voxels", type=int, default=6)
    parser.add_argument("--capacity", type=int, default=20_000_000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--bundle-threads", type=int, default=16)
    parser.add_argument(
        "--backend", type=Path,
        default=HERE / "build/permutationTest_cuda_bundle",
    )
    parser.add_argument("--cpp-backend", type=Path, default=HERE / "build/bundle_fwer_omp")
    parser.add_argument("--freedman-lane-plan", type=Path, default=None)
    parser.add_argument("--keep-sparse", action="store_true")
    add_partition_arguments(parser, stage="calibration")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    heights = tfce_reference.z_grid(args.tfce_z_min, args.tfce_z_max, args.tfce_z_step)
    floor_p = float(tfce_reference.two_sided_p_from_z(args.tfce_z_min))
    if args.cluster_forming_p < floor_p:
        raise ValueError(
            f"--cluster-forming-p {args.cluster_forming_p:g} is stricter than "
            f"the lowest integration height z={args.tfce_z_min:g} "
            f"(p={floor_p:.3g}); the integral would be silently truncated."
        )
    candidates = [
        (float(extent), float(height))
        for height in sorted(set(args.height_exponents))
        for extent in sorted(set(args.extent_exponents))
    ]
    if not candidates:
        raise ValueError("at least one (E, H) candidate is required")
    for extent, height in candidates:
        if extent < 0 or height < 0:
            raise ValueError("TFCE exponents must be non-negative")

    partition = partition_from_args(args)
    freedman_lane_plan = (
        args.freedman_lane_plan.resolve()
        if args.freedman_lane_plan is not None else None
    )
    filelist = args.filelist.resolve()
    permutations = args.permutations.resolve()
    mask = args.mask.resolve()
    backend = args.backend.resolve()
    cpp_backend = args.cpp_backend.resolve()
    required = [filelist, permutations, mask, backend, cpp_backend]
    if freedman_lane_plan is not None:
        required.append(freedman_lane_plan)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    n_subjects = count_nonempty_lines(filelist)
    partition_report = validate_permutation_file(
        permutations, partition,
        allow_extra_rows=args.allow_extra_permutation_rows,
        representation="full-index" if freedman_lane_plan else "group-a",
        n_subjects=n_subjects,
    )
    null_rows = partition.calibration_rows
    print(f"partition: {partition.describe()}", flush=True)
    print(
        f"calibrating (E, H) on rows {null_rows[0]}..{null_rows[-1]} only; "
        f"inference rows {partition.inference_start}.."
        f"{partition.inference_stop - 1} are held out and never read here",
        flush=True,
    )

    coordinates = np.loadtxt(mask, usecols=(0, 1, 2), dtype=np.float64, ndmin=2)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = output_dir / "sparse_work"
    sparse_dir.mkdir(exist_ok=True)
    prefix = sparse_dir / "tfcecal"

    config = {
        "filelist": str(filelist),
        "permutations": str(permutations),
        "mask": str(mask),
        "subjects": n_subjects,
        "voxels": int(coordinates.shape[0]),
        **partition_report,
        "cluster_forming_p": args.cluster_forming_p,
        "statistic": "tfce",
        "tfce_z_min": args.tfce_z_min,
        "tfce_z_max": args.tfce_z_max,
        "tfce_z_step": args.tfce_z_step,
        "tfce_heights": int(len(heights)),
        "extent_exponents": sorted(set(args.extent_exponents)),
        "height_exponents": sorted(set(args.height_exponents)),
        "neighbor_dist": args.neighbor_dist,
        "min_size": args.min_size,
        "min_cluster_voxels": args.min_cluster_voxels,
        "criterion": "minimise tail_ratio = q99 / median of the null max-TFCE",
        "edge_statistic": (
            "hc2_freedman_lane_adjusted" if freedman_lane_plan
            else "welch_unadjusted"
        ),
        "freedman_lane_plan": str(freedman_lane_plan) if freedman_lane_plan else None,
    }
    (output_dir / "tfce_calibration_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    batch_size = args.batch_size if args.batch_size is not None else len(null_rows)
    for batch in consecutive_batches(null_rows, batch_size):
        print(f"[CUDA] rows {batch[0]}..{batch[-1]} at p={args.cluster_forming_p:g}",
              flush=True)
        command = [
            str(backend), str(filelist), str(permutations), str(prefix), "0",
            "--start-perm", str(batch[0]), "--count", str(len(batch)),
            "--capacity", str(args.capacity),
            "--cluster-forming-p", str(args.cluster_forming_p), "--store-df",
        ]
        if freedman_lane_plan is not None:
            command.extend(["--freedman-lane", str(freedman_lane_plan)])
        started = perf_counter()
        subprocess.run(command, check=True)
        print(f"[CUDA] batch done in {perf_counter() - started:.3f}s", flush=True)

    rows: list[dict[str, object]] = []
    for index, (extent_exponent, height_exponent) in enumerate(candidates):
        maxima_path = output_dir / f"maxima_E{extent_exponent:g}_H{height_exponent:g}.csv"
        report_path = output_dir / f"giant_component_E{extent_exponent:g}_H{height_exponent:g}.csv"
        command = [
            str(cpp_backend), str(mask), str(prefix), str(null_rows[0]),
            str(len(null_rows)), "tfce", str(args.cluster_forming_p),
            str(args.neighbor_dist), str(args.min_size),
            str(args.min_cluster_voxels), str(maxima_path),
            "--threads", str(args.bundle_threads),
            "--df-aware", "--records-contain-df", "--subjects", str(n_subjects),
            "--giant-component-report", str(report_path),
            "--tfce-extent-exponent", str(extent_exponent),
            "--tfce-height-exponent", str(height_exponent),
            "--tfce-z-min", str(args.tfce_z_min),
            "--tfce-z-max", str(args.tfce_z_max),
            "--tfce-z-step", str(args.tfce_z_step),
        ]
        if not args.keep_sparse and index == len(candidates) - 1:
            command.append("--delete-inputs")
        started = perf_counter()
        subprocess.run(command, check=True)
        elapsed = perf_counter() - started

        maxima = pd.read_csv(maxima_path)
        if sorted(maxima["permutation"].astype(int).tolist()) != null_rows:
            raise RuntimeError(
                f"TFCE maxima at E={extent_exponent:g} H={height_exponent:g} "
                "do not cover the requested calibration rows."
            )
        values = maxima["max_statistic"].to_numpy(float)
        giant = pd.read_csv(report_path)
        giant = giant.set_index("permutation").loc[maxima["permutation"]]
        voxel_fraction = (
            giant["largest_bundle_voxels"].to_numpy(float)
            / float(giant["n_voxels"].iloc[0])
        )
        metrics = tail_metrics(values)
        # How much of the null maximum's ranking is explained by how far the
        # largest bundle spread: a high correlation means the tail is still
        # being set by percolation rather than by focal height.
        order_statistic = float(
            pd.Series(values).corr(pd.Series(voxel_fraction), method="spearman")
        ) if values.size > 2 else float("nan")
        rows.append({
            "extent_exponent": extent_exponent,
            "height_exponent": height_exponent,
            "bundle_seconds": elapsed,
            "giant_voxel_fraction_p95": float(np.quantile(voxel_fraction, 0.95)),
            "spearman_max_vs_giant_voxels": order_statistic,
            **metrics,
        })
        print(
            f"[tfce E={extent_exponent:g} H={height_exponent:g}] "
            f"{len(null_rows)} nulls in {elapsed:.1f}s | median={metrics['median']:.4g} "
            f"q99={metrics['q99']:.4g} tail_ratio={metrics['tail_ratio']:.2f} "
            f"rho(giant)={order_statistic:.3f}",
            flush=True,
        )

    curve = pd.DataFrame(rows)
    curve.to_csv(output_dir / "tfce_calibration_summary.csv", index=False)
    finite = curve.loc[np.isfinite(curve["tail_ratio"])]
    if finite.empty:
        raise RuntimeError(
            "No candidate produced a usable null maximum distribution; the "
            "height grid is probably above every edge in the data."
        )
    best = finite.loc[finite["tail_ratio"].idxmin()]
    default = curve.loc[
        (curve["extent_exponent"] == 0.5) & (curve["height_exponent"] == 2.0)
    ]
    results = {
        **config,
        "calibration_permutations": len(null_rows),
        "recommended_extent_exponent": float(best["extent_exponent"]),
        "recommended_height_exponent": float(best["height_exponent"]),
        "recommended_tail_ratio": float(best["tail_ratio"]),
        "smith_nichols_default_tail_ratio": (
            float(default["tail_ratio"].iloc[0]) if len(default) else None
        ),
        "candidates": rows,
    }
    (output_dir / "tfce_calibration_results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(
        f"\nrecommended: E={best['extent_exponent']:g} H={best['height_exponent']:g} "
        f"(tail_ratio {best['tail_ratio']:.2f})",
        flush=True,
    )
    if len(default):
        print(
            f"Smith & Nichols default E=0.5 H=2: tail_ratio "
            f"{default['tail_ratio'].iloc[0]:.2f}",
            flush=True,
        )
    print(f"Complete: {output_dir / 'tfce_calibration_results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
