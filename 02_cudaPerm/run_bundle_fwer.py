#!/usr/bin/env python3
"""Drive sparse CUDA permutations and calculate bundle-level FWER values.

Row 0 of the permutation file is treated as the observed grouping.  Rows 1..
are the random group-label permutations.  The CUDA backend writes only edges
above either a fixed cluster-forming |t| threshold or a df-aware two-sided
cluster-forming p threshold; this controller forms spatial bundles
for every row and builds the null distribution of the maximum bundle statistic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from time import perf_counter
from typing import Iterator

import numpy as np
import pandas as pd

from bundle_fwer import BundleStatisticResult, compute_bundle_statistics


MAGIC = 0x4C444E42
VERSION = 1  # backward-compatible fixed-threshold sparse format
DF_AWARE_VERSION = 2
DF_STORED_VERSION = 3
HEADER = struct.Struct("<IIQQQQfI")
RECORD_DTYPE = np.dtype([("edge_index", "<u8"), ("tstat", "<f4")])
DF_AWARE_RECORD_DTYPE = np.dtype(
    [("edge_index", "<u8"), ("tstat", "<f4"), ("excess", "<f4")]
)
DF_STORED_RECORD_DTYPE = np.dtype(
    [("edge_index", "<u8"), ("tstat", "<f4"), ("degrees_of_freedom", "<f4")]
)
EDGE_COLUMNS = (
    "i1", "j1", "k1", "i2", "j2", "k2", "pvalue", "tstat",
    "bundle", "network",
)
MAXIMA_COLUMNS = (
    "permutation", "observed", "threshold_edges", "retained_edges",
    "bundles", "max_statistic",
)
GRID_MAXIMA_COLUMNS = (
    "permutation", "observed", "cluster_forming_p", "threshold_index",
    "threshold_edges", "retained_edges", "bundles", "max_statistic",
)


def threshold_slug(value: float) -> str:
    """Stable, filesystem-safe label for a cluster-forming p threshold."""

    return f"p_{value:.10g}".replace(".", "p")


def read_sparse_edges(path: Path) -> tuple[dict[str, int | float], np.ndarray]:
    """Read and validate one sparse file emitted by the CUDA backend."""

    with path.open("rb") as stream:
        raw_header = stream.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise ValueError(f"Truncated sparse header: {path}")
        values = HEADER.unpack(raw_header)
        header = {
            "magic": values[0],
            "version": values[1],
            "permutation": values[2],
            "n_records": values[3],
            "n_voxels": values[4],
            "n_total_edges": values[5],
            "threshold": values[6],
            "flags": values[7],
        }
        if header["magic"] != MAGIC or header["version"] not in (
            VERSION, DF_AWARE_VERSION, DF_STORED_VERSION
        ):
            raise ValueError(f"Unsupported sparse format in {path}")
        if (header["version"] != VERSION) != bool(header["flags"] & 1):
            raise ValueError(f"Inconsistent sparse threshold mode in {path}")
        if (header["version"] == DF_STORED_VERSION) != bool(header["flags"] & 2):
            raise ValueError(f"Inconsistent sparse df-storage mode in {path}")
        record_dtype = {
            VERSION: RECORD_DTYPE,
            DF_AWARE_VERSION: DF_AWARE_RECORD_DTYPE,
            DF_STORED_VERSION: DF_STORED_RECORD_DTYPE,
        }[header["version"]]
        records = np.fromfile(stream, dtype=record_dtype)

    if records.size != header["n_records"]:
        raise ValueError(
            f"Sparse record count mismatch in {path}: header says "
            f"{header['n_records']}, file contains {records.size}."
        )
    expected_size = HEADER.size + records.size * record_dtype.itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(f"Unexpected trailing or partial data in {path}")
    if records.size:
        if int(records["edge_index"].max()) >= header["n_total_edges"]:
            raise ValueError(f"Out-of-range edge index in {path}")
        records = np.sort(records, order="edge_index")
        if np.any(np.diff(records["edge_index"]) == 0):
            raise ValueError(f"Duplicate edge index in {path}")
    return header, records


def condensed_indices(flat_indices: np.ndarray, n_voxels: int) -> tuple[np.ndarray, np.ndarray]:
    """Invert row-major upper-triangle (k=1) indices without an N² array."""

    flat = np.asarray(flat_indices, dtype=np.int64)
    n_edges = n_voxels * (n_voxels - 1) // 2
    if np.any(flat < 0) or np.any(flat >= n_edges):
        raise ValueError("Flat edge index is outside the upper triangle.")
    if flat.size == 0:
        return flat.copy(), flat.copy()

    a = 2 * n_voxels - 1
    rows = np.floor((a - np.sqrt(a * a - 8.0 * flat)) / 2.0).astype(np.int64)
    starts = rows * (2 * n_voxels - rows - 1) // 2
    while np.any(starts > flat):
        mask = starts > flat
        rows[mask] -= 1
        starts[mask] = rows[mask] * (2 * n_voxels - rows[mask] - 1) // 2
    next_starts = (rows + 1) * (2 * n_voxels - rows - 2) // 2
    while np.any(next_starts <= flat):
        mask = next_starts <= flat
        rows[mask] += 1
        starts[mask] = next_starts[mask]
        next_starts[mask] = (
            (rows[mask] + 1) * (2 * n_voxels - rows[mask] - 2) // 2
        )
    columns = rows + 1 + (flat - starts)
    return rows, columns


def records_to_edges(records: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Turn sparse flat indices and t values into COFFEE-DAC's 8 columns."""

    rows, columns = condensed_indices(records["edge_index"], len(coordinates))
    edges = np.empty((records.size, 8), dtype=np.float64)
    edges[:, :3] = coordinates[rows]
    edges[:, 3:6] = coordinates[columns]
    edges[:, 6] = np.nan  # no edgewise p-value is used for bundle inference
    edges[:, 7] = records["tstat"]
    return edges


def consecutive_batches(indices: list[int], batch_size: int) -> Iterator[list[int]]:
    """Yield consecutive ranges, because one backend call accepts one range."""

    position = 0
    while position < len(indices):
        batch = [indices[position]]
        position += 1
        while (
            position < len(indices)
            and len(batch) < batch_size
            and indices[position] == batch[-1] + 1
        ):
            batch.append(indices[position])
            position += 1
        yield batch


def result_row(permutation: int, result: BundleStatisticResult) -> dict[str, object]:
    return {
        "permutation": permutation,
        "observed": permutation == 0,
        "threshold_edges": result.threshold_edge_count,
        "retained_edges": result.retained_edge_count,
        "bundles": result.bundle_count,
        "max_statistic": result.max_statistic,
    }


def save_observed(output_dir: Path, result: BundleStatisticResult) -> None:
    pd.DataFrame(result.edges_bundled, columns=EDGE_COLUMNS).to_csv(
        output_dir / "observed_edges_bundled.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "bundle": item.bundle,
                "sign": item.sign,
                "edge_count": item.edge_count,
                "mass": item.mass,
                "statistic": item.statistic,
            }
            for item in result.bundles
        ],
        columns=("bundle", "sign", "edge_count", "mass", "statistic"),
    ).to_csv(output_dir / "observed_bundles_uncorrected.csv", index=False)


def count_nonempty_lines(path: Path) -> int:
    with path.open() as stream:
        return sum(bool(line.strip()) for line in stream)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filelist", type=Path, help="one subject ccmat path per line")
    parser.add_argument("permutations", type=Path, help="row 0 observed, rows 1..B null")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--mask", type=Path, default=project / "templates/mask3mm.dump",
        help="mask dump whose first three columns are voxel coordinates",
    )
    threshold_group = parser.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument(
        "--threshold", type=float, help="fixed cluster-forming |t| threshold"
    )
    threshold_group.add_argument(
        "--cluster-forming-p", type=float,
        help="df-aware two-sided uncorrected Welch p threshold",
    )
    threshold_group.add_argument(
        "--cluster-forming-p-grid", type=float, nargs="+",
        help=("df-aware two-sided Welch p thresholds; the threshold search "
              "is included in FWER correction using permutation min-p"),
    )
    parser.add_argument(
        "--null-permutations", type=int,
        help="number of null rows to use (default: every row after row 0)",
    )
    parser.add_argument("--statistic", choices=("mass", "extent"), default="mass")
    parser.add_argument("--neighbor-dist", type=float, default=1.0)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--min-cluster-voxels", type=int, default=6)
    parser.add_argument("--connected-components", action="store_true", help="use non-strict rather than strict bundles")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=10_000_000, help="maximum sparse edges per CUDA part")
    parser.add_argument(
        "--backend", type=Path,
        default=Path(__file__).resolve().parent / "build/permutationTest_cuda_bundle",
        help="sparse CUDA t-statistic backend",
    )
    parser.add_argument(
        "--bundle-engine", choices=("cpp", "python"), default="cpp",
        help="optimized C++ engine (default) or Python regression oracle",
    )
    parser.add_argument(
        "--cpp-backend", type=Path,
        help=("C++/OpenMP bundle backend; defaults to bundle_fwer_omp for "
              "strict or bundle_fwer_bounded_omp for bounded bundles"),
    )
    parser.add_argument(
        "--bundle-method", choices=("strict", "bounded"), default="strict",
        help=("strict is the active historical transitive implementation "
              "(default); bounded preserves the rejected 2026-08-27 "
              "fixed-radius experiment for reproducibility only"),
    )
    parser.add_argument("--bundle-threads", type=int, default=4)
    parser.add_argument("--keep-sparse", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue a run with the identical saved configuration")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grid_mode = args.cluster_forming_p_grid is not None
    threshold_grid = (
        sorted(set(args.cluster_forming_p_grid), reverse=True)
        if grid_mode else None
    )
    df_aware = args.cluster_forming_p is not None or grid_mode
    if df_aware:
        probabilities = threshold_grid if grid_mode else [args.cluster_forming_p]
        if any(not (0 < value < 1) or not math.isfinite(value)
               for value in probabilities):
            raise ValueError("Cluster-forming p values must be finite and between 0 and 1.")
        if grid_mode and len(threshold_grid) < 2:
            raise ValueError("--cluster-forming-p-grid requires at least two distinct values.")
        if args.bundle_engine == "python":
            raise ValueError("Df-aware thresholding currently requires --bundle-engine cpp.")
    elif args.threshold <= 0 or not math.isfinite(args.threshold):
        raise ValueError("--threshold must be finite and > 0.")
    if args.batch_size < 1 or args.capacity < 1 or args.bundle_threads < 1:
        raise ValueError("--batch-size, --capacity, and --bundle-threads must be positive.")
    if args.bundle_engine == "cpp" and args.connected_components:
        raise ValueError(
            "The optimized C++ backend implements strict bundles only; "
            "use --bundle-engine python with --connected-components."
        )
    if args.bundle_method == "bounded" and args.bundle_engine != "cpp":
        raise ValueError("Bounded bundles require --bundle-engine cpp.")

    filelist = args.filelist.resolve()
    permutations = args.permutations.resolve()
    mask = args.mask.resolve()
    backend = args.backend.resolve()
    cpp_backend = (
        args.cpp_backend
        if args.cpp_backend is not None
        else Path(__file__).resolve().parent / "build" / (
            "bundle_fwer_bounded_omp"
            if args.bundle_method == "bounded" else "bundle_fwer_omp"
        )
    ).resolve()
    required_paths = [filelist, permutations, mask, backend]
    if args.bundle_engine == "cpp":
        required_paths.append(cpp_backend)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    total_rows = count_nonempty_lines(permutations)
    if total_rows < 2:
        raise ValueError("Permutation file needs observed row 0 and at least one null row.")
    null_count = total_rows - 1 if args.null_permutations is None else args.null_permutations
    if null_count < 1 or null_count >= total_rows:
        raise ValueError(f"Requested {null_count} null permutations, but file provides {total_rows - 1}.")
    requested_rows = null_count + 1

    coordinates = np.loadtxt(mask, usecols=(0, 1, 2), dtype=np.float64, ndmin=2)
    n_voxels = int(coordinates.shape[0])
    n_subjects = count_nonempty_lines(filelist)
    config = {
        "filelist": str(filelist),
        "permutations": str(permutations),
        "mask": str(mask),
        "cuda_backend": str(backend),
        "bundle_engine": args.bundle_engine,
        "cpp_backend": str(cpp_backend) if args.bundle_engine == "cpp" else None,
        "bundle_threads": args.bundle_threads if args.bundle_engine == "cpp" else None,
        "subjects": n_subjects,
        "voxels": n_voxels,
        "null_permutations": null_count,
        "threshold_mode": "welch_df_aware_p" if df_aware else "fixed_t",
        "threshold": args.threshold,
        "cluster_forming_p": args.cluster_forming_p,
        "cluster_forming_p_grid": threshold_grid,
        "grid_correction": "symmetric_permutation_min_p" if grid_mode else None,
        "statistic": args.statistic,
        "neighbor_dist": args.neighbor_dist,
        "min_size": args.min_size,
        "min_cluster_voxels": args.min_cluster_voxels,
        "strict_bundles": not args.connected_components,
        "bundle_method": args.bundle_method,
        "bounded_endpoint_radius_voxels": (
            math.ceil(args.neighbor_dist) if args.bundle_method == "bounded"
            else None
        ),
        "split_signs": True,
        "two_sided": True,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "bundle_fwer_config.json"
    maxima_path = output_dir / (
        "permutation_bundle_maxima_grid.csv" if grid_mode
        else "permutation_bundle_maxima.csv"
    )
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if not args.resume:
            raise FileExistsError(f"Run already exists at {output_dir}; pass --resume to continue.")
        if previous != config:
            raise ValueError("Saved run configuration differs; use a new output directory.")
    else:
        if maxima_path.exists():
            raise FileExistsError(f"Untracked maxima file already exists: {maxima_path}")
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    completed: set[int] = set()
    if maxima_path.exists():
        previous_maxima = pd.read_csv(maxima_path)
        if grid_mode:
            expected = set(threshold_grid)
            completed = {
                int(permutation)
                for permutation, rows in previous_maxima.groupby("permutation")
                if len(rows) == len(threshold_grid)
                and set(rows["cluster_forming_p"].astype(float)) == expected
            }
        else:
            completed = set(previous_maxima["permutation"].astype(int))
        if any(index < 0 or index >= requested_rows for index in completed):
            raise ValueError("Saved maxima contain a permutation outside this run.")

    missing = [index for index in range(requested_rows) if index not in completed]
    sparse_dir = output_dir / "sparse_work"
    sparse_dir.mkdir(exist_ok=True)
    prefix = sparse_dir / "bundle"
    write_header = not maxima_path.exists()

    for batch in consecutive_batches(missing, args.batch_size):
        print(f"Running CUDA rows {batch[0]}..{batch[-1]} of {requested_rows - 1}", flush=True)
        cluster_forming_value = (
            threshold_grid[0] if grid_mode
            else (args.cluster_forming_p if df_aware else args.threshold)
        )
        command = [
            str(backend), str(filelist), str(permutations), str(prefix),
            "0" if df_aware else str(args.threshold),
            "--start-perm", str(batch[0]),
            "--count", str(len(batch)), "--capacity", str(args.capacity),
        ]
        if df_aware:
            command.extend(["--cluster-forming-p", str(cluster_forming_value)])
        if grid_mode:
            command.append("--store-df")
        cuda_started = perf_counter()
        subprocess.run(command, check=True)
        cuda_seconds = perf_counter() - cuda_started
        print(
            f"CUDA batch completed in {cuda_seconds:.3f} seconds",
            flush=True,
        )

        rows: list[dict[str, object]]
        bundle_started = perf_counter()
        if args.bundle_engine == "cpp":
            rows = []
            active_thresholds = threshold_grid if grid_mode else [cluster_forming_value]
            for threshold_index, active_threshold in enumerate(active_thresholds):
                cpp_maxima = sparse_dir / (
                    f"cpp_maxima_{batch[0]:06d}_{batch[-1]:06d}"
                    f"_{threshold_index:02d}.csv"
                )
                cpp_command = [
                    str(cpp_backend), str(mask), str(prefix), str(batch[0]),
                    str(len(batch)), args.statistic, str(active_threshold),
                    str(args.neighbor_dist),
                    str(args.min_size), str(args.min_cluster_voxels),
                    str(cpp_maxima), "--threads", str(args.bundle_threads),
                ]
                if batch[0] == 0:
                    observed_dir = (
                        output_dir / "thresholds" / threshold_slug(active_threshold)
                        if grid_mode else output_dir
                    )
                    observed_dir.mkdir(parents=True, exist_ok=True)
                    cpp_command.extend(
                        [
                            "--observed-edges",
                            str(observed_dir / "observed_edges_bundled.csv"),
                            "--observed-bundles",
                            str(observed_dir / "observed_bundles_uncorrected.csv"),
                        ]
                    )
                if df_aware:
                    cpp_command.append("--df-aware")
                if grid_mode:
                    cpp_command.extend(
                        ["--records-contain-df", "--subjects", str(n_subjects)]
                    )
                if args.bundle_method == "bounded":
                    cpp_command.append("--bounded-bundles")
                if (not args.keep_sparse
                        and threshold_index == len(active_thresholds) - 1):
                    cpp_command.append("--delete-inputs")
                subprocess.run(cpp_command, check=True)
                cpp_rows = pd.read_csv(cpp_maxima)
                cpp_maxima.unlink()
                if cpp_rows["permutation"].astype(int).tolist() != batch:
                    raise ValueError("C++ bundle result rows do not match requested batch.")
                if grid_mode:
                    cpp_rows.insert(2, "cluster_forming_p", active_threshold)
                    cpp_rows.insert(3, "threshold_index", threshold_index)
                rows.extend(cpp_rows.to_dict(orient="records"))
        else:
            rows = []
            for permutation_index in batch:
                sparse_path = Path(f"{prefix}_perm{permutation_index:06d}.bsp")
                header, records = read_sparse_edges(sparse_path)
                if header["permutation"] != permutation_index:
                    raise ValueError(f"Wrong permutation index in {sparse_path}")
                if header["n_voxels"] != n_voxels:
                    raise ValueError(
                        f"Mask has {n_voxels} voxels but backend data has "
                        f"{header['n_voxels']}."
                    )
                if header["version"] != VERSION:
                    raise ValueError("Python bundle engine only supports fixed thresholds.")
                if not math.isclose(float(header["threshold"]), args.threshold, rel_tol=1e-6):
                    raise ValueError(f"Threshold mismatch in {sparse_path}")

                edges = records_to_edges(records, coordinates)
                result = compute_bundle_statistics(
                    edges,
                    cluster_forming_threshold=args.threshold,
                    statistic=args.statistic,
                    neighbor_dist=args.neighbor_dist,
                    min_size=args.min_size,
                    min_cluster_voxels=args.min_cluster_voxels,
                    strict_bundles=not args.connected_components,
                    split_signs=True,
                )
                rows.append(result_row(permutation_index, result))
                if permutation_index == 0:
                    save_observed(output_dir, result)
                if not args.keep_sparse:
                    sparse_path.unlink()

        bundle_seconds = perf_counter() - bundle_started
        print(
            f"{args.bundle_engine} bundle batch completed in "
            f"{bundle_seconds:.3f} seconds",
            flush=True,
        )

        with maxima_path.open("a", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=GRID_MAXIMA_COLUMNS if grid_mode else MAXIMA_COLUMNS,
            )
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerows(rows)
        print(f"Saved bundle maxima through row {batch[-1]}", flush=True)

    maxima = pd.read_csv(maxima_path).sort_values("permutation")
    expected_maxima_rows = requested_rows * (len(threshold_grid) if grid_mode else 1)
    if (len(maxima) != expected_maxima_rows
            or maxima["permutation"].nunique() != requested_rows):
        raise RuntimeError("The maximum-statistic distribution is incomplete.")
    if grid_mode:
        maxima = maxima.sort_values(["permutation", "threshold_index"])
        n_rows = requested_rows
        maxima["threshold_rank_p"] = maxima.groupby(
            "threshold_index", group_keys=False
        )["max_statistic"].rank(method="max", ascending=False) / n_rows
        maxima["min_rank_p"] = maxima.groupby("permutation")[
            "threshold_rank_p"
        ].transform("min")
    maxima.to_csv(maxima_path, index=False)

    if grid_mode:
        min_rank_by_permutation = maxima.groupby("permutation")[
            "min_rank_p"
        ].first().sort_index()
        np.save(
            output_dir / "null_min_threshold_rank_p.npy",
            min_rank_by_permutation.loc[min_rank_by_permutation.index > 0]
            .to_numpy(float),
        )
        all_min_rank = min_rank_by_permutation.to_numpy(float)
        combined_observed = []
        for threshold_index, active_threshold in enumerate(threshold_grid):
            observed_dir = output_dir / "thresholds" / threshold_slug(active_threshold)
            observed_path = observed_dir / "observed_bundles_uncorrected.csv"
            observed = pd.read_csv(observed_path)
            threshold_maxima = maxima.loc[
                maxima["threshold_index"] == threshold_index, "max_statistic"
            ].to_numpy(float)
            if len(observed):
                single_p = np.array([
                    np.count_nonzero(threshold_maxima >= value) / requested_rows
                    for value in observed["statistic"].to_numpy(float)
                ])
                observed["p_threshold_fwer"] = single_p
                observed["p_grid_fwer"] = [
                    np.count_nonzero(all_min_rank <= value) / requested_rows
                    for value in single_p
                ]
            else:
                observed["p_threshold_fwer"] = pd.Series(dtype=float)
                observed["p_grid_fwer"] = pd.Series(dtype=float)
            observed.insert(0, "cluster_forming_p", active_threshold)
            observed.insert(1, "threshold_index", threshold_index)
            observed.to_csv(observed_dir / "observed_bundles_grid_fwer.csv", index=False)
            combined_observed.append(observed)
        observed_all = pd.concat(combined_observed, ignore_index=True)
        observed_all.to_csv(output_dir / "observed_bundles_grid_fwer.csv", index=False)
        summary = {
            **config,
            "minimum_attainable_p_fwer": 1 / requested_rows,
            "observed_bundles_across_thresholds": int(len(observed_all)),
            "complete": True,
        }
        (output_dir / "bundle_fwer_results.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(
            f"Complete: {output_dir / 'observed_bundles_grid_fwer.csv'}",
            flush=True,
        )
        return 0

    observed_stat = float(maxima.loc[maxima["permutation"] == 0, "max_statistic"].iloc[0])
    null_maxima = maxima.loc[maxima["permutation"] > 0, "max_statistic"].to_numpy(float)
    np.save(output_dir / "null_max_bundle_statistics.npy", null_maxima)

    observed_path = output_dir / "observed_bundles_uncorrected.csv"
    observed = pd.read_csv(observed_path)
    if len(observed):
        observed["p_fwer"] = [
            (1 + int(np.count_nonzero(null_maxima >= value))) / (null_count + 1)
            for value in observed["statistic"].to_numpy(float)
        ]
    else:
        observed["p_fwer"] = pd.Series(dtype=float)
    observed.to_csv(output_dir / "observed_bundles_fwer.csv", index=False)

    summary = {
        **config,
        "observed_max_statistic": observed_stat,
        "minimum_attainable_p_fwer": 1 / (null_count + 1),
        "observed_bundles": int(len(observed)),
        "complete": True,
    }
    (output_dir / "bundle_fwer_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Complete: {output_dir / 'observed_bundles_fwer.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
