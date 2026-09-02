#!/usr/bin/env python3
"""Aggregate run-level CCMAT correlation matrices into one matrix per participant.

This is the optional ``01p5_FisherCC`` stage that sits between ``01_cudaCC``
(run-level Pearson ``r`` matrices) and ``02_cudaPerm`` (participant-level
permutation inference):

    01_cudaCC run-level Pearson r
        -> 01p5_FisherCC run-level atanh(r)
        -> equal-run mean Fisher z matrix per participant
        -> 02_cudaPerm participant-level permutation inference

The inferential unit downstream is the participant, so this stage emits exactly
one aggregate matrix per participant.  It deliberately offers no mode in which
repeated run-level matrices are handed to the group test as independent
observations.

Aggregation modes
-----------------
``fisher-equal``
    Transform each run matrix with ``atanh`` and take an equal-weight mean
    within participant.  Planned primary mode.
``fisher-duration``
    Transform each run matrix with ``atanh`` and take a weighted mean using a
    prospectively defined weight table.  Sensitivity mode only.
``raw-equal``
    Equal-weight arithmetic mean of the raw correlations, reproducing the
    behaviour of ``02_cudaPerm/average_ccmat_runs.py`` for validation and
    direct sensitivity comparison.

Outputs stay on the aggregation scale.  Fisher ``z`` participant matrices are
*not* back-transformed before the group test; apply ``tanh`` only to estimates
that are presented on the correlation scale for interpretation.

The binary CCMAT container is reused unchanged because the downstream C reader
(``02_cudaPerm/ccmat_io.c``) accepts only ``CCMAT_MAGIC``.  The Fisher ``z``
scale is therefore declared in the output filename, a per-file JSON sidecar,
and the run manifest instead of in the 24-byte header.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ── Binary CCMAT container (see 02_cudaPerm/ccmat_io.h) ──────────────────────
CCMAT_MAGIC = 0x43434D54
CCMAT_VERSION = 1
HEADER_SIZE = 24
HEADER_STRUCT = struct.Struct("<IIQQ")

SUBJECT_RE = re.compile(r"^s(\d+)_")
RUN_RE = re.compile(r"^s\d+_(\d+)")

MODULE_VERSION = "1.0.0"

# ── Aggregation modes ───────────────────────────────────────────────────────
MODES = ("fisher-equal", "fisher-duration", "raw-equal")

MODE_PROPERTIES = {
    "fisher-equal": {
        "transform": "atanh",
        "output_scale": "fisher_z",
        "weighting": "equal",
        "suffix": "fisherz",
        "role": "planned primary confirmatory mode",
    },
    "fisher-duration": {
        "transform": "atanh",
        "output_scale": "fisher_z",
        "weighting": "prospective weight table",
        "suffix": "fisherz_w",
        "role": "sensitivity mode only",
    },
    "raw-equal": {
        "transform": "identity",
        "output_scale": "pearson_r",
        "weighting": "equal",
        "suffix": "rawmean",
        "role": "validation / sensitivity reproduction of the current pipeline",
    },
}

# ── Clipping policy ─────────────────────────────────────────────────────────
# atanh is undefined at exactly +-1.  Only values with |r| >= 1 are clipped,
# and they are clipped to the float64 neighbour of +-1 rather than to an
# arbitrary ceiling.  Every other value passes through untouched.
CLAMP_R = float(np.nextafter(np.float64(1.0), np.float64(0.0)))
MAX_ABS_Z = float(np.arctanh(CLAMP_R))
CLIP_POLICY = (
    "clip |r| >= 1 to +-nextafter(1, 0) in float64 before atanh; "
    "no other value is modified"
)


# ── Container I/O ───────────────────────────────────────────────────────────
def read_header(path: Path) -> tuple[int, int]:
    """Return ``(n_voxels, n_elements)`` for a binary CCMAT file."""
    with path.open("rb") as stream:
        raw = stream.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise ValueError(f"short CCMAT header: {path}")
    magic, version, n_voxels, n_elements = HEADER_STRUCT.unpack(raw)
    if magic != CCMAT_MAGIC or version != CCMAT_VERSION:
        raise ValueError(f"unsupported CCMAT header: {path}")
    expected = HEADER_SIZE + 4 * n_elements
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"CCMAT size mismatch for {path}: {actual} != {expected}")
    return int(n_voxels), int(n_elements)


def valid_container(path: Path, n_voxels: int, n_elements: int) -> bool:
    try:
        return read_header(path) == (n_voxels, n_elements)
    except (OSError, ValueError):
        return False


# ── Participant / run bookkeeping ───────────────────────────────────────────
def subject_id(path: Path) -> str:
    match = SUBJECT_RE.match(path.name)
    if not match:
        raise ValueError(f"cannot extract participant ID from {path.name}")
    return match.group(1)


def run_sort_key(path: Path) -> tuple[int, str]:
    """Deterministic within-participant run order.

    Runs are processed in this canonical order regardless of their order in the
    input file list, so the aggregate is bitwise invariant to file-list
    ordering rather than merely close to invariant.
    """
    match = RUN_RE.match(path.name)
    index = int(match.group(1)) if match else -1
    return (index, path.name)


def collect_group(paths: list[Path]) -> OrderedDict[str, list[Path]]:
    """Group run paths by participant, preserving first-appearance order."""
    subjects: OrderedDict[str, list[Path]] = OrderedDict()
    for path in paths:
        subjects.setdefault(subject_id(path), []).append(path)
    for identifier, run_paths in subjects.items():
        ordered = sorted(run_paths, key=run_sort_key)
        duplicates = [
            str(candidate)
            for candidate in ordered
            if ordered.count(candidate) > 1
        ]
        if duplicates:
            raise ValueError(
                f"participant s{identifier} lists a run more than once: "
                f"{sorted(set(duplicates))[0]}"
            )
        subjects[identifier] = ordered
    return subjects


# ── Weight table ────────────────────────────────────────────────────────────
def load_weight_table(
    path: Path, weight_column: str, timepoint_column: str | None
) -> dict[str, dict[str, float | None]]:
    """Read a prospectively defined run weight table.

    The table is keyed on the run matrix filename (with or without the
    ``.ccmat`` suffix).  Weights are used exactly as supplied and are never
    derived from the data by this program.
    """
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"weight table is empty: {path}")

    fieldnames = rows[0].keys()
    for required in ("run", weight_column):
        if required not in fieldnames:
            raise ValueError(
                f"weight table {path} has no '{required}' column "
                f"(found: {sorted(fieldnames)})"
            )
    if timepoint_column is not None and timepoint_column not in fieldnames:
        raise ValueError(
            f"weight table {path} has no '{timepoint_column}' column "
            f"(found: {sorted(fieldnames)})"
        )

    table: dict[str, dict[str, float | None]] = {}
    for row in rows:
        key = Path(str(row["run"]).strip()).name
        key = key[:-6] if key.endswith(".ccmat") else key
        if not key:
            raise ValueError(f"weight table {path} has an empty 'run' entry")
        if key in table:
            raise ValueError(f"weight table {path} lists run '{key}' twice")

        weight = float(row[weight_column])
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"weight table {path}: run '{key}' has non-positive or "
                f"non-finite weight {weight!r}"
            )

        timepoints: float | None = None
        if timepoint_column is not None:
            raw = str(row[timepoint_column]).strip()
            timepoints = float(raw) if raw else None

        table[key] = {"weight": weight, "n_timepoints": timepoints}
    return table


def run_weights(
    run_paths: list[Path], table: dict[str, dict[str, float | None]] | None
) -> tuple[list[float], list[float | None]]:
    if table is None:
        return [1.0] * len(run_paths), [None] * len(run_paths)
    weights: list[float] = []
    timepoints: list[float | None] = []
    for path in run_paths:
        entry = table.get(path.stem)
        if entry is None:
            raise KeyError(f"weight table has no entry for run '{path.stem}'")
        weights.append(float(entry["weight"]))
        timepoints.append(entry["n_timepoints"])
    return weights, timepoints


# ── Core aggregation ────────────────────────────────────────────────────────
def aggregate_subject(
    run_paths: list[Path],
    weights: list[float],
    transform: str,
    output_path: Path,
    n_voxels: int,
    n_elements: int,
    chunk_elements: int,
) -> dict[str, object]:
    """Stream a weighted mean over run matrices into one output container.

    Accumulation is in float64 and strictly element-wise, so the result is
    bitwise identical for any ``chunk_elements``.  Runs are consumed in the
    order given, which ``collect_group`` has already canonicalised.
    """
    if len(run_paths) != len(weights):
        raise ValueError("run/weight length mismatch")
    total_weight = float(np.sum(np.asarray(weights, dtype=np.float64)))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(f"non-positive total weight for {output_path}")

    run_arrays = [
        np.memmap(path, dtype="<f4", mode="r", offset=HEADER_SIZE, shape=(n_elements,))
        for path in run_paths
    ]

    clipped_at_one = 0
    clipped_beyond_one = 0
    clipped_min = np.inf
    clipped_max = -np.inf
    output_min = np.inf
    output_max = -np.inf

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(
                HEADER_STRUCT.pack(CCMAT_MAGIC, CCMAT_VERSION, n_voxels, n_elements)
            )
            for start in range(0, n_elements, chunk_elements):
                stop = min(start + chunk_elements, n_elements)
                accumulator = np.zeros(stop - start, dtype=np.float64)

                for path, values, weight in zip(run_paths, run_arrays, weights):
                    block = np.asarray(values[start:stop], dtype=np.float64)

                    nonfinite = ~np.isfinite(block)
                    if nonfinite.any():
                        offset = start + int(np.flatnonzero(nonfinite)[0])
                        raise ValueError(
                            f"non-finite correlation in {path} at element "
                            f"{offset}: {block[nonfinite][0]!r} "
                            f"({int(nonfinite.sum())} in this chunk). Refusing "
                            "to coerce; fix the run matrix upstream."
                        )

                    if transform == "atanh":
                        outside = np.abs(block) >= 1.0
                        if outside.any():
                            offenders = block[outside]
                            clipped_min = min(clipped_min, float(offenders.min()))
                            clipped_max = max(clipped_max, float(offenders.max()))
                            clipped_at_one += int(
                                np.count_nonzero(np.abs(offenders) == 1.0)
                            )
                            clipped_beyond_one += int(
                                np.count_nonzero(np.abs(offenders) > 1.0)
                            )
                            np.clip(block, -CLAMP_R, CLAMP_R, out=block)
                        block = np.arctanh(block)

                    accumulator += weight * block

                accumulator /= total_weight
                if accumulator.size:
                    output_min = min(output_min, float(accumulator.min()))
                    output_max = max(output_max, float(accumulator.max()))
                accumulator.astype("<f4").tofile(stream)

            stream.flush()
            os.fsync(stream.fileno())

        if not valid_container(temporary, n_voxels, n_elements):
            raise RuntimeError(f"failed output validation: {temporary}")
        temporary.replace(output_path)
    finally:
        run_arrays.clear()
        if temporary.exists():
            temporary.unlink()

    total_clipped = clipped_at_one + clipped_beyond_one
    return {
        "clipped_total": total_clipped,
        "clipped_at_unit": clipped_at_one,
        "clipped_beyond_unit": clipped_beyond_one,
        "clipped_min_input": float(clipped_min) if total_clipped else None,
        "clipped_max_input": float(clipped_max) if total_clipped else None,
        "output_min": float(output_min) if n_elements else None,
        "output_max": float(output_max) if n_elements else None,
        "total_weight": total_weight,
    }


# ── Provenance ──────────────────────────────────────────────────────────────
def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path, policy: str) -> dict[str, object]:
    """Record a checksum under the requested policy.

    Full ``sha256`` over run matrices of this size is hours of I/O, so the
    default is the cheap ``size-mtime`` fingerprint.  Whichever policy was used
    is recorded in the manifest so a reader is never left guessing how strong
    the recorded identity is.
    """
    stat = path.stat()
    record: dict[str, object] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "checksum_policy": policy,
    }
    if policy == "sha256":
        record["sha256"] = sha256_file(path)
    return record


def software_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "module": "01p5_FisherCC/fisher_aggregate_ccmat.py",
        "module_version": MODULE_VERSION,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }
    repo_root = Path(__file__).resolve().parent.parent
    try:
        metadata["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        metadata["git_dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        metadata["git_commit"] = None
        metadata["git_dirty"] = None
    return metadata


def check_free_space(
    output_dir: Path, outputs: list[Path], n_elements: int, allow_low: bool
) -> dict[str, int]:
    """Refuse to start a multi-hour job that cannot fit its own output.

    At production size each participant matrix is several GB, so a partially
    written set that dies on ENOSPC costs an hour of I/O to discover. Outputs
    already present at full size are not counted, so resuming an interrupted
    run is not blocked by space the run has already claimed.
    """
    each = HEADER_SIZE + 4 * n_elements
    pending = [
        path
        for path in outputs
        if not (path.is_file() and path.stat().st_size == each)
    ]
    required = len(pending) * each
    probe = output_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    available = shutil.disk_usage(probe).free
    record = {
        "pending_outputs": len(pending),
        "required_bytes": required,
        "available_bytes": int(available),
    }
    if required > available and not allow_low:
        raise RuntimeError(
            f"output needs {required / 2**30:.1f} GiB but only "
            f"{available / 2**30:.1f} GiB is free at {probe}; free space or "
            "pass --allow-low-space to proceed anyway"
        )
    return record


def sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".json")


def sidecar_is_current(
    output_path: Path, expected: dict[str, object], n_voxels: int, n_elements: int
) -> bool:
    """True when a previous run already produced exactly this output."""
    if not valid_container(output_path, n_voxels, n_elements):
        return False
    path = sidecar_path(output_path)
    if not path.is_file():
        return False
    try:
        recorded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    keys = ("mode", "transform", "output_scale", "runs", "weights", "n_elements")
    return all(recorded.get(key) == expected.get(key) for key in keys)


# ── Orchestration ───────────────────────────────────────────────────────────
def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file-list",
        required=True,
        type=Path,
        help="Group-ordered list of run-level .ccmat paths, one per line.",
    )
    parser.add_argument(
        "--group-a-runs",
        required=True,
        type=int,
        help="Number of leading file-list entries belonging to group A.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--output-file-list",
        required=True,
        type=Path,
        help="Participant-ordered file list written for 02_cudaPerm.",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="fisher-equal",
        help="Aggregation rule (default: fisher-equal, the planned primary mode).",
    )
    parser.add_argument(
        "--weight-table",
        type=Path,
        help="CSV with a 'run' column and a weight column; required by "
        "--mode fisher-duration and rejected by the equal-weight modes.",
    )
    parser.add_argument(
        "--weight-column",
        default="weight",
        help="Weight column name in --weight-table (default: weight).",
    )
    parser.add_argument(
        "--timepoint-column",
        default=None,
        help="Optional column in --weight-table recording each run's time-point "
        "count, carried into the manifest as provenance only. Frame counts are "
        "never converted into weights by this program.",
    )
    parser.add_argument(
        "--checksum",
        choices=("size-mtime", "sha256"),
        default="size-mtime",
        help="Input/output fingerprint policy for the manifest. 'sha256' reads "
        "every matrix in full and is slow at production sizes "
        "(default: size-mtime).",
    )
    parser.add_argument(
        "--chunk-elements",
        type=int,
        default=8_000_000,
        help="Float elements processed per chunk (default: 8 million). Results "
        "are bitwise invariant to this value.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute participants whose output and sidecar are already current.",
    )
    parser.add_argument(
        "--allow-low-space",
        action="store_true",
        help="Proceed even when the output volume has less free space than the "
        "participant matrices will need.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs, participant grouping, weights and free space, "
        "then stop.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    properties = MODE_PROPERTIES[args.mode]

    if args.chunk_elements <= 0:
        raise ValueError("--chunk-elements must be positive")

    # ── input file list ─────────────────────────────────────────────────────
    paths = [
        Path(line.strip())
        for line in args.file_list.read_text().splitlines()
        if line.strip()
    ]
    if not 0 < args.group_a_runs < len(paths):
        raise ValueError("--group-a-runs must split the input file list")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} run matrices; first: {missing[0]}"
        )

    group_a = collect_group(paths[: args.group_a_runs])
    group_b = collect_group(paths[args.group_a_runs :])
    overlap = set(group_a).intersection(group_b)
    if overlap:
        raise ValueError(f"participants cross the group boundary: {sorted(overlap)}")

    n_voxels, n_elements = read_header(paths[0])
    for path in paths[1:]:
        if read_header(path) != (n_voxels, n_elements):
            raise ValueError(f"matrix dimensions differ: {path}")

    # ── weights ─────────────────────────────────────────────────────────────
    if args.mode == "fisher-duration":
        if args.weight_table is None:
            raise ValueError("--mode fisher-duration requires --weight-table")
        table = load_weight_table(
            args.weight_table, args.weight_column, args.timepoint_column
        )
    else:
        if args.weight_table is not None:
            raise ValueError(
                f"--mode {args.mode} is an equal-weight mode; "
                "--weight-table is not accepted"
            )
        table = None

    participants = [("A", key, value) for key, value in group_a.items()]
    participants += [("B", key, value) for key, value in group_b.items()]

    # Resolve every weight before writing anything, so a missing table entry
    # fails before hours of I/O rather than midway through.
    resolved = []
    for group, identifier, run_paths in participants:
        weights, timepoints = run_weights(run_paths, table)
        resolved.append((group, identifier, run_paths, weights, timepoints))

    print(
        f"mode={args.mode} ({properties['role']}); "
        f"transform={properties['transform']}; "
        f"output scale={properties['output_scale']}",
        flush=True,
    )
    print(
        f"{len(paths)} run matrices -> {len(participants)} participants "
        f"({len(group_a)} group A, {len(group_b)} group B); "
        f"{n_voxels} voxels, {n_elements} edges",
        flush=True,
    )

    expected_outputs = [
        args.output_dir / f"s{identifier}_{properties['suffix']}.ccmat"
        for _, identifier, _, _, _ in resolved
    ]
    space = check_free_space(
        args.output_dir, expected_outputs, n_elements, args.allow_low_space
    )
    print(
        f"{space['pending_outputs']} matrices still to write, needing "
        f"{space['required_bytes'] / 2**30:.1f} GiB; "
        f"{space['available_bytes'] / 2**30:.1f} GiB free",
        flush=True,
    )

    if args.dry_run:
        print(
            "dry run: inputs, grouping, weights and free space validated; "
            "nothing written."
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── per-participant aggregation ─────────────────────────────────────────
    output_paths: list[Path] = []
    subject_records: list[dict[str, object]] = []
    clipped_total = 0

    for index, (group, identifier, run_paths, weights, timepoints) in enumerate(
        resolved, start=1
    ):
        output = expected_outputs[index - 1]
        output_paths.append(output)

        descriptor: dict[str, object] = {
            "mode": args.mode,
            "transform": properties["transform"],
            "input_scale": "pearson_r",
            "output_scale": properties["output_scale"],
            "group": group,
            "subject": identifier,
            "runs": [path.name for path in run_paths],
            "weights": weights,
            "n_voxels": n_voxels,
            "n_elements": n_elements,
        }

        if not args.force and sidecar_is_current(
            output, descriptor, n_voxels, n_elements
        ):
            print(
                f"[{index}/{len(resolved)}] s{identifier}: current output; skipping",
                flush=True,
            )
            recorded = json.loads(sidecar_path(output).read_text())
            statistics = recorded.get("clipping", {})
        else:
            print(
                f"[{index}/{len(resolved)}] s{identifier}: aggregating "
                f"{len(run_paths)} run(s) [{args.mode}]",
                flush=True,
            )
            statistics = aggregate_subject(
                run_paths,
                weights,
                properties["transform"],
                output,
                n_voxels,
                n_elements,
                args.chunk_elements,
            )
            sidecar = dict(descriptor)
            sidecar["clipping"] = statistics
            sidecar["clip_policy"] = CLIP_POLICY
            sidecar["accumulator_dtype"] = "float64"
            sidecar["storage_dtype"] = "float32"
            sidecar["created_utc"] = datetime.now(timezone.utc).isoformat()
            sidecar["software"] = software_metadata()
            sidecar_path(output).write_text(json.dumps(sidecar, indent=2) + "\n")

        clipped_total += int(statistics.get("clipped_total", 0) or 0)

        subject_records.append(
            {
                "group": group,
                "subject": identifier,
                "n_runs": len(run_paths),
                "runs": [file_fingerprint(path, args.checksum) for path in run_paths],
                "weights": weights,
                "n_timepoints": timepoints,
                "output": file_fingerprint(output, args.checksum),
                "clipping": statistics,
            }
        )

    # ── participant-ordered file list for 02_cudaPerm ───────────────────────
    args.output_file_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_file_list.write_text("".join(f"{path}\n" for path in output_paths))

    # Independent re-derivation of the group boundary from the written list,
    # rather than trusting the loop that produced it.
    written = [
        Path(line.strip())
        for line in args.output_file_list.read_text().splitlines()
        if line.strip()
    ]
    expected_a = list(group_a)
    if [subject_id(path) for path in written[: len(expected_a)]] != expected_a:
        raise RuntimeError("group A participants are not the leading output entries")
    if [subject_id(path) for path in written[len(expected_a) :]] != list(group_b):
        raise RuntimeError("group B participant ordering does not match the input")

    # ── manifest ────────────────────────────────────────────────────────────
    manifest = {
        "stage": "01p5_FisherCC",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "mode_role": properties["role"],
        "transform": properties["transform"],
        "weighting": properties["weighting"],
        "input_scale": "pearson_r",
        "output_scale": properties["output_scale"],
        "back_transformed": False,
        "clip_policy": CLIP_POLICY,
        "clip_value_r": CLAMP_R,
        "max_abs_z_after_clip": MAX_ABS_Z,
        "clipped_values_total": clipped_total,
        "accumulator_dtype": "float64",
        "storage_dtype": "float32",
        "chunk_elements": args.chunk_elements,
        "checksum_policy": args.checksum,
        "source_file_list": str(args.file_list.resolve()),
        "output_file_list": str(args.output_file_list.resolve()),
        "weight_table": (
            str(args.weight_table.resolve()) if args.weight_table else None
        ),
        "weight_column": args.weight_column if args.weight_table else None,
        "timepoint_column": args.timepoint_column,
        "group_a_runs": args.group_a_runs,
        "group_b_runs": len(paths) - args.group_a_runs,
        "group_a_subjects": len(group_a),
        "group_b_subjects": len(group_b),
        "n_voxels": n_voxels,
        "n_elements": n_elements,
        "disk_space_at_start": space,
        "software": software_metadata(),
        "command_line": sys.argv,
        "subjects": subject_records,
    }
    manifest_path = args.output_dir / f"fisher_aggregation_manifest_{args.mode}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"Wrote {len(output_paths)} participant matrices: "
        f"{len(group_a)} group A, {len(group_b)} group B",
        flush=True,
    )
    if properties["transform"] == "atanh":
        print(f"Clipped |r| >= 1 values: {clipped_total}", flush=True)
    print(f"File list: {args.output_file_list}", flush=True)
    print(f"Manifest : {manifest_path}", flush=True)
    print(
        f"02_cudaPerm/generatePermutations.py needs -nA {len(group_a)} "
        f"-nB {len(group_b)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
