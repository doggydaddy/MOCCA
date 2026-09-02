#!/usr/bin/env python3
"""Disjoint calibration/inference partition of one master permutation file.

Per ``manuscript/ANALYSIS_DECISIONS.md`` (2026-09-02, "disjoint calibration
and inference permutations"), a production analysis described as using 10,000
inference permutations generates 11,000 unique null permutations plus the
observed assignment, partitioned as::

    row 0             observed assignment
    rows 1..1000      calibration set only   (1,000 null permutations)
    rows 1001..11000  inference set only    (10,000 null permutations)

The two null subsets must be disjoint.  One master file with recorded,
non-overlapping row ranges is easier to audit than separate seeds or files,
and guarantees that no label assignment is reused across the two stages.

Because the cluster-forming threshold is chosen on the calibration rows only,
those rows must not enter the FWER numerator or denominator.  The minimum
attainable production p-value therefore stays ``1 / 10001`` even though
11,000 null permutations were computed in total.

This module makes the partition an explicit, validated configuration rather
than an implicit convention.  ``RowPartition.validate`` and
``validate_permutation_file`` between them reject overlapping row ranges,
duplicate rows where uniqueness is required, a non-observed row 0, an
incorrect total row count, and off-by-one errors in either subset.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ── Production defaults (see the decision log) ──────────────────────────────
DEFAULT_CALIBRATION_PERMUTATIONS = 1_000
DEFAULT_INFERENCE_PERMUTATIONS = 10_000
DEFAULT_CALIBRATION_START_ROW = 1
DEFAULT_INFERENCE_START_ROW = 1_001

OBSERVED_ROW = 0

# ── Row representations ─────────────────────────────────────────────────────
# "group-a"    : each row lists the sorted indices assigned to group A.  This
#                is what the Welch CUDA backend consumes.
# "full-index" : each row is a complete permutation of 0..n-1.  Freedman--Lane
#                permutes all n residual vectors, so a group-membership row
#                does not specify the draw; the existing subject-level
#                permutation files cannot be reused for the adjusted model.
REPRESENTATIONS = ("group-a", "full-index")
DEFAULT_REPRESENTATION = "group-a"


@dataclass(frozen=True)
class RowPartition:
    """A validated observed/calibration/inference split of one master file."""

    calibration_start: int = DEFAULT_CALIBRATION_START_ROW
    calibration_count: int = DEFAULT_CALIBRATION_PERMUTATIONS
    inference_start: int = DEFAULT_INFERENCE_START_ROW
    inference_count: int = DEFAULT_INFERENCE_PERMUTATIONS

    # ── derived ranges ──────────────────────────────────────────────────────
    @property
    def calibration_stop(self) -> int:
        """Exclusive end of the calibration range."""
        return self.calibration_start + self.calibration_count

    @property
    def inference_stop(self) -> int:
        """Exclusive end of the inference range."""
        return self.inference_start + self.inference_count

    @property
    def calibration_rows(self) -> list[int]:
        return list(range(self.calibration_start, self.calibration_stop))

    @property
    def inference_rows(self) -> list[int]:
        return list(range(self.inference_start, self.inference_stop))

    @property
    def inference_rows_with_observed(self) -> list[int]:
        """Row 0 followed by the inference rows: exactly what inference runs."""
        return [OBSERVED_ROW, *self.inference_rows]

    @property
    def required_rows(self) -> int:
        """Total rows the master permutation file must contain."""
        return max(self.calibration_stop, self.inference_stop)

    @property
    def null_permutations_total(self) -> int:
        return self.calibration_count + self.inference_count

    @property
    def fwer_denominator(self) -> int:
        """Denominator of ``p_FWER``: inference nulls plus the observed row.

        Calibration rows are excluded, so this is *not* the total null count.
        """
        return self.inference_count + 1

    # ── validation ──────────────────────────────────────────────────────────
    def validate(self) -> None:
        if self.calibration_count < 1 or self.inference_count < 1:
            raise ValueError(
                "calibration and inference permutation counts must both be "
                f"positive (got {self.calibration_count} and "
                f"{self.inference_count})"
            )
        for name, start in (
            ("calibration", self.calibration_start),
            ("inference", self.inference_start),
        ):
            if start <= OBSERVED_ROW:
                raise ValueError(
                    f"{name} rows must start after the observed row "
                    f"{OBSERVED_ROW} (got start row {start})"
                )
        overlap_start = max(self.calibration_start, self.inference_start)
        overlap_stop = min(self.calibration_stop, self.inference_stop)
        if overlap_start < overlap_stop:
            raise ValueError(
                "calibration and inference row ranges overlap on rows "
                f"{overlap_start}..{overlap_stop - 1}: calibration is "
                f"[{self.calibration_start}, {self.calibration_stop}) and "
                f"inference is [{self.inference_start}, {self.inference_stop}). "
                "The two null subsets must be disjoint."
            )

    def to_manifest(self) -> dict[str, object]:
        return {
            "observed_row": OBSERVED_ROW,
            "calibration_start_row": self.calibration_start,
            "calibration_stop_row_exclusive": self.calibration_stop,
            "calibration_permutations": self.calibration_count,
            "inference_start_row": self.inference_start,
            "inference_stop_row_exclusive": self.inference_stop,
            "inference_permutations": self.inference_count,
            "null_permutations_total": self.null_permutations_total,
            "required_permutation_rows": self.required_rows,
            "ranges_disjoint": True,
            "calibration_rows_excluded_from_fwer": True,
            "fwer_denominator": self.fwer_denominator,
            "minimum_attainable_p_fwer": 1.0 / self.fwer_denominator,
        }

    def describe(self) -> str:
        return (
            f"row {OBSERVED_ROW} observed | "
            f"calibration rows {self.calibration_start}.."
            f"{self.calibration_stop - 1} ({self.calibration_count}) | "
            f"inference rows {self.inference_start}.."
            f"{self.inference_stop - 1} ({self.inference_count}) | "
            f"p_FWER denominator {self.fwer_denominator}"
        )


# ── CLI wiring ──────────────────────────────────────────────────────────────
def add_partition_arguments(
    parser: argparse.ArgumentParser, *, stage: str, include_file_checks: bool = True
) -> None:
    """Attach the four partition options to a parser.

    ``stage`` is ``"calibration"``, ``"inference"`` or ``"generate"``; every
    stage takes all four values so that each one's manifest records the whole
    partition, not only the half it consumed.
    """
    group = parser.add_argument_group(
        "permutation row partition",
        "Disjoint calibration/inference split of the master permutation file "
        "(see manuscript/ANALYSIS_DECISIONS.md, 2026-09-02).",
    )
    group.add_argument(
        "--calibration-permutations",
        type=int,
        default=DEFAULT_CALIBRATION_PERMUTATIONS,
        help=(
            "null rows reserved for cluster-forming-threshold calibration "
            f"(default: {DEFAULT_CALIBRATION_PERMUTATIONS})"
            + ("" if stage == "calibration" else "; recorded, not consumed, here")
        ),
    )
    group.add_argument(
        "--calibration-start-row",
        type=int,
        default=DEFAULT_CALIBRATION_START_ROW,
        help=f"first calibration row (default: {DEFAULT_CALIBRATION_START_ROW})",
    )
    group.add_argument(
        "--inference-permutations",
        type=int,
        default=DEFAULT_INFERENCE_PERMUTATIONS,
        help=(
            "null rows reserved for FWER inference "
            f"(default: {DEFAULT_INFERENCE_PERMUTATIONS})"
            + ("" if stage == "inference" else "; recorded, not consumed, here")
        ),
    )
    group.add_argument(
        "--inference-start-row",
        type=int,
        default=DEFAULT_INFERENCE_START_ROW,
        help=f"first inference row (default: {DEFAULT_INFERENCE_START_ROW})",
    )
    if include_file_checks:
        group.add_argument(
            "--allow-extra-permutation-rows",
            action="store_true",
            help=(
                "accept a master permutation file with more rows than the "
                "partition needs; by default the row count must match exactly, "
                "so a file built for a different design is rejected rather "
                "than silently truncated"
            ),
        )


def partition_from_args(args: argparse.Namespace) -> RowPartition:
    partition = RowPartition(
        calibration_start=args.calibration_start_row,
        calibration_count=args.calibration_permutations,
        inference_start=args.inference_start_row,
        inference_count=args.inference_permutations,
    )
    partition.validate()
    return partition


# ── Master-file validation ──────────────────────────────────────────────────
def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_permutation_rows(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if rows.size == 0:
        raise ValueError(f"permutation file is empty: {path}")
    return rows


def validate_permutation_file(
    path: Path,
    partition: RowPartition,
    *,
    allow_extra_rows: bool = False,
    require_unique: bool = True,
    representation: str = DEFAULT_REPRESENTATION,
    n_subjects: int | None = None,
) -> dict[str, object]:
    """Check a master permutation file against a partition.

    Rejects a non-observed row 0, an incorrect total row count, ranges that
    run past the end of the file, and duplicate rows.  Returns a manifest
    fragment describing what was checked, so the caller can record the file's
    identity alongside the partition it was used under.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {REPRESENTATIONS}, got {representation!r}"
        )
    partition.validate()
    rows = load_permutation_rows(path)
    total_rows, n_columns = rows.shape

    # Row 0 is the observed assignment under both representations: the first
    # nA filelist entries for "group-a", the identity order for "full-index".
    expected_observed = np.arange(n_columns, dtype=np.int64)
    if not np.array_equal(rows[OBSERVED_ROW], expected_observed):
        raise ValueError(
            f"row {OBSERVED_ROW} of {path} is not the observed assignment "
            f"(expected 0..{n_columns - 1}, got "
            f"{rows[OBSERVED_ROW][:8].tolist()}...). Row 0 must be the true "
            "grouping; generatePermutations.py prepends it automatically."
        )

    required = partition.required_rows
    if total_rows < required:
        raise ValueError(
            f"{path} has {total_rows} rows but the partition needs {required} "
            f"({partition.describe()}). The highest requested row is "
            f"{required - 1}."
        )
    if total_rows != required and not allow_extra_rows:
        raise ValueError(
            f"{path} has {total_rows} rows but the partition needs exactly "
            f"{required}. Generate a file for this design, or pass "
            "--allow-extra-permutation-rows to use a longer file deliberately."
        )

    duplicate_example: list[int] | None = None
    n_unique = None
    if require_unique:
        _, first_index, counts = np.unique(
            rows, axis=0, return_index=True, return_counts=True
        )
        n_unique = int(len(first_index))
        if n_unique != total_rows:
            repeated = first_index[counts > 1]
            duplicate_example = rows[int(repeated[0])].tolist()
            raise ValueError(
                f"{path} contains {total_rows - n_unique} duplicate "
                f"permutation row(s); the calibration and inference sets must "
                f"draw distinct assignments. First repeated row: "
                f"{duplicate_example[:8]}..."
            )

    if representation == "full-index":
        if n_subjects is not None and n_columns != n_subjects:
            raise ValueError(
                f"{path} has {n_columns} columns but the design has "
                f"{n_subjects} participants; a full-index row must reorder "
                "every participant."
            )
        expected_sorted = np.arange(n_columns, dtype=np.int64)
        offending = np.flatnonzero(
            (np.sort(rows, axis=1) != expected_sorted).any(axis=1)
        )
        if offending.size:
            raise ValueError(
                f"{path} row {int(offending[0])} is not a full permutation of "
                f"0..{n_columns - 1}. Freedman-Lane needs a complete "
                "participant reordering; group-membership rows cannot be used."
            )
    elif n_subjects is not None and int(rows.max()) >= n_subjects:
        raise ValueError(
            f"{path} references participant index {int(rows.max())} but there "
            f"are only {n_subjects} participants."
        )

    # Row 0 must not reappear as a null draw in either subset.
    null_rows = rows[partition.calibration_rows + partition.inference_rows]
    if np.any(np.all(null_rows == expected_observed, axis=1)):
        raise ValueError(
            f"{path} reuses the observed assignment as a null permutation "
            "inside the calibration or inference range."
        )

    return {
        "permutation_file": str(path.resolve()),
        "permutation_file_sha256": sha256_file(path),
        "permutation_representation": representation,
        "permutation_representation_note": (
            "group A member indices, one row per assignment"
            if representation == "group-a"
            else "complete participant reordering, one row per Freedman-Lane draw"
        ),
        "permutation_file_rows": int(total_rows),
        "permutation_file_columns": int(n_columns),
        "unique_permutation_rows": n_unique,
        "uniqueness_checked": require_unique,
        "extra_rows_allowed": allow_extra_rows,
        **partition.to_manifest(),
    }
