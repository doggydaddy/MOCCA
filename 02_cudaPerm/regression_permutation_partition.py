#!/usr/bin/env python3
"""Regression tests for the disjoint calibration/inference row partition.

`manuscript/ANALYSIS_DECISIONS.md` (2026-09-02, "disjoint calibration and
inference permutations") requires automated validation to reject overlapping
row ranges, duplicate rows where uniqueness is required, a non-observed row 0,
an incorrect total row count, and off-by-one errors in either subset -- and a
regression test over a small synthetic permutation file whose expected
calibration rows, inference rows, exceedance count and denominator are known
exactly.  That file is built here.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


CUDA_PERM_DIR = Path(__file__).resolve().parent
if str(CUDA_PERM_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_PERM_DIR))

import generatePermutations
from permutation_rows import (
    REPRESENTATIONS,
    DEFAULT_CALIBRATION_PERMUTATIONS,
    DEFAULT_CALIBRATION_START_ROW,
    DEFAULT_INFERENCE_PERMUTATIONS,
    DEFAULT_INFERENCE_START_ROW,
    RowPartition,
    validate_permutation_file,
)


# ── A synthetic master file with exactly known contents ─────────────────────
N_GROUP_A = 3
N_GROUP_B = 3
# 3-of-6 has 20 distinct assignments: row 0 observed plus 19 usable nulls.
SMALL = RowPartition(
    calibration_start=1, calibration_count=4,
    inference_start=5, inference_count=6,
)


def write_permutation_file(path: Path, rows: list[list[int]]) -> Path:
    np.savetxt(path, np.asarray(rows, dtype=np.uint16), fmt="% 4d")
    return path


def synthetic_rows(partition: RowPartition = SMALL) -> list[list[int]]:
    """Row 0 observed, then distinct 3-of-6 assignments filling the partition."""
    from itertools import combinations

    observed = list(range(N_GROUP_A))
    pool = [
        list(combination)
        for combination in combinations(range(N_GROUP_A + N_GROUP_B), N_GROUP_A)
        if list(combination) != observed
    ]
    needed = partition.required_rows - 1
    if needed > len(pool):
        raise ValueError("synthetic fixture is too small for this partition")
    return [observed, *pool[:needed]]


class DefaultsTests(unittest.TestCase):
    def test_production_defaults_match_the_decision_log(self) -> None:
        self.assertEqual(DEFAULT_CALIBRATION_PERMUTATIONS, 1000)
        self.assertEqual(DEFAULT_INFERENCE_PERMUTATIONS, 10000)
        self.assertEqual(DEFAULT_CALIBRATION_START_ROW, 1)
        self.assertEqual(DEFAULT_INFERENCE_START_ROW, 1001)

        partition = RowPartition()
        partition.validate()
        self.assertEqual(partition.calibration_rows[0], 1)
        self.assertEqual(partition.calibration_rows[-1], 1000)
        self.assertEqual(partition.inference_rows[0], 1001)
        self.assertEqual(partition.inference_rows[-1], 11000)
        self.assertEqual(partition.required_rows, 11001)
        self.assertEqual(partition.null_permutations_total, 11000)

    def test_fwer_denominator_excludes_calibration_rows(self) -> None:
        partition = RowPartition()
        # 11,000 nulls are computed in total, but only the 10,000 held-out
        # inference rows may reach the p-value.
        self.assertEqual(partition.null_permutations_total, 11000)
        self.assertEqual(partition.fwer_denominator, 10001)
        self.assertAlmostEqual(
            partition.to_manifest()["minimum_attainable_p_fwer"], 1 / 10001
        )

    def test_ranges_are_disjoint_and_cover_no_shared_row(self) -> None:
        partition = RowPartition()
        self.assertEqual(
            set(partition.calibration_rows) & set(partition.inference_rows), set()
        )
        self.assertNotIn(0, partition.calibration_rows)
        self.assertNotIn(0, partition.inference_rows)
        self.assertEqual(partition.inference_rows_with_observed[0], 0)


class PartitionValidationTests(unittest.TestCase):
    def test_overlapping_ranges_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap on rows 1000..1000"):
            RowPartition(
                calibration_start=1, calibration_count=1000,
                inference_start=1000, inference_count=10000,
            ).validate()

    def test_off_by_one_overlap_is_caught(self) -> None:
        """The classic error: inference starting on the last calibration row."""
        # calibration occupies rows 1..1000, so inference must start at 1001.
        RowPartition(
            calibration_start=1, calibration_count=1000,
            inference_start=1001, inference_count=10,
        ).validate()
        with self.assertRaisesRegex(ValueError, "overlap"):
            RowPartition(
                calibration_start=1, calibration_count=1001,
                inference_start=1001, inference_count=10,
            ).validate()

    def test_a_range_may_not_start_on_the_observed_row(self) -> None:
        for kwargs in (
            {"calibration_start": 0},
            {"inference_start": 0},
        ):
            with self.assertRaisesRegex(ValueError, "must start after"):
                RowPartition(**kwargs).validate()

    def test_non_positive_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must both be positive"):
            RowPartition(calibration_count=0).validate()
        with self.assertRaisesRegex(ValueError, "must both be positive"):
            RowPartition(inference_count=0).validate()


class FileValidationTests(unittest.TestCase):
    def test_valid_file_reports_exact_row_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = write_permutation_file(
                Path(raw) / "perm.txt", synthetic_rows()
            )
            report = validate_permutation_file(path, SMALL)
            self.assertEqual(report["permutation_file_rows"], 11)
            self.assertEqual(report["unique_permutation_rows"], 11)
            self.assertEqual(report["calibration_start_row"], 1)
            self.assertEqual(report["calibration_stop_row_exclusive"], 5)
            self.assertEqual(report["inference_start_row"], 5)
            self.assertEqual(report["inference_stop_row_exclusive"], 11)
            self.assertEqual(report["fwer_denominator"], 7)
            self.assertTrue(report["calibration_rows_excluded_from_fwer"])
            self.assertEqual(len(report["permutation_file_sha256"]), 64)

    def test_non_observed_row_zero_is_rejected(self) -> None:
        rows = synthetic_rows()
        rows[0] = [0, 1, 3]  # not the true grouping
        with tempfile.TemporaryDirectory() as raw:
            path = write_permutation_file(Path(raw) / "perm.txt", rows)
            with self.assertRaisesRegex(ValueError, "not the observed assignment"):
                validate_permutation_file(path, SMALL)

    def test_duplicate_rows_are_rejected(self) -> None:
        rows = synthetic_rows()
        rows[7] = list(rows[2])  # an inference row repeats a calibration row
        with tempfile.TemporaryDirectory() as raw:
            path = write_permutation_file(Path(raw) / "perm.txt", rows)
            with self.assertRaisesRegex(ValueError, "duplicate permutation row"):
                validate_permutation_file(path, SMALL)

    def test_observed_row_reused_as_a_null_is_rejected(self) -> None:
        rows = synthetic_rows()
        rows[6] = list(rows[0])
        with tempfile.TemporaryDirectory() as raw:
            path = write_permutation_file(Path(raw) / "perm.txt", rows)
            # caught as a duplicate first; without the uniqueness check it is
            # still caught by the observed-reuse rule
            with self.assertRaises(ValueError):
                validate_permutation_file(path, SMALL)
            with self.assertRaisesRegex(ValueError, "reuses the observed"):
                validate_permutation_file(path, SMALL, require_unique=False)

    def test_short_file_is_rejected_with_the_highest_requested_row(self) -> None:
        rows = synthetic_rows()[:-1]
        with tempfile.TemporaryDirectory() as raw:
            path = write_permutation_file(Path(raw) / "perm.txt", rows)
            with self.assertRaisesRegex(ValueError, "highest requested row is 10"):
                validate_permutation_file(path, SMALL)

    def test_extra_rows_are_rejected_unless_allowed(self) -> None:
        longer = RowPartition(
            calibration_start=1, calibration_count=4,
            inference_start=5, inference_count=7,
        )
        rows = synthetic_rows(longer)  # 12 rows, one more than SMALL needs
        with tempfile.TemporaryDirectory() as raw:
            path = write_permutation_file(Path(raw) / "perm.txt", rows)
            with self.assertRaisesRegex(ValueError, "needs exactly 11"):
                validate_permutation_file(path, SMALL)
            report = validate_permutation_file(path, SMALL, allow_extra_rows=True)
            self.assertEqual(report["permutation_file_rows"], 12)
            self.assertTrue(report["extra_rows_allowed"])


class GeneratorTests(unittest.TestCase):
    def test_generator_writes_a_validated_partitioned_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "perm.txt"
            generatePermutations.main(
                [
                    "-nA", "8", "-nB", "9", "-o", str(output),
                    "--calibration-permutations", "20",
                    "--inference-permutations", "30",
                    "--inference-start-row", "21",
                    "--seed", "11",
                ]
            )
            rows = np.loadtxt(output, dtype=np.int64)
            self.assertEqual(rows.shape, (51, 8))
            np.testing.assert_array_equal(rows[0], np.arange(8))
            self.assertEqual(len(np.unique(rows, axis=0)), 51)

            sidecar = json.loads(
                Path(str(output) + ".partition.json").read_text()
            )
            self.assertEqual(sidecar["seed"], 11)
            self.assertEqual(sidecar["calibration_permutations"], 20)
            self.assertEqual(sidecar["inference_permutations"], 30)
            self.assertEqual(sidecar["inference_start_row"], 21)
            self.assertEqual(sidecar["fwer_denominator"], 31)
            self.assertEqual(sidecar["unique_permutation_rows"], 51)

    def test_generator_rejects_a_gap_between_the_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(ValueError, "Expected --inference-start-row 21"):
                generatePermutations.main(
                    [
                        "-nA", "8", "-nB", "9",
                        "-o", str(Path(raw) / "perm.txt"),
                        "--calibration-permutations", "20",
                        "--inference-permutations", "30",
                        "--inference-start-row", "25",
                        "--seed", "11",
                    ]
                )

    def test_generator_rejects_nperm_contradicting_the_partition(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(ValueError, "contradicts the row partition"):
                generatePermutations.main(
                    [
                        "-nPerm", "5000", "-nA", "8", "-nB", "9",
                        "-o", str(Path(raw) / "perm.txt"),
                        "--calibration-permutations", "20",
                        "--inference-permutations", "30",
                        "--inference-start-row", "21",
                    ]
                )

    def test_generator_default_is_the_production_partition(self) -> None:
        parsed = generatePermutations.parse_args(
            ["-nA", "26", "-nB", "42", "-o", "/dev/null"]
        )
        self.assertEqual(parsed.calibration_permutations, 1000)
        self.assertEqual(parsed.inference_permutations, 10000)
        self.assertEqual(parsed.inference_start_row, 1001)
        self.assertIsNone(parsed.numberPermutations)


class InferenceWiringTests(unittest.TestCase):
    """The rows and denominator run_bundle_fwer.py would use, without a GPU."""

    def test_inference_selects_row_zero_plus_the_held_out_range(self) -> None:
        selected = SMALL.inference_rows_with_observed
        self.assertEqual(selected, [0, 5, 6, 7, 8, 9, 10])
        self.assertEqual(len(selected), SMALL.fwer_denominator)
        for row in SMALL.calibration_rows:
            self.assertNotIn(row, selected)

    def test_p_value_uses_only_inference_maxima(self) -> None:
        """Known-by-hand exceedance count and denominator."""
        # Six inference nulls; the observed statistic is 5.0.
        null_maxima = np.array([1.0, 9.0, 3.0, 7.0, 5.0, 2.0])
        observed = 5.0
        exceedances = int(np.count_nonzero(null_maxima >= observed))
        self.assertEqual(exceedances, 3)  # 9.0, 7.0 and the tie at 5.0
        p_fwer = (1 + exceedances) / SMALL.fwer_denominator
        self.assertEqual(SMALL.fwer_denominator, 7)
        self.assertAlmostEqual(p_fwer, 4 / 7)

        # Calibration maxima, however extreme, must not move the p-value.
        calibration_maxima = np.array([99.0, 98.0, 97.0, 96.0])
        self.assertEqual(
            (1 + int(np.count_nonzero(null_maxima >= observed)))
            / SMALL.fwer_denominator,
            p_fwer,
        )
        self.assertNotEqual(
            (
                1
                + int(
                    np.count_nonzero(
                        np.concatenate([null_maxima, calibration_maxima]) >= observed
                    )
                )
            )
            / (SMALL.fwer_denominator + len(calibration_maxima)),
            p_fwer,
        )

    def test_production_denominator_is_10001(self) -> None:
        partition = RowPartition()
        null_maxima = np.zeros(partition.inference_count)
        p_min = (1 + 0) / partition.fwer_denominator
        self.assertEqual(len(null_maxima), 10000)
        self.assertAlmostEqual(p_min, 1 / 10001)


class FullIndexRepresentationTests(unittest.TestCase):
    """Freedman--Lane needs complete reorderings, not group-membership rows."""

    PARTITION = RowPartition(
        calibration_start=1, calibration_count=4,
        inference_start=5, inference_count=6,
    )

    def full_index_rows(self, n: int = 9, seed: int = 0) -> list[list[int]]:
        generator = np.random.default_rng(seed)
        rows = [list(range(n))]
        seen = {tuple(rows[0])}
        while len(rows) < self.PARTITION.required_rows:
            candidate = tuple(int(v) for v in generator.permutation(n))
            if candidate in seen:
                continue
            seen.add(candidate)
            rows.append(list(candidate))
        return rows

    def test_both_representations_are_known(self) -> None:
        self.assertEqual(REPRESENTATIONS, ("group-a", "full-index"))

    def test_valid_full_index_file_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = write_permutation_file(
                Path(raw) / "perm.txt", self.full_index_rows()
            )
            report = validate_permutation_file(
                path, self.PARTITION, representation="full-index", n_subjects=9
            )
            self.assertEqual(report["permutation_representation"], "full-index")
            self.assertIn("complete participant reordering",
                          report["permutation_representation_note"])

    def test_group_membership_file_is_rejected_as_full_index(self) -> None:
        """The existing 26-of-68 files must not be usable for Freedman--Lane."""
        with tempfile.TemporaryDirectory() as raw:
            path = write_permutation_file(
                Path(raw) / "perm.txt", synthetic_rows(self.PARTITION)
            )
            # these rows are sorted 3-of-6 subsets, not permutations of 0..2
            with self.assertRaisesRegex(ValueError, "not a full permutation"):
                validate_permutation_file(
                    path, self.PARTITION, representation="full-index"
                )

    def test_a_row_that_is_not_a_permutation_is_rejected(self) -> None:
        rows = self.full_index_rows()
        rows[6][0] = rows[6][1]
        with tempfile.TemporaryDirectory() as raw:
            path = write_permutation_file(Path(raw) / "perm.txt", rows)
            with self.assertRaisesRegex(ValueError, "row 6 is not a full permutation"):
                validate_permutation_file(
                    path, self.PARTITION, representation="full-index"
                )

    def test_column_count_must_match_the_design(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = write_permutation_file(
                Path(raw) / "perm.txt", self.full_index_rows()
            )
            with self.assertRaisesRegex(ValueError, "must reorder every participant"):
                validate_permutation_file(
                    path, self.PARTITION, representation="full-index", n_subjects=68
                )

    def test_generator_writes_full_index_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "perm.txt"
            generatePermutations.main(
                [
                    "-nA", "8", "-nB", "9", "-o", str(output),
                    "--representation", "full-index",
                    "--calibration-permutations", "20",
                    "--inference-permutations", "30",
                    "--inference-start-row", "21",
                    "--seed", "3",
                ]
            )
            rows = np.loadtxt(output, dtype=np.int64)
            self.assertEqual(rows.shape, (51, 17))  # 17 participants, not 8
            np.testing.assert_array_equal(rows[0], np.arange(17))
            for index, row in enumerate(rows):
                np.testing.assert_array_equal(
                    np.sort(row), np.arange(17), err_msg=f"row {index}"
                )
            self.assertEqual(len(np.unique(rows, axis=0)), 51)
            sidecar = json.loads(Path(str(output) + ".partition.json").read_text())
            self.assertEqual(sidecar["permutation_representation"], "full-index")

    def test_default_representation_stays_group_a(self) -> None:
        """The unadjusted Welch pipeline must keep working unchanged."""
        parsed = generatePermutations.parse_args(
            ["-nA", "26", "-nB", "42", "-o", "/dev/null"]
        )
        self.assertEqual(parsed.representation, "group-a")


class CalibrationStabilityTests(unittest.TestCase):
    """The stability assessment must use calibration rows and nothing else."""

    GRID = [1e-3, 1e-4, 1e-5, 5e-6]
    ROWS = list(range(1, 1001))

    def curve(self, spans: list[tuple[float, float]], seed: int):
        import pandas as pd

        generator = np.random.default_rng(seed)
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "threshold_index": index,
                        "cluster_forming_p": threshold,
                        "permutation": self.ROWS,
                        "giant_voxel_fraction": generator.uniform(
                            *spans[index], size=len(self.ROWS)
                        ),
                    }
                )
                for index, threshold in enumerate(self.GRID)
            ],
            ignore_index=True,
        )

    def test_rule_picks_the_most_liberal_sub_critical_grid_point(self) -> None:
        from percolation_calibration import select_threshold_index

        # Only indices 2 and 3 stay under epsilon; the most liberal is 2.
        curve = self.curve(
            [(0.6, 0.95), (0.2, 0.5), (0.005, 0.03), (0.0, 0.01)], seed=0
        )
        index = select_threshold_index(curve, self.GRID, 0.05, 95.0)
        self.assertEqual(index, 2)
        self.assertEqual(self.GRID[index], 1e-5)

    def test_no_qualifying_grid_point_returns_none(self) -> None:
        from percolation_calibration import select_threshold_index

        curve = self.curve([(0.6, 0.95)] * 4, seed=0)
        self.assertIsNone(select_threshold_index(curve, self.GRID, 0.05, 95.0))

    def test_a_clear_transition_is_reported_as_stable(self) -> None:
        from percolation_calibration import assess_stability

        curve = self.curve(
            [(0.6, 0.95), (0.2, 0.5), (0.005, 0.03), (0.0, 0.01)], seed=0
        )
        report = assess_stability(
            curve, self.GRID, self.ROWS, 0.05, 95.0,
            replicates=200, subdivisions=4, seed=1,
        )
        self.assertEqual(report["bootstrap_modal_selection"], 1e-5)
        self.assertEqual(report["bootstrap_modal_fraction"], 1.0)
        self.assertTrue(report["subdivision_unanimous"])
        self.assertEqual(report["calibration_rows"], 1000)

    def test_a_borderline_transition_is_reported_as_unstable(self) -> None:
        from percolation_calibration import assess_stability

        # Grid point 1 sits right on epsilon: p95 of U(0, 0.0527) is ~0.050,
        # so resampling flips the selection between two adjacent grid points.
        curve = self.curve(
            [(0.6, 0.95), (0.0, 0.0527), (0.005, 0.03), (0.0, 0.01)], seed=3
        )
        report = assess_stability(
            curve, self.GRID, self.ROWS, 0.05, 95.0,
            replicates=300, subdivisions=4, seed=7,
        )
        self.assertLess(report["bootstrap_modal_fraction"], 1.0)
        self.assertGreater(len(report["bootstrap_selection_counts"]), 1)
        self.assertFalse(report["subdivision_unanimous"])

    def test_subdivisions_are_disjoint_and_cover_the_calibration_rows(self) -> None:
        blocks = np.array_split(np.asarray(self.ROWS), 4)
        self.assertEqual(sum(len(block) for block in blocks), len(self.ROWS))
        combined = np.concatenate(blocks)
        self.assertEqual(len(np.unique(combined)), len(self.ROWS))
        for row in combined:
            self.assertLessEqual(row, 1000)  # never an inference row


class CommandLineTests(unittest.TestCase):
    def test_null_permutations_flag_now_fails_with_guidance(self) -> None:
        result = subprocess.run(
            [
                sys.executable, str(CUDA_PERM_DIR / "run_bundle_fwer.py"),
                "missing_filelist.txt", "missing_perm.txt", "out",
                "--cluster-forming-p", "5e-6",
                "--null-permutations", "10000",
            ],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--inference-permutations", result.stderr)

    def test_calibration_and_inference_defaults_agree_across_programs(self) -> None:
        """Both stages must default to the same partition of the same file."""
        sources = {
            name: (CUDA_PERM_DIR / name).read_text()
            for name in ("run_bundle_fwer.py", "percolation_calibration.py")
        }
        for name, text in sources.items():
            self.assertIn("add_partition_arguments", text, msg=name)
            self.assertIn("validate_permutation_file", text, msg=name)


if __name__ == "__main__":
    unittest.main()
