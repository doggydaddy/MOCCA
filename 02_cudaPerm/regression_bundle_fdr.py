"""Regression of the bundle-level FDR path.

Covers the correction arithmetic (`false_discovery`), the opt-in per-bundle
emitter added to the C++ bundler, the orchestration glue in
`run_bundle_fwer.py`, and the invariant that ties the two correction schemes
together.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
COFFEE_DIR = PROJECT / "04_coffee-dac"
RESULT_FIXTURES = COFFEE_DIR / "archives/results_archives"
MASK = PROJECT / "templates/mask3mm.dump"
BACKEND = HERE / "build/bundle_fwer_omp"
RUNNER = HERE / "run_bundle_fwer.py"

for module_dir in (HERE, COFFEE_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import false_discovery
from bundle_fwer import compute_bundle_statistics
from regression_bundle_fwer_cpp import csv_to_sparse
from run_bundle_fwer import (
    BUNDLE_STATISTIC_COLUMNS,
    apply_fdr_correction,
    validate_bundle_statistics,
)


def classic_bh_rejections(p_values: np.ndarray, q: float) -> np.ndarray:
    """The 1995 step-up procedure written out longhand, as an oracle."""

    count = p_values.size
    order = np.argsort(p_values, kind="stable")
    ordered = p_values[order]
    thresholds = q * np.arange(1, count + 1) / count
    passing = np.nonzero(ordered <= thresholds)[0]
    rejected = np.zeros(count, dtype=bool)
    if passing.size:
        rejected[order[: passing[-1] + 1]] = True
    return rejected


class FalseDiscoveryMathTests(unittest.TestCase):
    # Benjamini & Hochberg (1995), Table 1: the 15 p-values of the Needleman
    # cardiology study, whose BH-adjusted values are a published reference.
    NEEDLEMAN = np.array([
        0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
        0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
    ])

    def test_benjamini_hochberg_matches_the_published_example(self) -> None:
        expected = np.array([
            0.0015, 0.0030, 0.0095, 0.035625, 0.0603,
            # 15/6 * 0.0278 = 0.0695 is pulled down to the i=7 value by the
            # step-up minimum; this pair is the point of the example.
            0.06385714285714286, 0.06385714285714286,
            0.0645, 0.0765, 0.486, 0.5811818181818182, 0.714875,
            0.7532307692307693, 0.8132142857142857, 1.0,
        ])
        np.testing.assert_allclose(
            false_discovery.benjamini_hochberg(self.NEEDLEMAN),
            expected, rtol=1e-12,
        )

    def test_matches_scipy_false_discovery_control(self) -> None:
        """Independent oracle for both procedures, over random and tied input."""

        from scipy.stats import false_discovery_control

        generator = np.random.default_rng(20260903)
        cases = [self.NEEDLEMAN]
        for trial in range(100):
            count = int(generator.integers(1, 120))
            values = generator.random(count)
            if trial % 3 == 0:  # ties, which the step-up ordering must survive
                values = np.round(values, 2)
            if trial % 5 == 0:  # a regime where the adjustment clips at 1
                values = 0.5 + values / 2
            cases.append(values)
        for index, values in enumerate(cases):
            with self.subTest(case=index):
                np.testing.assert_allclose(
                    false_discovery.benjamini_hochberg(values),
                    false_discovery_control(values, method="bh"),
                    rtol=1e-12, atol=1e-15,
                )
                np.testing.assert_allclose(
                    false_discovery.benjamini_yekutieli(values),
                    false_discovery_control(values, method="by"),
                    rtol=1e-12, atol=1e-15,
                )

    def test_bh_rejections_match_the_longhand_step_up_procedure(self) -> None:
        generator = np.random.default_rng(20260903)
        for trial in range(200):
            count = int(generator.integers(1, 60))
            p_values = generator.random(count)
            if trial % 3 == 0:  # force ties
                p_values = np.round(p_values, 2)
            for q in (0.01, 0.05, 0.1, 0.5):
                with self.subTest(trial=trial, q=q):
                    adjusted = false_discovery.benjamini_hochberg(p_values)
                    np.testing.assert_array_equal(
                        adjusted <= q, classic_bh_rejections(p_values, q)
                    )

    def test_benjamini_yekutieli_is_bh_scaled_by_the_harmonic_penalty(self) -> None:
        p_values = np.array([0.0001, 0.0002, 0.0004, 0.0009, 0.0011])
        penalty = false_discovery.harmonic_penalty(p_values.size)
        np.testing.assert_allclose(
            false_discovery.benjamini_yekutieli(p_values),
            np.minimum(false_discovery.benjamini_hochberg(p_values) * penalty, 1.0),
            rtol=1e-12,
        )

    def test_yekutieli_is_never_more_permissive_than_hochberg(self) -> None:
        generator = np.random.default_rng(7)
        for _ in range(100):
            p_values = generator.random(int(generator.integers(1, 80)))
            self.assertTrue(
                np.all(
                    false_discovery.benjamini_yekutieli(p_values)
                    >= false_discovery.benjamini_hochberg(p_values) - 1e-15
                )
            )

    def test_adjusted_values_are_monotone_in_the_raw_ordering(self) -> None:
        generator = np.random.default_rng(11)
        for _ in range(100):
            p_values = generator.random(40)
            for adjust in (
                false_discovery.benjamini_hochberg,
                false_discovery.benjamini_yekutieli,
            ):
                adjusted = adjust(p_values)
                order = np.argsort(p_values, kind="stable")
                self.assertTrue(np.all(np.diff(adjusted[order]) >= -1e-15))
                self.assertTrue(np.all(adjusted >= p_values - 1e-15))

    def test_adjusted_values_stay_within_the_unit_interval(self) -> None:
        p_values = np.array([0.4, 0.6, 0.9, 1.0])
        for adjust in (
            false_discovery.benjamini_hochberg,
            false_discovery.benjamini_yekutieli,
        ):
            adjusted = adjust(p_values)
            self.assertTrue(np.all(adjusted <= 1.0))
            self.assertTrue(np.all(adjusted >= 0.0))

    def test_singleton_and_empty_inputs(self) -> None:
        np.testing.assert_allclose(
            false_discovery.benjamini_hochberg(np.array([0.03])), [0.03]
        )
        np.testing.assert_allclose(
            false_discovery.benjamini_yekutieli(np.array([0.03])), [0.03]
        )
        self.assertEqual(false_discovery.benjamini_hochberg(np.array([])).size, 0)
        self.assertEqual(false_discovery.benjamini_yekutieli(np.array([])).size, 0)
        self.assertEqual(false_discovery.harmonic_penalty(0), 0.0)

    def test_rejects_invalid_p_values(self) -> None:
        for bad in (np.array([0.1, 1.5]), np.array([-0.1]), np.array([np.nan])):
            with self.assertRaises(ValueError):
                false_discovery.benjamini_hochberg(bad)


class PooledNullPValueTests(unittest.TestCase):
    def test_matches_a_bruteforce_count_including_ties(self) -> None:
        generator = np.random.default_rng(31415)
        for _ in range(50):
            null = np.round(generator.random(500) * 10, 1)
            observed = np.round(generator.random(20) * 10, 1)
            exceedances, p_values = false_discovery.pooled_null_p_values(
                observed, null
            )
            brute = np.array([np.count_nonzero(null >= v) for v in observed])
            np.testing.assert_array_equal(exceedances, brute)
            np.testing.assert_allclose(
                p_values, (1 + brute) / (null.size + 1), rtol=1e-15
            )

    def test_a_tie_counts_as_an_exceedance(self) -> None:
        null = np.array([1.0, 2.0, 3.0])
        exceedances, _ = false_discovery.pooled_null_p_values(
            np.array([2.0]), null
        )
        self.assertEqual(int(exceedances[0]), 2)

    def test_p_value_is_strictly_positive_and_bounded(self) -> None:
        null = np.arange(100, dtype=float)
        _, p_values = false_discovery.pooled_null_p_values(
            np.array([1e9, -1e9]), null
        )
        self.assertAlmostEqual(p_values[0], 1 / 101)
        self.assertAlmostEqual(p_values[1], 101 / 101)

    def test_rejects_an_empty_or_non_finite_null(self) -> None:
        with self.assertRaises(ValueError):
            false_discovery.pooled_null_p_values(np.array([1.0]), np.array([]))
        with self.assertRaises(ValueError):
            false_discovery.pooled_null_p_values(
                np.array([1.0]), np.array([1.0, np.inf])
            )


class FdrCorrectionFrameTests(unittest.TestCase):
    def observed_frame(self, statistics: list[float]) -> pd.DataFrame:
        return pd.DataFrame({
            "bundle": range(len(statistics)),
            "sign": [1] * len(statistics),
            "edge_count": [10] * len(statistics),
            "mass": statistics,
            "statistic": statistics,
        })

    def test_columns_and_significance_follow_the_selected_method(self) -> None:
        null = np.concatenate([np.arange(1, 500, dtype=float)])
        observed = self.observed_frame([600.0, 480.0, 300.0, 5.0])
        by_bh = apply_fdr_correction(observed, null, 0.05, "bh")
        by_by = apply_fdr_correction(observed, null, 0.05, "by")
        for frame in (by_bh, by_by):
            for column in (
                "null_bundle_exceedances", "p_uncorrected",
                "p_fdr_bh", "p_fdr_by", "significant",
            ):
                self.assertIn(column, frame.columns)
        # Identical p-value columns; only the reported decision differs.
        np.testing.assert_allclose(by_bh["p_fdr_bh"], by_by["p_fdr_bh"])
        np.testing.assert_array_equal(
            by_bh["significant"].to_numpy(), (by_bh["p_fdr_bh"] <= 0.05).to_numpy()
        )
        np.testing.assert_array_equal(
            by_by["significant"].to_numpy(), (by_by["p_fdr_by"] <= 0.05).to_numpy()
        )

    def test_empty_observed_frame_keeps_the_schema(self) -> None:
        corrected = apply_fdr_correction(
            self.observed_frame([]), np.arange(10, dtype=float), 0.05, "bh"
        )
        self.assertEqual(len(corrected), 0)
        self.assertIn("p_fdr_by", corrected.columns)

    def test_input_frame_is_not_mutated(self) -> None:
        observed = self.observed_frame([5.0, 4.0])
        before = list(observed.columns)
        apply_fdr_correction(observed, np.arange(10, dtype=float), 0.05, "bh")
        self.assertEqual(list(observed.columns), before)


class BundleStatisticIntegrityTests(unittest.TestCase):
    def maxima(self, counts: dict[int, int]) -> pd.DataFrame:
        return pd.DataFrame({
            "permutation": list(counts),
            "bundles": list(counts.values()),
        })

    def statistics(self, counts: dict[int, int]) -> pd.DataFrame:
        rows = [
            {"permutation": permutation, "statistic": float(index)}
            for permutation, count in counts.items()
            for index in range(count)
        ]
        return pd.DataFrame(rows, columns=["permutation", "statistic"])

    def test_consistent_files_pass(self) -> None:
        counts = {0: 3, 1: 0, 2: 5}
        validate_bundle_statistics(
            self.statistics(counts), self.maxima(counts), "file.csv"
        )

    def test_duplicated_batch_is_rejected(self) -> None:
        counts = {0: 3, 1: 2}
        doubled = pd.concat([self.statistics(counts), self.statistics({1: 2})])
        with self.assertRaisesRegex(RuntimeError, "disagreeing"):
            validate_bundle_statistics(doubled, self.maxima(counts), "file.csv")

    def test_missing_rows_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "disagreeing"):
            validate_bundle_statistics(
                self.statistics({0: 3}), self.maxima({0: 3, 1: 2}), "file.csv"
            )

    def test_a_permutation_outside_the_maxima_file_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "absent from the maxima"):
            validate_bundle_statistics(
                self.statistics({0: 1, 9: 1}), self.maxima({0: 1}), "file.csv"
            )


@unittest.skipUnless(BACKEND.exists(), "optimized bundle backend is not built")
class CppBundleStatisticsTests(unittest.TestCase):
    CASES = (
        "LTLEvsRTLE_runAll_10k_p0005_neg.csv",
        "controlsVSpatients_runAll_10k_p0001_pos.csv",
    )
    NEIGHBOR_DIST = 1.0
    MIN_SIZE = 10
    MIN_CLUSTER_VOXELS = 6

    def run_backend(
        self, prefix: Path, maxima: Path, bundles: Path | None
    ) -> None:
        command = [
            str(BACKEND), str(MASK), str(prefix), "0", "1", "extent", "0",
            str(self.NEIGHBOR_DIST), str(self.MIN_SIZE),
            str(self.MIN_CLUSTER_VOXELS), str(maxima), "--threads", "2",
        ]
        if bundles is not None:
            command.extend(["--bundle-statistics", str(bundles)])
        subprocess.run(command, check=True)

    def test_emitted_bundles_match_the_python_oracle(self) -> None:
        for filename in self.CASES:
            with self.subTest(filename=filename), \
                    tempfile.TemporaryDirectory() as temporary:
                raw = RESULT_FIXTURES / filename
                oracle = compute_bundle_statistics(
                    raw, statistic="extent",
                    neighbor_dist=self.NEIGHBOR_DIST,
                    min_size=self.MIN_SIZE,
                    min_cluster_voxels=self.MIN_CLUSTER_VOXELS,
                    strict_bundles=True, split_signs=False,
                )
                root = Path(temporary)
                prefix = root / "fixture"
                csv_to_sparse(raw, Path(f"{prefix}_perm000000.bsp"))
                maxima_path = root / "maxima.csv"
                bundles_path = root / "bundles.csv"
                self.run_backend(prefix, maxima_path, bundles_path)

                emitted = pd.read_csv(bundles_path)
                self.assertEqual(len(emitted), len(oracle.bundles))
                self.assertTrue((emitted["permutation"] == 0).all())
                self.assertTrue((emitted["observed"]).all())
                np.testing.assert_array_equal(
                    np.sort(emitted["edge_count"].to_numpy()),
                    np.sort([item.edge_count for item in oracle.bundles]),
                )
                np.testing.assert_allclose(
                    np.sort(emitted["statistic"].to_numpy(float)),
                    np.sort([item.statistic for item in oracle.bundles]),
                    rtol=1e-9, atol=1e-6,
                )

    def test_row_count_and_maximum_agree_with_the_maxima_file(self) -> None:
        """The two invariants the orchestrator's integrity check relies on."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / "fixture"
            csv_to_sparse(
                RESULT_FIXTURES / self.CASES[0], Path(f"{prefix}_perm000000.bsp")
            )
            maxima_path = root / "maxima.csv"
            bundles_path = root / "bundles.csv"
            self.run_backend(prefix, maxima_path, bundles_path)
            maxima = pd.read_csv(maxima_path)
            emitted = pd.read_csv(bundles_path)
            self.assertEqual(len(emitted), int(maxima.iloc[0]["bundles"]))
            self.assertAlmostEqual(
                float(emitted["statistic"].max()),
                float(maxima.iloc[0]["max_statistic"]),
                places=9,
            )
            validate_bundle_statistics(emitted, maxima, "bundles.csv")

    def test_header_matches_the_orchestrator_column_contract(self) -> None:
        """The C++ batches are concatenated verbatim into the master file.

        run_bundle_fwer.py streams each batch through without re-parsing it, so
        the backend's header must equal BUNDLE_STATISTIC_COLUMNS exactly or the
        master file would be silently misaligned.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / "fixture"
            csv_to_sparse(
                RESULT_FIXTURES / self.CASES[0], Path(f"{prefix}_perm000000.bsp")
            )
            bundles_path = root / "bundles.csv"
            self.run_backend(prefix, root / "maxima.csv", bundles_path)
            header = bundles_path.read_text().splitlines()[0]
            self.assertEqual(header, ",".join(BUNDLE_STATISTIC_COLUMNS))

    def test_flag_is_opt_in_and_leaves_existing_output_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = RESULT_FIXTURES / self.CASES[0]
            without = root / "without"
            with_flag = root / "with"
            for directory in (without, with_flag):
                directory.mkdir()
                csv_to_sparse(raw, directory / "fixture_perm000000.bsp")

            self.run_backend(without / "fixture", without / "maxima.csv", None)
            self.run_backend(
                with_flag / "fixture", with_flag / "maxima.csv",
                with_flag / "bundles.csv",
            )
            self.assertFalse((without / "bundles.csv").exists())
            self.assertEqual(
                (without / "maxima.csv").read_bytes(),
                (with_flag / "maxima.csv").read_bytes(),
            )

    def test_pooled_null_is_never_stricter_than_the_null_maxima(self) -> None:
        """Every permutation maximum is itself a pooled bundle statistic.

        So for any observed value, the pooled exceedance count is at least the
        maxima exceedance count -- the structural reason FDR is the weaker
        guarantee, asserted here on real bundle data rather than assumed.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / "fixture"
            csv_to_sparse(
                RESULT_FIXTURES / self.CASES[0], Path(f"{prefix}_perm000000.bsp")
            )
            maxima_path = root / "maxima.csv"
            bundles_path = root / "bundles.csv"
            self.run_backend(prefix, maxima_path, bundles_path)
            pooled = pd.read_csv(bundles_path)["statistic"].to_numpy(float)
            maxima = pd.read_csv(maxima_path)["max_statistic"].to_numpy(float)
            probes = np.quantile(pooled, [0.1, 0.5, 0.9, 0.99, 1.0])
            for probe in probes:
                self.assertGreaterEqual(
                    int(np.count_nonzero(pooled >= probe)),
                    int(np.count_nonzero(maxima >= probe)),
                )


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(RUNNER), "missing_filelist.txt",
             "missing_perms.txt", "out", *extra],
            capture_output=True, text=True,
        )

    def test_fwer_and_fdr_are_mutually_exclusive(self) -> None:
        result = self.run_cli("--cluster-forming-p", "5e-6", "--fwer", "--fdr")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not allowed with", result.stderr)

    def test_fdr_rejects_the_threshold_grid(self) -> None:
        result = self.run_cli(
            "--cluster-forming-p-grid", "1e-5", "5e-6", "--fdr"
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("--fdr cannot be combined", result.stderr)

    def test_fdr_q_must_be_a_probability(self) -> None:
        for value in ("0", "1", "1.5", "-0.1"):
            with self.subTest(value=value):
                result = self.run_cli(
                    "--cluster-forming-p", "5e-6", "--fdr", "--fdr-q", value
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("--fdr-q must be", result.stderr)

    def test_default_correction_is_fwer(self) -> None:
        sys.argv = ["run_bundle_fwer.py", "a", "b", "c",
                    "--cluster-forming-p", "5e-6"]
        from run_bundle_fwer import parse_args
        self.assertEqual(parse_args().correction, "fwer")

class NullCalibrationTests(unittest.TestCase):
    """The complete-null check in `fdr_null_calibration.py`."""

    def frame(self, statistics: dict[int, list[float]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"permutation": permutation, "statistic": value}
                for permutation, values in statistics.items()
                for value in values
            ]
        )

    def test_leave_one_out_matches_brute_force(self) -> None:
        from fdr_null_calibration import leave_one_out_p_values

        generator = np.random.default_rng(2026)
        for _ in range(20):
            counts = generator.integers(1, 12, size=15)
            frame = self.frame({
                index: list(np.round(generator.random(int(count)) * 10, 2))
                for index, count in enumerate(counts, start=1)
            })
            statistics = frame["statistic"].to_numpy(float)
            permutation = frame["permutation"].to_numpy(np.int64)
            exceedances, p_values = leave_one_out_p_values(statistics, permutation)
            for position in range(statistics.size):
                others = statistics[permutation != permutation[position]]
                expected = int(np.count_nonzero(others >= statistics[position]))
                self.assertEqual(int(exceedances[position]), expected)
                self.assertAlmostEqual(
                    p_values[position],
                    (1 + expected) / (others.size + 1),
                )

    def test_a_permutation_never_judges_itself(self) -> None:
        """A permutation holding every large value must not look significant."""

        from fdr_null_calibration import leave_one_out_p_values

        frame = self.frame({1: [100.0, 99.0, 98.0], 2: [1.0], 3: [1.0]})
        statistics = frame["statistic"].to_numpy(float)
        permutation = frame["permutation"].to_numpy(np.int64)
        exceedances, _ = leave_one_out_p_values(statistics, permutation)
        # Row 1's own three values are excluded, leaving only the two small ones.
        np.testing.assert_array_equal(exceedances[:3], [0, 0, 0])

    def test_exchangeable_data_controls_the_complete_null_rate(self) -> None:
        """The identity FDR = P(any rejection) must hold on exchangeable input.

        Bundle counts and statistics are drawn from one shared distribution, so
        every permutation is a genuine null draw and no procedure should
        declare anything in more than q of them.
        """

        from fdr_null_calibration import calibrate

        generator = np.random.default_rng(11)
        statistics = {}
        for permutation in range(1, 601):
            count = int(generator.integers(5, 60))
            statistics[permutation] = list(generator.lognormal(0.0, 1.0, count))
        _, summary = calibrate(self.frame(statistics), 0.05)
        self.assertLessEqual(summary["any_rejection_rate_bh"], 0.05)
        self.assertLessEqual(
            summary["any_rejection_rate_by"], summary["any_rejection_rate_bh"]
        )

    def test_a_planted_outlier_permutation_is_detected(self) -> None:
        """Sanity: the check can see rejections when they genuinely occur."""

        from fdr_null_calibration import calibrate

        generator = np.random.default_rng(13)
        statistics = {
            permutation: list(generator.lognormal(0.0, 0.5, 20))
            for permutation in range(1, 201)
        }
        statistics[201] = [1e6] * 10
        per_permutation, summary = calibrate(self.frame(statistics), 0.05)
        planted = per_permutation.loc[per_permutation["permutation"] == 201]
        self.assertGreater(int(planted["rejections_bh"].iloc[0]), 0)
        self.assertGreater(summary["any_rejection_rate_bh"], 0.0)

    def test_wilson_interval_brackets_the_rate(self) -> None:
        from fdr_null_calibration import wilson_interval

        low, high = wilson_interval(50, 1000)
        self.assertLess(low, 0.05)
        self.assertGreater(high, 0.05)
        self.assertEqual(wilson_interval(0, 0), [0.0, 1.0])

if __name__ == "__main__":
    unittest.main(verbosity=2)
