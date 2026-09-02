#!/usr/bin/env python3
"""Regression tests for the covariate-adjusted Freedman--Lane implementation.

Covers the requirements in ``manuscript/ANALYSIS_DECISIONS.md`` (2026-09-02,
"covariate-adjusted control--TLE analysis"):

- the studentized statistic stays heteroscedasticity-robust, and reduces to
  the pipeline's existing Welch statistic exactly when covariates are dropped;
- the fast two-GEMM path agrees with a literal per-draw regression;
- permutations are full participant reorderings, not group-membership rows;
- the design matrix records its own coding and refuses confounded covariates;
- the group sign convention matches ``t = mean(A) - mean(B)``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats


CUDA_PERM_DIR = Path(__file__).resolve().parent
if str(CUDA_PERM_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_PERM_DIR))

import design_matrix as dm
import freedman_lane as fl


N_A, N_B = 26, 42
N = N_A + N_B


def make_design(seed: int = 0, heteroscedastic: bool = True):
    generator = np.random.default_rng(seed)
    group = np.concatenate([np.ones(N_A), np.zeros(N_B)])
    age = generator.normal(35, 10, N)
    sex = generator.integers(0, 2, N).astype(float)
    design = np.column_stack([np.ones(N), age - age.mean(), sex, group])
    return design, group, generator


class WelchEquivalenceTests(unittest.TestCase):
    """HC2 is the right generalization because it *is* Welch when p = 2."""

    def test_hc2_reproduces_welch_exactly_without_covariates(self) -> None:
        generator = np.random.default_rng(1)
        group = np.concatenate([np.ones(N_A), np.zeros(N_B)])
        design = np.column_stack([np.ones(N), group])
        plan = fl.build_plan(design, group_index=1, hc_kind="HC2")
        for _ in range(50):
            values = generator.normal(size=N) * np.where(group > 0, 2.0, 0.6)
            welch = stats.ttest_ind(
                values[:N_A], values[N_A:], equal_var=False
            ).statistic
            self.assertAlmostEqual(
                fl.statistic_direct(plan, values), float(welch), places=12
            )

    def test_hc0_and_hc3_do_not_equal_welch(self) -> None:
        """Guards against silently swapping the variance estimator."""
        generator = np.random.default_rng(2)
        group = np.concatenate([np.ones(N_A), np.zeros(N_B)])
        design = np.column_stack([np.ones(N), group])
        values = generator.normal(size=N) * np.where(group > 0, 2.0, 0.6)
        welch = float(
            stats.ttest_ind(values[:N_A], values[N_A:], equal_var=False).statistic
        )
        for kind in ("HC0", "HC3"):
            plan = fl.build_plan(design, 1, hc_kind=kind)
            self.assertNotAlmostEqual(
                fl.statistic_direct(plan, values), welch, places=6
            )

    def test_pooled_variance_statistic_is_not_what_we_compute(self) -> None:
        generator = np.random.default_rng(3)
        group = np.concatenate([np.ones(N_A), np.zeros(N_B)])
        design = np.column_stack([np.ones(N), group])
        plan = fl.build_plan(design, 1, hc_kind="HC2")
        values = generator.normal(size=N) * np.where(group > 0, 3.0, 0.5)
        pooled = float(
            stats.ttest_ind(values[:N_A], values[N_A:], equal_var=True).statistic
        )
        self.assertNotAlmostEqual(fl.statistic_direct(plan, values), pooled, places=3)


class ReducedFormTests(unittest.TestCase):
    """The two-GEMM path must equal a literal regression per draw."""

    def test_fast_path_matches_direct_regression(self) -> None:
        design, group, generator = make_design(seed=4)
        plan = fl.build_plan(design, group_index=3)
        values = generator.normal(size=N) * np.where(group > 0, 2.0, 0.7)
        permutations = np.array(
            [np.arange(N)] + [generator.permutation(N) for _ in range(200)]
        )
        fast = fl.statistics(plan, values[:, None], permutations, dtype=np.float64)
        direct = np.array(
            [fl.statistic_direct(plan, values, row) for row in permutations]
        )
        np.testing.assert_allclose(fast[:, 0], direct, rtol=1e-10, atol=1e-12)

    def test_identity_row_reproduces_the_unpermuted_statistic(self) -> None:
        design, group, generator = make_design(seed=5)
        plan = fl.build_plan(design, 3)
        values = generator.normal(size=N) * np.where(group > 0, 2.0, 0.7)
        observed = fl.statistics(
            plan, values[:, None], np.arange(N)[None, :], dtype=np.float64
        )[0, 0]
        self.assertAlmostEqual(observed, fl.statistic_direct(plan, values), places=10)

    def test_many_edges_at_once_match_one_edge_at_a_time(self) -> None:
        design, group, generator = make_design(seed=6)
        plan = fl.build_plan(design, 3)
        data = generator.normal(size=(N, 40)) * np.where(group > 0, 2.0, 0.7)[:, None]
        permutations = np.array(
            [np.arange(N)] + [generator.permutation(N) for _ in range(25)]
        )
        together = fl.statistics(plan, data, permutations, dtype=np.float64)
        for edge in range(data.shape[1]):
            one = fl.statistics(
                plan, data[:, edge : edge + 1], permutations, dtype=np.float64
            )
            np.testing.assert_allclose(together[:, edge], one[:, 0], rtol=1e-11)

    def test_float32_tables_stay_within_the_documented_tolerance(self) -> None:
        design, group, generator = make_design(seed=7)
        plan = fl.build_plan(design, 3)
        data = generator.normal(size=(N, 200)) * np.where(group > 0, 2.0, 0.7)[:, None]
        permutations = np.array(
            [np.arange(N)] + [generator.permutation(N) for _ in range(50)]
        )
        exact = fl.statistics(plan, data, permutations, dtype=np.float64)
        weights32, packed32 = fl.permutation_tables(
            plan, permutations, dtype=np.float32
        )
        approx = fl.statistics_from_tables(
            weights32, packed32, fl.nuisance_residuals(plan, data).astype(np.float32)
        )
        absolute = np.abs(approx - exact)
        self.assertLess(float(absolute.max()), 1e-5)
        # Relative error blows up only where the numerator cancels and t ~ 0;
        # those edges are nowhere near a cluster-forming threshold. Where the
        # statistic is large enough to matter, float32 is accurate to ~1e-6.
        large = np.abs(exact) > 1.0
        self.assertGreater(int(large.sum()), 100)
        relative = absolute[large] / np.abs(exact[large])
        self.assertLess(float(relative.max()), 5e-6)

    def test_nuisance_residuals_are_permutation_independent(self) -> None:
        """u is computed once per edge; that is what makes the GEMM worth it."""
        design, group, generator = make_design(seed=8)
        plan = fl.build_plan(design, 3)
        values = generator.normal(size=N)
        residuals = fl.nuisance_residuals(plan, values)
        nuisance = np.delete(design, 3, axis=1)
        # residuals are orthogonal to every nuisance column
        np.testing.assert_allclose(nuisance.T @ residuals, 0.0, atol=1e-10)

    def test_packed_quadratic_form_matches_the_dense_one(self) -> None:
        generator = np.random.default_rng(9)
        size = 11
        raw = generator.normal(size=(size, size))
        symmetric = raw + raw.T
        vector = generator.normal(size=size)
        self.assertAlmostEqual(
            float(fl.pack_symmetric(symmetric) @ fl.pack_outer(vector)),
            float(vector @ symmetric @ vector),
            places=10,
        )


class PermutationContractTests(unittest.TestCase):
    def test_group_membership_rows_are_rejected(self) -> None:
        """The existing 26-index permutation files cannot be used as-is."""
        design, _, _ = make_design(seed=10)
        plan = fl.build_plan(design, 3)
        membership = np.sort(
            np.random.default_rng(0).permutation(N)[:N_A]
        )[None, :]
        with self.assertRaisesRegex(ValueError, "membership of group A"):
            fl.permutation_tables(plan, membership)

    def test_a_row_with_repeats_is_rejected(self) -> None:
        design, _, _ = make_design(seed=11)
        plan = fl.build_plan(design, 3)
        broken = np.arange(N)[None, :].copy()
        broken[0, 5] = broken[0, 6]
        with self.assertRaisesRegex(ValueError, "full permutation"):
            fl.permutation_tables(plan, broken)

    def test_the_same_permutation_is_applied_to_every_edge(self) -> None:
        design, group, generator = make_design(seed=12)
        plan = fl.build_plan(design, 3)
        data = generator.normal(size=(N, 8))
        row = generator.permutation(N)
        fast = fl.statistics(plan, data, row[None, :], dtype=np.float64)
        for edge in range(data.shape[1]):
            self.assertAlmostEqual(
                fast[0, edge],
                fl.statistic_direct(plan, data[:, edge], row),
                places=10,
            )


class DegenerateDesignTests(unittest.TestCase):
    def test_rank_deficient_design_is_rejected(self) -> None:
        group = np.concatenate([np.ones(N_A), np.zeros(N_B)])
        design = np.column_stack([np.ones(N), group, group])
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            fl.build_plan(design, 1)

    def test_group_explained_by_nuisance_is_rejected(self) -> None:
        group = np.concatenate([np.ones(N_A), np.zeros(N_B)])
        # a nuisance column identical to group, with group last
        design = np.column_stack([np.ones(N), group, group * 1.0])
        with self.assertRaises(ValueError):
            fl.build_plan(design, 2)

    def test_residual_degrees_of_freedom(self) -> None:
        design, _, _ = make_design(seed=13)
        self.assertEqual(fl.effective_degrees_of_freedom(fl.build_plan(design, 3)), N - 4)


class DesignMatrixTests(unittest.TestCase):
    COVARIATES = (
        CUDA_PERM_DIR.parent / "data/share_with_KI/KI_shared_subjects_list.csv"
    )

    def fake_covariates(self) -> pd.DataFrame:
        rows = []
        for index in range(N):
            rows.append(
                {
                    "serial": 100 + index,
                    "tag": "control" if index < N_A else "L TLE",
                    "diagnosis": "",
                    "gender": "f" if index % 3 == 0 else "m",
                    "Hand": "L" if index >= N_A and index % 11 == 0 else "R",
                    "age": 20 + (index % 30),
                }
            )
        table = pd.DataFrame(rows)
        table["serial"] = table["serial"].astype(str)
        return table.set_index("serial", drop=False)

    def filelist(self) -> list[Path]:
        return [Path(f"/x/s{100 + index}_fisherz.ccmat") for index in range(N)]

    def test_default_model_columns_and_coding(self) -> None:
        design = dm.build_design(self.filelist(), N_A, self.fake_covariates())
        self.assertEqual(
            design.column_names, ["intercept", "age_centered", "sex_female", "group"]
        )
        self.assertEqual(design.nuisance_columns, ["intercept", "age_centered", "sex_female"])
        # age is centered on the analysis-sample mean
        self.assertAlmostEqual(float(design.matrix[:, 1].mean()), 0.0, places=10)
        self.assertEqual(design.coding["sex_female"]["encoding"], "1 = female, 0 = male")
        self.assertEqual(design.coding["group"]["indicator_level"], "A")

    def test_group_column_marks_the_leading_filelist_entries(self) -> None:
        design = dm.build_design(self.filelist(), N_A, self.fake_covariates())
        group = design.matrix[:, design.group_index]
        np.testing.assert_array_equal(group[:N_A], 1.0)
        np.testing.assert_array_equal(group[N_A:], 0.0)

    def test_sign_convention_matches_mean_a_minus_mean_b(self) -> None:
        design = dm.build_design(self.filelist(), N_A, self.fake_covariates())
        plan = fl.build_plan(design.matrix, design.group_index)
        # a value pattern where group A is clearly larger must give t > 0
        values = np.concatenate([np.ones(N_A) * 5.0, np.zeros(N_B)])
        values = values + np.random.default_rng(0).normal(scale=0.1, size=N)
        self.assertGreater(fl.statistic_direct(plan, values), 0.0)

    def test_handedness_is_refused_as_a_primary_covariate(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an automatic primary-model"):
            dm.build_design(
                self.filelist(), N_A, self.fake_covariates(), include_handedness=True
            )

    def test_handedness_restriction_keeps_group_ordering(self) -> None:
        design = dm.build_design(
            self.filelist(), N_A, self.fake_covariates(), restrict_handedness="R"
        )
        self.assertLess(design.n_subjects, N)
        self.assertEqual(design.group_a_subjects, N_A)  # all controls are right-handed
        self.assertEqual(
            design.group_labels,
            ["A"] * design.group_a_subjects
            + ["B"] * (design.n_subjects - design.group_a_subjects),
        )
        self.assertTrue(all(item["reason"] == "Hand != R" for item in design.excluded))

    def test_missing_covariate_row_is_rejected(self) -> None:
        files = self.filelist() + [Path("/x/s999_fisherz.ccmat")]
        with self.assertRaisesRegex(ValueError, "no covariate row"):
            dm.build_design(files, N_A, self.fake_covariates())

    def test_manifest_records_the_model_and_contrast(self) -> None:
        design = dm.build_design(self.filelist(), N_A, self.fake_covariates())
        manifest = design.to_manifest()
        self.assertEqual(manifest["contrast_of_interest"], "group")
        self.assertEqual(manifest["contrast_vector"], [0.0, 0.0, 0.0, 1.0])
        self.assertTrue(manifest["design_full_rank"])
        self.assertIn("mean_subtracted", manifest["coding"]["age_centered"])

    @unittest.skipUnless(COVARIATES.is_file(), "real covariate table not present")
    def test_real_covariate_table_matches_the_decision_log(self) -> None:
        table = dm.load_covariates(self.COVARIATES)
        self.assertEqual(len(table), 68)
        controls = table[table["tag"] == "control"]
        patients = table[table["tag"] != "control"]
        self.assertEqual(len(controls), 26)
        self.assertEqual(len(patients), 42)
        self.assertAlmostEqual(float(controls["age"].mean()), 32.2, places=1)
        self.assertAlmostEqual(float(patients["age"].mean()), 37.1, places=1)
        self.assertEqual(int((controls["gender"] == "f").sum()), 8)
        self.assertEqual(int((patients["gender"] == "f").sum()), 22)
        # every left-handed participant is a patient: the documented confound
        self.assertEqual(int((controls["Hand"] == "L").sum()), 0)
        self.assertEqual(int((patients["Hand"] == "L").sum()), 6)
        self.assertEqual(int((table["Hand"] == "R").sum()), 62)


class ExchangeabilityTests(unittest.TestCase):
    """The null distribution must actually be calibrated, not merely computable."""

    def test_uncorrected_p_is_uniform_under_the_null(self) -> None:
        design, group, generator = make_design(seed=30)
        plan = fl.build_plan(design, 3)
        # true-null edges: no group effect, but real nuisance signal and
        # heteroscedastic noise, which is what the statistic must survive
        n_edges = 400
        data = generator.normal(size=(N, n_edges)) * np.where(group > 0, 0.9, 0.4)[:, None]
        data += design[:, 1][:, None] * 0.02 + design[:, 2][:, None] * 0.15
        permutations = np.array(
            [np.arange(N)] + [generator.permutation(N) for _ in range(2000)]
        )
        statistics = fl.statistics(plan, data, permutations, dtype=np.float64)
        observed, null = np.abs(statistics[0]), np.abs(statistics[1:])
        p_values = (null >= observed[None, :]).mean(axis=0)

        # Uniform(0,1) has mean 0.5; the standard error here is ~1/sqrt(12*400)
        self.assertAlmostEqual(float(p_values.mean()), 0.5, delta=0.05)
        # and the nominal 5% false-positive rate must be about 5%
        self.assertLess(abs(float((p_values < 0.05).mean()) - 0.05), 0.03)

    def test_a_real_group_effect_is_detected(self) -> None:
        design, group, generator = make_design(seed=31)
        plan = fl.build_plan(design, 3)
        data = generator.normal(size=(N, 100)) * 0.5
        data += group[:, None] * 1.2
        permutations = np.array(
            [np.arange(N)] + [generator.permutation(N) for _ in range(500)]
        )
        statistics = fl.statistics(plan, data, permutations, dtype=np.float64)
        observed, null = np.abs(statistics[0]), np.abs(statistics[1:])
        p_values = (null >= observed[None, :]).mean(axis=0)
        self.assertGreater(float((p_values < 0.05).mean()), 0.9)
        self.assertGreater(float(statistics[0].mean()), 0.0)  # group A greater

    def test_nuisance_signal_alone_does_not_create_group_effects(self) -> None:
        """Age/sex structure must be absorbed, not read as a group difference."""
        design, group, generator = make_design(seed=32)
        plan = fl.build_plan(design, 3)
        # data driven purely by the covariates, with no group term at all
        data = (
            design[:, 1][:, None] * generator.normal(size=(1, 300))
            + design[:, 2][:, None] * generator.normal(size=(1, 300))
            + generator.normal(size=(N, 300)) * 0.3
        )
        permutations = np.array(
            [np.arange(N)] + [generator.permutation(N) for _ in range(1000)]
        )
        statistics = fl.statistics(plan, data, permutations, dtype=np.float64)
        observed, null = np.abs(statistics[0]), np.abs(statistics[1:])
        p_values = (null >= observed[None, :]).mean(axis=0)
        self.assertLess(abs(float((p_values < 0.05).mean()) - 0.05), 0.03)


class TableRoundTripTests(unittest.TestCase):
    def test_cli_writes_tables_that_reproduce_the_statistic(self) -> None:
        design, group, generator = make_design(seed=20)
        permutations = np.array(
            [np.arange(N)] + [generator.permutation(N) for _ in range(30)]
        )
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            np.savez(
                directory / "design.npz",
                matrix=design,
                column_names=np.array(
                    ["intercept", "age_centered", "sex_female", "group"]
                ),
                group_index=np.array(3),
            )
            np.savetxt(directory / "perm.txt", permutations, fmt="%d")
            # 31 rows: row 0 observed, row 1 calibration-only, rows 2..30 inference
            fl.main(
                [
                    "--design", str(directory / "design.npz"),
                    "--permutations", str(directory / "perm.txt"),
                    "--output-dir", str(directory / "out"),
                    "--dtype", "float64",
                    "--rows", "all",
                    "--calibration-permutations", "1",
                    "--calibration-start-row", "1",
                    "--inference-permutations", "29",
                    "--inference-start-row", "2",
                ]
            )
            tables = np.load(directory / "out/freedman_lane_tables.npz")
            plan = fl.build_plan(design, 3)
            values = generator.normal(size=N) * np.where(group > 0, 2.0, 0.7)
            produced = fl.statistics_from_tables(
                tables["weights"],
                tables["packed_kernel"],
                fl.nuisance_residuals(plan, values[:, None]),
            )
            expected = np.array(
                [fl.statistic_direct(plan, values, row) for row in permutations]
            )
            np.testing.assert_allclose(produced[:, 0], expected, rtol=1e-10)

            manifest = json.loads(
                (directory / "out/freedman_lane_manifest.json").read_text()
            )
            self.assertEqual(manifest["statistic"], "HC2-studentized group coefficient")
            self.assertEqual(
                manifest["permutation_representation"],
                "full participant index order per row",
            )
            self.assertEqual(manifest["residual_degrees_of_freedom"], N - 4)

    def test_cli_rejects_a_non_identity_row_zero(self) -> None:
        design, _, generator = make_design(seed=21)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            np.savez(
                directory / "design.npz",
                matrix=design,
                column_names=np.array(["a", "b", "c", "group"]),
                group_index=np.array(3),
            )
            np.savetxt(
                directory / "perm.txt",
                np.array(
                    [generator.permutation(N), np.arange(N), generator.permutation(N)]
                ),
                fmt="%d",
            )
            with self.assertRaisesRegex(
                ValueError, "observed assignment|identity permutation"
            ):
                fl.main(
                    [
                        "--design", str(directory / "design.npz"),
                        "--permutations", str(directory / "perm.txt"),
                        "--output-dir", str(directory / "out"),
                        "--calibration-permutations", "1",
                        "--calibration-start-row", "1",
                        "--inference-permutations", "1",
                        "--inference-start-row", "2",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
