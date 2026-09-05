"""Regression tests for the post-hoc covariate analysis tools."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent / "02_cudaPerm"
for directory in (HERE, PIPELINE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from covariate_collinearity import explained
from covariate_decomposition import condensed_index, hc2_statistics, read_edges
from freedman_lane import build_plan, statistic_direct

CCMAT_HEADER = 24


class Hc2EquivalenceTests(unittest.TestCase):
    """The decomposition's vectorized HC2 must be the pipeline's statistic."""

    def test_matches_the_freedman_lane_reference(self) -> None:
        generator = np.random.default_rng(7)
        for _ in range(30):
            columns = int(generator.integers(2, 5))
            design = np.column_stack([
                np.ones(68), generator.normal(size=(68, columns - 2)),
                np.r_[np.ones(26), np.zeros(42)],
            ])
            group = design.shape[1] - 1
            # Heteroscedastic on purpose: HC2 exists for exactly this case.
            values = generator.normal(size=(68, 6)) * generator.uniform(
                0.5, 3.0, size=(68, 1)
            )
            plan = build_plan(design, group_index=group, hc_kind="HC2")
            reference = np.array([
                statistic_direct(plan, values[:, edge])
                for edge in range(values.shape[1])
            ])
            np.testing.assert_allclose(
                hc2_statistics(design, values, group), reference,
                rtol=1e-11, atol=1e-11,
            )

    def test_reproduces_welch_without_covariates(self) -> None:
        generator = np.random.default_rng(11)
        design = np.column_stack([np.ones(68), np.r_[np.ones(26), np.zeros(42)]])
        values = generator.normal(size=(68, 5)) * generator.uniform(
            0.5, 3.0, size=(68, 1)
        )
        welch = np.array([
            stats.ttest_ind(values[:26, e], values[26:, e], equal_var=False).statistic
            for e in range(values.shape[1])
        ])
        np.testing.assert_allclose(
            hc2_statistics(design, values, 1), welch, rtol=1e-12
        )


class EdgeIndexTests(unittest.TestCase):
    def test_condensed_index_round_trips_the_upper_triangle(self) -> None:
        n = 9
        expected = 0
        for first in range(n):
            for second in range(first + 1, n):
                self.assertEqual(
                    int(condensed_index(np.array([first]), np.array([second]), n)[0]),
                    expected,
                )
                expected += 1

    def test_index_is_orientation_invariant(self) -> None:
        a, b = np.array([3, 7, 1]), np.array([8, 2, 5])
        np.testing.assert_array_equal(
            condensed_index(a, b, 12), condensed_index(b, a, 12)
        )


class ReadEdgesTests(unittest.TestCase):
    def test_random_access_returns_the_right_values_in_request_order(self) -> None:
        """read_edges sorts internally for locality; order must be restored."""

        generator = np.random.default_rng(3)
        payload = generator.normal(size=5000).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for subject in range(3):
                path = Path(temporary) / f"s{subject}.ccmat"
                with path.open("wb") as stream:
                    stream.write(b"\0" * CCMAT_HEADER)
                    (payload + subject).tofile(stream)
                paths.append(path)
            wanted = generator.choice(5000, 400, replace=False)
            values = read_edges(paths, wanted)
            self.assertEqual(values.shape, (3, 400))
            for subject in range(3):
                np.testing.assert_allclose(
                    values[subject], (payload + subject)[wanted], rtol=0, atol=0
                )


class ExplainedVarianceTests(unittest.TestCase):
    def test_perfect_predictor_explains_everything(self) -> None:
        target = np.r_[np.ones(26), np.zeros(42)]
        self.assertAlmostEqual(explained(target, [target]), 1.0, places=12)

    def test_orthogonal_predictor_explains_nothing(self) -> None:
        generator = np.random.default_rng(5)
        target = np.r_[np.ones(26), np.zeros(42)]
        noise = generator.normal(size=68)
        noise -= noise.mean()
        noise -= (noise @ (target - target.mean())) / (
            (target - target.mean()) @ (target - target.mean())
        ) * (target - target.mean())
        self.assertAlmostEqual(explained(target, [noise]), 0.0, places=12)

    def test_matches_squared_correlation_for_one_predictor(self) -> None:
        generator = np.random.default_rng(9)
        target, predictor = generator.normal(size=68), generator.normal(size=68)
        self.assertAlmostEqual(
            explained(target, [predictor]),
            np.corrcoef(target, predictor)[0, 1] ** 2,
            places=12,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
