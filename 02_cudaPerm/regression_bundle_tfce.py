"""Regression of the C++ TFCE statistic against the Python reference.

`tfce.py` is the oracle: it integrates with scipy's Student-t quantiles and the
already-validated strict bundler, while the backend uses tabulated quantiles
and its own labelling. These tests hold the two together on synthetic
fixtures whose geometry produces many distinct multi-edge bundles across the
height grid.
"""

from __future__ import annotations

from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
COFFEE_DIR = PROJECT / "04_coffee-dac"
BACKEND = HERE / "build/bundle_fwer_omp"
HEADER = struct.Struct("<IIQQQQfI")
MAGIC = 0x4C444E42
DF_STORED_VERSION = 3
DF_STORED_RECORDS = np.dtype(
    [("edge_index", "<u8"), ("tstat", "<f4"), ("degrees_of_freedom", "<f4")]
)
SUBJECTS = 68

for module_dir in (HERE, COFFEE_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import tfce as tfce_reference


def condensed_index(first: int, second: int, n_voxels: int) -> int:
    row, column = sorted((first, second))
    return row * (2 * n_voxels - row - 1) // 2 + column - row - 1


def slab_mask(width: int, depth: int, height: int) -> np.ndarray:
    return np.array(
        [(x, y, z) for x in range(width) for y in range(depth) for z in range(height)],
        dtype=int,
    )


def build_fixture(
    seed: int, n_hubs: int, z_values: np.ndarray, patch: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A mask plus hub-and-patch edges spanning the height grid.

    Each hub is joined to every voxel of a small nearby patch, so those edges
    share an endpoint and have mutually neighbouring free ends -- the two
    conditions the strict bundler unites on. Varying |t| across the grid makes
    bundles shrink as the height rises, which is the behaviour TFCE integrates.
    """

    generator = np.random.default_rng(seed)
    mask = slab_mask(12, 12, 4)
    index_of = {tuple(coord): index for index, coord in enumerate(mask)}

    endpoints: list[tuple[int, int]] = []
    for _ in range(n_hubs):
        hub = tuple(generator.integers([0, 0, 0], [12, 12, 4]))
        corner = generator.integers([0, 0, 0], [12 - patch, 12 - patch, 4 - patch])
        for dx in range(patch):
            for dy in range(patch):
                for dz in range(patch):
                    target = (corner[0] + dx, corner[1] + dy, corner[2] + dz)
                    if target == hub:
                        continue
                    pair = (index_of[hub], index_of[target])
                    if pair[0] != pair[1]:
                        endpoints.append(tuple(sorted(pair)))
    endpoints = sorted(set(endpoints))

    count = len(endpoints)
    degrees_of_freedom = generator.uniform(30.0, 64.0, size=count)
    # Spread |t| so that edges leave the suprathreshold set at different
    # heights, then push any value sitting on a grid critical value away from
    # it: the backend interpolates its quantiles, so a t within rounding of a
    # threshold could legitimately be classified either way.
    span = tfce_reference.critical_t(degrees_of_freedom, float(z_values[0]))
    top = tfce_reference.critical_t(degrees_of_freedom, float(z_values[-1]) + 1.0)
    magnitude = span + generator.random(count) * (top - span)
    for height in z_values:
        critical = tfce_reference.critical_t(degrees_of_freedom, float(height))
        too_close = np.abs(magnitude - critical) < 1e-3
        magnitude[too_close] = critical[too_close] + 5e-3
    signs = generator.choice([-1.0, 1.0], size=count)
    return mask, np.array(endpoints), (magnitude * signs), degrees_of_freedom


def write_v3_sparse(
    path: Path, endpoints: np.ndarray, tstat: np.ndarray,
    degrees_of_freedom: np.ndarray, n_voxels: int,
) -> None:
    records = np.empty(len(endpoints), dtype=DF_STORED_RECORDS)
    records["edge_index"] = [
        condensed_index(int(a), int(b), n_voxels) for a, b in endpoints
    ]
    records["tstat"] = tstat
    records["degrees_of_freedom"] = degrees_of_freedom
    with path.open("wb") as stream:
        stream.write(
            HEADER.pack(
                MAGIC, DF_STORED_VERSION, 0, len(records), n_voxels,
                n_voxels * (n_voxels - 1) // 2, 0.0, 3,
            )
        )
        records.tofile(stream)


def reference_scores(
    mask: np.ndarray, endpoints: np.ndarray, tstat: np.ndarray,
    degrees_of_freedom: np.ndarray, z_values: np.ndarray, z_step: float,
    extent_exponent: float, height_exponent: float,
) -> np.ndarray:
    """Per-edge TFCE from the oracle, with the backend's sign split."""

    # float32 storage is what the backend reads, so the oracle must see the
    # same values rather than the full-precision originals.
    tstat = np.asarray(tstat, dtype=np.float32).astype(float)
    degrees_of_freedom = np.asarray(
        degrees_of_freedom, dtype=np.float32
    ).astype(float)
    coordinates = np.hstack([mask[endpoints[:, 0]], mask[endpoints[:, 1]]])
    scores = np.zeros(len(endpoints), dtype=float)
    for keep in (tstat > 0, tstat < 0):
        members = np.nonzero(keep)[0]
        if members.size == 0:
            continue
        scores[members] = tfce_reference.tfce_scores(
            coordinates[members], tstat[members], degrees_of_freedom[members],
            z_values=z_values, z_step=z_step,
            extent_exponent=extent_exponent, height_exponent=height_exponent,
            neighbor_dist=1.0,
        )
    return scores


@unittest.skipUnless(BACKEND.exists(), "optimized bundle backend is not built")
class TfceEquivalenceTests(unittest.TestCase):
    Z_MIN, Z_MAX, Z_STEP = 2.0, 4.0, 0.25

    def run_backend(
        self, root: Path, mask: np.ndarray, endpoints: np.ndarray,
        tstat: np.ndarray, degrees_of_freedom: np.ndarray,
        extent_exponent: float, height_exponent: float,
        min_size: int = 1, min_cluster_voxels: int = 1,
    ) -> tuple[pd.Series, pd.DataFrame]:
        mask_path = root / "mask.dump"
        np.savetxt(mask_path, mask, fmt="%d")
        prefix = root / "fixture"
        write_v3_sparse(
            Path(f"{prefix}_perm000000.bsp"), endpoints, tstat,
            degrees_of_freedom, len(mask),
        )
        maxima = root / "maxima.csv"
        bundles = root / "bundles.csv"
        subprocess.run(
            [
                str(BACKEND), str(mask_path), str(prefix), "0", "1",
                "tfce", "0.001", "1.0", str(min_size), str(min_cluster_voxels),
                str(maxima), "--threads", "2", "--df-aware",
                "--records-contain-df", "--subjects", str(SUBJECTS),
                "--observed-bundles", str(bundles),
                "--tfce-extent-exponent", str(extent_exponent),
                "--tfce-height-exponent", str(height_exponent),
                "--tfce-z-min", str(self.Z_MIN),
                "--tfce-z-max", str(self.Z_MAX),
                "--tfce-z-step", str(self.Z_STEP),
            ],
            check=True, capture_output=True,
        )
        return pd.read_csv(maxima).iloc[0], pd.read_csv(bundles)

    def reference(
        self, mask: np.ndarray, endpoints: np.ndarray, tstat: np.ndarray,
        degrees_of_freedom: np.ndarray, extent_exponent: float,
        height_exponent: float,
    ) -> np.ndarray:
        return reference_scores(
            mask, endpoints, tstat, degrees_of_freedom,
            tfce_reference.z_grid(self.Z_MIN, self.Z_MAX, self.Z_STEP),
            self.Z_STEP, extent_exponent, height_exponent,
        )

    def test_grid_matches_the_backend_construction(self) -> None:
        grid = tfce_reference.z_grid(2.0, 4.0, 0.25)
        np.testing.assert_allclose(grid, np.arange(2.0, 4.0001, 0.25))
        self.assertEqual(len(tfce_reference.z_grid(2.0, 6.0, 0.1)), 41)

    def test_permutation_maximum_matches_the_reference(self) -> None:
        """The quantity FWER actually consumes, across several (E, H)."""

        z_values = tfce_reference.z_grid(self.Z_MIN, self.Z_MAX, self.Z_STEP)
        for seed, (extent_exponent, height_exponent) in enumerate(
            ((0.5, 2.0), (0.5, 3.0), (1.0, 2.0), (0.25, 1.0), (0.0, 2.0))
        ):
            with self.subTest(E=extent_exponent, H=height_exponent), \
                    tempfile.TemporaryDirectory() as temporary:
                mask, endpoints, tstat, dof = build_fixture(
                    100 + seed, n_hubs=14, z_values=z_values
                )
                maxima, _ = self.run_backend(
                    Path(temporary), mask, endpoints, tstat, dof,
                    extent_exponent, height_exponent,
                )
                expected = self.reference(
                    mask, endpoints, tstat, dof,
                    extent_exponent, height_exponent,
                )
                self.assertGreater(expected.max(), 0.0)
                self.assertAlmostEqual(
                    float(maxima["max_statistic"]), float(expected.max()),
                    delta=1e-6 * max(1.0, float(expected.max())),
                )

    def test_per_bundle_sums_and_maxima_match_when_nothing_is_pruned(self) -> None:
        """Per-bundle sums pin down every individual edge score, not just the top.

        With min_size and min_cluster_voxels at 1 on a dense fixture the
        legibility filters remove nothing, which the retained/threshold edge
        counts assert directly -- so the reported bundles are exactly the
        strict bundles at the lowest height and their masses are sums of the
        per-edge TFCE the oracle computed.
        """

        z_values = tfce_reference.z_grid(self.Z_MIN, self.Z_MAX, self.Z_STEP)
        with tempfile.TemporaryDirectory() as temporary:
            mask, endpoints, tstat, dof = build_fixture(
                7, n_hubs=16, z_values=z_values
            )
            maxima, bundles = self.run_backend(
                Path(temporary), mask, endpoints, tstat, dof, 0.5, 2.0
            )
            self.assertEqual(
                int(maxima["retained_edges"]), int(maxima["threshold_edges"]),
                "fixture was pruned; the comparison below would not be like for like",
            )
            scores = self.reference(mask, endpoints, tstat, dof, 0.5, 2.0)

            coordinates = np.hstack([mask[endpoints[:, 0]], mask[endpoints[:, 1]]])
            expected_sums: list[float] = []
            expected_maxima: list[float] = []
            for keep in (tstat > 0, tstat < 0):
                members = np.nonzero(keep)[0]
                if members.size == 0:
                    continue
                _, labels = tfce_reference._strict_labels(
                    coordinates[members], tstat[members], 1.0
                )
                for label in np.unique(labels):
                    inside = members[labels == label]
                    expected_sums.append(float(scores[inside].sum()))
                    expected_maxima.append(float(scores[inside].max()))

            self.assertEqual(len(bundles), len(expected_sums))
            np.testing.assert_allclose(
                np.sort(bundles["mass"].to_numpy(float)),
                np.sort(expected_sums), rtol=1e-9, atol=1e-9,
            )
            np.testing.assert_allclose(
                np.sort(bundles["statistic"].to_numpy(float)),
                np.sort(expected_maxima), rtol=1e-9, atol=1e-9,
            )

    def test_permutation_maximum_ignores_pruning(self) -> None:
        """The null maximum must not be narrowed by the legibility filters.

        Raising min_size discards bundles from the reported set but must leave
        max_statistic untouched, since a bundle is ranked against a null that
        pruning never shaped.
        """

        z_values = tfce_reference.z_grid(self.Z_MIN, self.Z_MAX, self.Z_STEP)
        with tempfile.TemporaryDirectory() as loose, \
                tempfile.TemporaryDirectory() as strict:
            mask, endpoints, tstat, dof = build_fixture(
                11, n_hubs=12, z_values=z_values
            )
            unpruned, loose_bundles = self.run_backend(
                Path(loose), mask, endpoints, tstat, dof, 0.5, 2.0,
                min_size=1, min_cluster_voxels=1,
            )
            pruned, strict_bundles = self.run_backend(
                Path(strict), mask, endpoints, tstat, dof, 0.5, 2.0,
                min_size=6, min_cluster_voxels=4,
            )
            self.assertLess(len(strict_bundles), len(loose_bundles))
            self.assertAlmostEqual(
                float(unpruned["max_statistic"]),
                float(pruned["max_statistic"]), places=12,
            )

    def test_tfce_requires_per_edge_degrees_of_freedom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask_path = root / "mask.dump"
            np.savetxt(mask_path, slab_mask(4, 4, 2), fmt="%d")
            result = subprocess.run(
                [
                    str(BACKEND), str(mask_path), str(root / "x"), "0", "1",
                    "tfce", "0.001", "1.0", "1", "1", str(root / "m.csv"),
                    "--threads", "1",
                ],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--records-contain-df", result.stderr)

    def test_rejects_a_degenerate_height_grid(self) -> None:
        for bad in (
            ("--tfce-z-step", "0"),
            ("--tfce-z-max", "1.0"),
            ("--tfce-z-min", "-1"),
        ):
            with self.subTest(option=bad), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                mask_path = root / "mask.dump"
                np.savetxt(mask_path, slab_mask(4, 4, 2), fmt="%d")
                result = subprocess.run(
                    [
                        str(BACKEND), str(mask_path), str(root / "x"), "0", "1",
                        "tfce", "0.001", "1.0", "1", "1", str(root / "m.csv"),
                        "--threads", "1", "--df-aware", "--records-contain-df",
                        "--subjects", str(SUBJECTS), *bad,
                    ],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("tfce", result.stderr)


class CalibrationMetricTests(unittest.TestCase):
    def test_a_heavier_tail_scores_worse(self) -> None:
        from tfce_calibration import tail_metrics

        generator = np.random.default_rng(3)
        light = generator.normal(100.0, 5.0, size=5000)
        heavy = np.concatenate([light[:-50], generator.normal(100.0, 5.0, 50) * 40])
        self.assertLess(
            tail_metrics(light)["tail_ratio"], tail_metrics(heavy)["tail_ratio"]
        )

    def test_ratios_are_scale_free(self) -> None:
        from tfce_calibration import tail_metrics

        values = np.random.default_rng(5).lognormal(0.0, 1.0, size=4000)
        for key in ("tail_ratio", "tail_ratio_999", "max_ratio"):
            self.assertAlmostEqual(
                tail_metrics(values)[key], tail_metrics(values * 1000.0)[key],
                places=9,
            )

    def test_an_all_zero_null_is_reported_not_divided_by(self) -> None:
        from tfce_calibration import tail_metrics

        metrics = tail_metrics(np.zeros(100))
        self.assertEqual(metrics["nonzero_fraction"], 0.0)
        self.assertEqual(metrics["tail_ratio"], float("inf"))


class OrchestrationTests(unittest.TestCase):
    RUNNER = HERE / "run_bundle_fwer.py"

    def run_cli(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.RUNNER), "missing_filelist.txt",
             "missing_perms.txt", "out", "--statistic", "tfce", *extra],
            capture_output=True, text=True,
        )

    def test_tfce_rejects_the_threshold_grid(self) -> None:
        result = self.run_cli("--cluster-forming-p-grid", "1e-5", "5e-6")
        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot be combined", result.stderr)

    def test_tfce_rejects_the_python_engine(self) -> None:
        result = self.run_cli(
            "--cluster-forming-p", "1e-3", "--bundle-engine", "python"
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("--bundle-engine cpp", result.stderr)

    def test_storage_threshold_must_reach_the_lowest_height(self) -> None:
        """A p stricter than the grid floor would silently truncate the integral."""

        result = self.run_cli(
            "--cluster-forming-p", "5e-6", "--tfce-z-min", "3.0"
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("silently", result.stderr)

    def test_a_storage_threshold_reaching_the_floor_is_accepted(self) -> None:
        """Same command, loose enough p: must fail later, on the missing files."""

        result = self.run_cli(
            "--cluster-forming-p", "1e-2", "--tfce-z-min", "3.0"
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing_filelist.txt", result.stderr)


class ReferenceUnitTests(unittest.TestCase):
    def test_z_and_p_are_inverse(self) -> None:
        for z in (2.0, 3.5, 4.417173, 5.0):
            recovered = stats.norm.isf(
                tfce_reference.two_sided_p_from_z(z) / 2.0
            )
            self.assertAlmostEqual(recovered, z, places=10)

    def test_height_of_a_single_bundle_is_analytic(self) -> None:
        """One 2-edge bundle of three voxels, integrated by hand."""

        edges = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 0]], dtype=float)
        dof = np.array([40.0, 40.0])
        z_values = tfce_reference.z_grid(2.0, 3.0, 0.5)
        # Both edges sit above every height, so the bundle is 3 voxels at each.
        tstat = tfce_reference.critical_t(dof, 3.5)
        scores = tfce_reference.tfce_scores(
            edges, tstat, dof, z_values=z_values, z_step=0.5,
            extent_exponent=0.5, height_exponent=2.0,
        )
        expected = sum(3 ** 0.5 * z ** 2 * 0.5 for z in z_values)
        np.testing.assert_allclose(scores, [expected, expected], rtol=1e-12)

    def test_rejects_malformed_input(self) -> None:
        with self.assertRaises(ValueError):
            tfce_reference.tfce_scores(
                np.zeros((2, 3)), np.zeros(2), np.zeros(2),
                z_values=np.array([2.0]), z_step=1.0,
            )
        with self.assertRaises(ValueError):
            tfce_reference.z_grid(3.0, 2.0, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
