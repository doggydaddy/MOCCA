"""Regression tests for the separate bundle-FWER statistic module."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


CUDA_PERM_DIR = Path(__file__).resolve().parent
COFFEE_DIR = CUDA_PERM_DIR.parent / "04_coffee-dac"
for module_dir in (CUDA_PERM_DIR, COFFEE_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from bundle_fwer import compute_bundle_statistics, max_bundle_statistic
from run_bundle_fwer import (
    DF_AWARE_RECORD_DTYPE,
    DF_AWARE_VERSION,
    HEADER,
    MAGIC,
    RECORD_DTYPE,
    VERSION,
    condensed_indices,
    read_sparse_edges,
    records_to_edges,
)


class SparseFormatRegressionTests(unittest.TestCase):
    def test_condensed_index_inverse_matches_numpy(self) -> None:
        for n_voxels in (2, 4, 17, 101):
            expected_rows, expected_columns = np.triu_indices(n_voxels, k=1)
            flat = np.arange(expected_rows.size, dtype=np.uint64)
            rows, columns = condensed_indices(flat, n_voxels)
            np.testing.assert_array_equal(rows, expected_rows)
            np.testing.assert_array_equal(columns, expected_columns)

    def test_sparse_reader_and_coordinate_conversion(self) -> None:
        records = np.array([(0, 3.5), (4, -4.25)], dtype=RECORD_DTYPE)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "fixture.bsp"
            with path.open("wb") as stream:
                stream.write(
                    HEADER.pack(MAGIC, VERSION, 7, 2, 4, 6, 3.0, 0)
                )
                records.tofile(stream)

            header, loaded = read_sparse_edges(path)
            self.assertEqual(header["permutation"], 7)
            coordinates = np.arange(12, dtype=float).reshape(4, 3)
            edges = records_to_edges(loaded, coordinates)
            np.testing.assert_array_equal(edges[0, :6], coordinates[[0, 1]].ravel())
            np.testing.assert_array_equal(edges[1, :6], coordinates[[1, 3]].ravel())
            np.testing.assert_allclose(edges[:, 7], [3.5, -4.25])
            self.assertTrue(np.isnan(edges[:, 6]).all())

    def test_df_aware_sparse_reader_preserves_threshold_excess(self) -> None:
        records = np.array(
            [(0, 3.6, 0.12), (4, -4.1, 0.57)],
            dtype=DF_AWARE_RECORD_DTYPE,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "fixture_v2.bsp"
            with path.open("wb") as stream:
                stream.write(
                    HEADER.pack(MAGIC, DF_AWARE_VERSION, 9, 2, 4, 6, 0.001, 1)
                )
                records.tofile(stream)

            header, loaded = read_sparse_edges(path)
            self.assertEqual(header["version"], DF_AWARE_VERSION)
            self.assertAlmostEqual(header["threshold"], 0.001)
            np.testing.assert_allclose(loaded["excess"], [0.12, 0.57])


class CurrentCacheRegressionTests(unittest.TestCase):
    CASES = (
        "LTLEvsRTLE_runAll_10k_p0005_neg.csv",
        "LTLEvsRTLE_runAll_10k_p0005_pos.csv",
        "controlsVSpatients_runAll_10k_p0001_neg.csv",
        "controlsVSpatients_runAll_10k_p0001_pos.csv",
    )

    def test_bundle_stage_matches_current_v2_caches(self) -> None:
        for filename in self.CASES:
            with self.subTest(filename=filename):
                raw_path = COFFEE_DIR / filename
                stem = raw_path.with_suffix("")
                processed_path = Path(f"{stem}_v2_processed.csv")
                params_path = Path(f"{stem}_v2_params.json")

                manifest = json.loads(params_path.read_text())
                parameters = manifest["parameters"]
                expected = pd.read_csv(processed_path).to_numpy()

                result = compute_bundle_statistics(
                    raw_path,
                    statistic="extent",
                    neighbor_dist=parameters["neighbor_dist"],
                    min_size=parameters["min_network_size"],
                    min_cluster_voxels=parameters["min_cluster_voxels"],
                    strict_bundles=parameters["strict_bundles"],
                    split_signs=False,
                )

                self.assertEqual(
                    result.retained_edge_count,
                    manifest["results"]["retained_edges"],
                )
                self.assertEqual(
                    result.bundle_count,
                    manifest["results"]["bundles"],
                )
                np.testing.assert_allclose(
                    result.edges_bundled[:, :9],
                    expected[:, :9],
                    rtol=0,
                    atol=1e-12,
                )

                bundle_labels = expected[:, 8].astype(int)
                expected_max_extent = max(
                    np.unique(bundle_labels, return_counts=True)[1]
                )
                self.assertEqual(
                    result.max_statistic,
                    float(expected_max_extent),
                )

                expected_max_mass = max(
                    float(np.abs(expected[bundle_labels == label, 7]).sum())
                    for label in np.unique(bundle_labels)
                )
                observed_max_mass = max(
                    summary.mass for summary in result.bundles
                )
                self.assertAlmostEqual(
                    observed_max_mass,
                    expected_max_mass,
                    places=9,
                )
                self.assertEqual(
                    max_bundle_statistic(
                        raw_path,
                        statistic="extent",
                        neighbor_dist=parameters["neighbor_dist"],
                        min_size=parameters["min_network_size"],
                        min_cluster_voxels=parameters["min_cluster_voxels"],
                        strict_bundles=parameters["strict_bundles"],
                        split_signs=False,
                    ),
                    float(expected_max_extent),
                )


if __name__ == "__main__":
    unittest.main()
