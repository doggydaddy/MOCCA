"""Regression of the optimized C++ bundler against the Python oracle."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
COFFEE_DIR = PROJECT / "04_coffee-dac"
MASK = PROJECT / "templates/mask3mm.dump"
BACKEND = HERE / "build/bundle_fwer_omp"
HEADER = struct.Struct("<IIQQQQfI")
MAGIC = 0x4C444E42
VERSION = 1
RECORDS = np.dtype([("edge_index", "<u8"), ("tstat", "<f4")])
DF_AWARE_VERSION = 2
DF_AWARE_RECORDS = np.dtype(
    [("edge_index", "<u8"), ("tstat", "<f4"), ("excess", "<f4")]
)

for module_dir in (HERE, COFFEE_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from bundle_fwer import compute_bundle_statistics


def csv_to_sparse(csv_path: Path, sparse_path: Path) -> None:
    coordinates = np.loadtxt(MASK, usecols=(0, 1, 2), dtype=np.int64)
    coordinate_index = {tuple(coord): index for index, coord in enumerate(coordinates)}
    raw = pd.read_csv(csv_path).to_numpy()
    first = np.array(
        [coordinate_index[tuple(np.rint(row[:3]).astype(int))] for row in raw],
        dtype=np.int64,
    )
    second = np.array(
        [coordinate_index[tuple(np.rint(row[3:6]).astype(int))] for row in raw],
        dtype=np.int64,
    )
    rows = np.minimum(first, second)
    columns = np.maximum(first, second)
    n_voxels = len(coordinates)
    flat = rows * (2 * n_voxels - rows - 1) // 2 + columns - rows - 1
    records = np.empty(len(raw), dtype=RECORDS)
    records["edge_index"] = flat
    records["tstat"] = raw[:, 7]
    records.sort(order="edge_index")
    with sparse_path.open("wb") as stream:
        stream.write(
            HEADER.pack(
                MAGIC,
                VERSION,
                0,
                len(records),
                n_voxels,
                n_voxels * (n_voxels - 1) // 2,
                0.0,
                0,
            )
        )
        records.tofile(stream)


@unittest.skipUnless(BACKEND.exists(), "optimized bundle backend is not built")
class CppOracleEquivalenceTests(unittest.TestCase):
    CASES = (
        "LTLEvsRTLE_runAll_10k_p0005_neg.csv",
        "LTLEvsRTLE_runAll_10k_p0005_pos.csv",
        "controlsVSpatients_runAll_10k_p0001_neg.csv",
        "controlsVSpatients_runAll_10k_p0001_pos.csv",
    )

    def test_all_current_bundle_caches(self) -> None:
        for filename in self.CASES:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                raw_path = COFFEE_DIR / filename
                params = json.loads(
                    Path(f"{raw_path.with_suffix('')}_v2_params.json").read_text()
                )["parameters"]
                oracle = compute_bundle_statistics(
                    raw_path,
                    statistic="extent",
                    neighbor_dist=params["neighbor_dist"],
                    min_size=params["min_network_size"],
                    min_cluster_voxels=params["min_cluster_voxels"],
                    strict_bundles=True,
                    split_signs=False,
                )

                temporary_path = Path(temporary)
                prefix = temporary_path / "fixture"
                sparse = Path(f"{prefix}_perm000000.bsp")
                maxima = temporary_path / "maxima.csv"
                cpp_edges_path = temporary_path / "edges.csv"
                cpp_bundles_path = temporary_path / "bundles.csv"
                csv_to_sparse(raw_path, sparse)
                subprocess.run(
                    [
                        str(BACKEND), str(MASK), str(prefix), "0", "1",
                        "extent", "0", str(params["neighbor_dist"]),
                        str(params["min_network_size"]),
                        str(params["min_cluster_voxels"]), str(maxima),
                        "--threads", "2",
                        "--observed-edges", str(cpp_edges_path),
                        "--observed-bundles", str(cpp_bundles_path),
                    ],
                    check=True,
                )

                cpp_maxima = pd.read_csv(maxima).iloc[0]
                cpp_edges = pd.read_csv(cpp_edges_path).to_numpy()
                cpp_bundles = pd.read_csv(cpp_bundles_path)
                self.assertEqual(cpp_maxima["threshold_edges"], oracle.threshold_edge_count)
                self.assertEqual(cpp_maxima["retained_edges"], oracle.retained_edge_count)
                self.assertEqual(cpp_maxima["bundles"], oracle.bundle_count)
                self.assertEqual(cpp_maxima["max_statistic"], oracle.max_statistic)
                np.testing.assert_allclose(
                    cpp_edges[:, :6], oracle.edges_bundled[:, :6], rtol=0, atol=0
                )
                np.testing.assert_allclose(
                    cpp_edges[:, 7:], oracle.edges_bundled[:, 7:], rtol=0, atol=1e-6
                )
                np.testing.assert_allclose(
                    cpp_bundles["mass"].max(),
                    max(bundle.mass for bundle in oracle.bundles),
                    rtol=2e-9,
                    atol=2e-5,
                )

    def test_df_aware_v2_mass_sums_per_edge_excess(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            mask = temporary_path / "mask.dump"
            np.savetxt(
                mask,
                np.array(
                    [[0, 0, 0, 1], [10, 0, 0, 1], [10, 0, 1, 1], [30, 0, 0, 1]]
                ),
                fmt="%d",
            )
            prefix = temporary_path / "fixture"
            sparse = Path(f"{prefix}_perm000000.bsp")
            records = np.array(
                [(0, 3.6, 0.1), (1, 4.2, 0.2)], dtype=DF_AWARE_RECORDS
            )
            with sparse.open("wb") as stream:
                stream.write(
                    HEADER.pack(MAGIC, DF_AWARE_VERSION, 0, 2, 4, 6, 0.001, 1)
                )
                records.tofile(stream)

            maxima = temporary_path / "maxima.csv"
            bundles = temporary_path / "bundles.csv"
            subprocess.run(
                [
                    str(BACKEND), str(mask), str(prefix), "0", "1",
                    "mass", "0.001", "1", "2", "1", str(maxima),
                    "--threads", "1", "--df-aware",
                    "--observed-bundles", str(bundles),
                ],
                check=True,
            )
            result = pd.read_csv(maxima).iloc[0]
            self.assertEqual(result["bundles"], 1)
            self.assertAlmostEqual(result["max_statistic"], 0.3, places=6)
            self.assertAlmostEqual(pd.read_csv(bundles).iloc[0]["mass"], 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
