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
from scipy import stats


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
COFFEE_DIR = PROJECT / "04_coffee-dac"
RESULT_FIXTURES = COFFEE_DIR / "archives/results_archives"
MASK = PROJECT / "templates/mask3mm.dump"
BACKEND = HERE / "build/bundle_fwer_omp"
BOUNDED_BACKEND = HERE / "build/bundle_fwer_bounded_omp"
HEADER = struct.Struct("<IIQQQQfI")
MAGIC = 0x4C444E42
VERSION = 1
RECORDS = np.dtype([("edge_index", "<u8"), ("tstat", "<f4")])
DF_AWARE_VERSION = 2
DF_AWARE_RECORDS = np.dtype(
    [("edge_index", "<u8"), ("tstat", "<f4"), ("excess", "<f4")]
)
DF_STORED_VERSION = 3
DF_STORED_RECORDS = np.dtype(
    [("edge_index", "<u8"), ("tstat", "<f4"), ("degrees_of_freedom", "<f4")]
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


def condensed_index(first: int, second: int, n_voxels: int) -> int:
    row, column = sorted((first, second))
    return row * (2 * n_voxels - row - 1) // 2 + column - row - 1


def write_fixed_sparse(
    path: Path, endpoints: list[tuple[int, int]], tstats: list[float],
    n_voxels: int, reverse_records: bool = False,
) -> None:
    records = np.empty(len(endpoints), dtype=RECORDS)
    records["edge_index"] = [
        condensed_index(first, second, n_voxels) for first, second in endpoints
    ]
    records["tstat"] = tstats
    if reverse_records:
        records = records[::-1]
    with path.open("wb") as stream:
        stream.write(
            HEADER.pack(
                MAGIC, VERSION, 0, len(records), n_voxels,
                n_voxels * (n_voxels - 1) // 2, 0.0, 0,
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
                raw_path = RESULT_FIXTURES / filename
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

    def test_df_stored_v3_can_be_rethresholded_without_cuda_rerun(self) -> None:
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
                [(0, 5.0, 20.0), (1, 5.0, 20.0)], dtype=DF_STORED_RECORDS
            )
            with sparse.open("wb") as stream:
                stream.write(
                    HEADER.pack(MAGIC, DF_STORED_VERSION, 0, 2, 4, 6, 0.001, 3)
                )
                records.tofile(stream)

            for probability, expected_edges in ((0.001, 2), (0.0001, 2)):
                maxima = temporary_path / f"maxima_{probability}.csv"
                bundles = temporary_path / f"bundles_{probability}.csv"
                subprocess.run(
                    [
                        str(BACKEND), str(mask), str(prefix), "0", "1",
                        "mass", str(probability), "1", "1", "1", str(maxima),
                        "--threads", "1", "--df-aware", "--records-contain-df",
                        "--subjects", "22", "--observed-bundles", str(bundles),
                    ],
                    check=True,
                )
                result = pd.read_csv(maxima).iloc[0]
                critical = stats.t.ppf(1 - probability / 2, 20)
                expected_mass = sum(max(abs(value) - critical, 0) for value in (5, 5))
                self.assertEqual(result["threshold_edges"], 2)
                self.assertEqual(result["retained_edges"], expected_edges)
                self.assertAlmostEqual(result["max_statistic"], expected_mass, places=5)


@unittest.skipUnless(
    BACKEND.exists() and BOUNDED_BACKEND.exists(),
    "strict and bounded C++ bundle backends must be built",
)
class BoundedBundleTests(unittest.TestCase):
    def run_backend(
        self, backend: Path, temporary: Path, mask: Path,
        endpoints: list[tuple[int, int]], tstats: list[float],
        suffix: str, reverse_records: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        prefix = temporary / f"fixture_{suffix}"
        sparse = Path(f"{prefix}_perm000000.bsp")
        write_fixed_sparse(
            sparse, endpoints, tstats,
            len(np.loadtxt(mask, usecols=(0,), ndmin=1)), reverse_records,
        )
        maxima = temporary / f"maxima_{suffix}.csv"
        edges = temporary / f"edges_{suffix}.csv"
        subprocess.run(
            [
                str(backend), str(mask), str(prefix), "0", "1", "extent",
                "0", "1", "1", "1", str(maxima), "--threads", "1",
                "--observed-edges", str(edges),
            ],
            check=True,
        )
        return pd.read_csv(maxima), pd.read_csv(edges)

    def test_chain_cannot_percolate_beyond_endpoint_diameter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            coordinates = np.array(
                [(x, 0, 0, 1) for x in range(5)]
                + [(10 + x, 0, 0, 1) for x in range(5)]
            )
            mask = temporary / "mask.dump"
            np.savetxt(mask, coordinates, fmt="%d")
            endpoints = []
            for index in range(4):
                endpoints.extend(
                    [(index, 5 + index), (index, 5 + index + 1)]
                )
            endpoints.append((4, 9))
            tstats = [10.0 - 0.1 * index for index in range(len(endpoints))]

            strict_maxima, _ = self.run_backend(
                BACKEND, temporary, mask, endpoints, tstats, "strict"
            )
            bounded_maxima, bounded_edges = self.run_backend(
                BOUNDED_BACKEND, temporary, mask, endpoints, tstats, "bounded"
            )
            self.assertEqual(strict_maxima.loc[0, "bundles"], 1)
            self.assertGreater(bounded_maxima.loc[0, "bundles"], 1)

            for _, bundle in bounded_edges.groupby("bundle"):
                values = bundle[["i1", "j1", "k1", "i2", "j2", "k2"]].to_numpy(int)
                for first in values:
                    for second in values:
                        direct = max(
                            np.max(np.abs(first[:3] - second[:3])),
                            np.max(np.abs(first[3:] - second[3:])),
                        )
                        swapped = max(
                            np.max(np.abs(first[:3] - second[3:])),
                            np.max(np.abs(first[3:] - second[:3])),
                        )
                        self.assertLessEqual(min(direct, swapped), 2)

            _, reversed_edges = self.run_backend(
                BOUNDED_BACKEND, temporary, mask, endpoints, tstats,
                "bounded_reversed", reverse_records=True,
            )
            pd.testing.assert_frame_equal(bounded_edges, reversed_edges)

    def test_dense_endpoint_cubes_have_finite_bundle_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            first_cube = [
                (x, y, z, 1)
                for x in range(5) for y in range(5) for z in range(5)
            ]
            second_cube = [
                (20 + x, y, z, 1)
                for x in range(5) for y in range(5) for z in range(5)
            ]
            coordinates = np.array(first_cube + second_cube)
            mask = temporary / "mask.dump"
            np.savetxt(mask, coordinates, fmt="%d")
            endpoints = [
                (first, 125 + second)
                for first in range(125) for second in range(125)
            ]
            tstats = [10.0] * len(endpoints)
            _, edges = self.run_backend(
                BOUNDED_BACKEND, temporary, mask, endpoints, tstats, "dense"
            )
            sizes = edges.groupby("bundle").size()
            self.assertGreater(len(sizes), 1)
            self.assertLessEqual(int(sizes.max()), 27 * 27)


if __name__ == "__main__":
    unittest.main()
