"""Small end-to-end regression for the separate CUDA bundle-FWER path."""

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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_bundle_fwer import DF_AWARE_VERSION, DF_STORED_VERSION, read_sparse_edges


BACKEND = HERE / "build/permutationTest_cuda_bundle"
CCMAT_HEADER = struct.Struct("<IIQQ")
CCMAT_MAGIC = 0x43434D54
CUDA_DEVICE = Path("/dev/nvidia0")


def write_ccmat(path: Path, values: np.ndarray, n_voxels: int) -> None:
    values = np.asarray(values, dtype="<f4")
    with path.open("wb") as stream:
        stream.write(CCMAT_HEADER.pack(CCMAT_MAGIC, 1, n_voxels, values.size))
        values.tofile(stream)


@unittest.skipUnless(
    BACKEND.exists() and CUDA_DEVICE.exists(),
    "built CUDA bundle backend and a visible GPU are required",
)
class CudaBundleEndToEndTests(unittest.TestCase):
    def test_two_edge_observed_bundle_and_null_maximum(self) -> None:
        n_voxels = 4
        n_edges = n_voxels * (n_voxels - 1) // 2
        subject_effects = (2.0, 2.2, 0.0, 0.2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            subject_paths = []
            for subject, effect in enumerate(subject_effects):
                values = np.zeros(n_edges, dtype=np.float32)
                values[:2] = effect
                path = temporary / f"subject_{subject}.ccmat"
                write_ccmat(path, values, n_voxels)
                subject_paths.append(path)

            filelist = temporary / "filelist.txt"
            filelist.write_text("".join(f"{path}\n" for path in subject_paths))
            permutations = temporary / "permutations.txt"
            permutations.write_text("0 1\n0 2\n")
            mask = temporary / "mask.dump"
            np.savetxt(
                mask,
                np.array(
                    [[0, 0, 0, 1], [10, 0, 0, 1], [10, 0, 1, 1], [30, 0, 0, 1]]
                ),
                fmt="%d",
            )
            output_dir = temporary / "results"

            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "run_bundle_fwer.py"),
                    str(filelist),
                    str(permutations),
                    str(output_dir),
                    "--mask", str(mask),
                    "--threshold", "5",
                    "--min-size", "2",
                    "--min-cluster-voxels", "1",
                    "--batch-size", "2",
                    "--capacity", "100",
                    "--backend", str(BACKEND),
                ],
                check=True,
            )

            maxima = pd.read_csv(output_dir / "permutation_bundle_maxima.csv")
            self.assertEqual(maxima["permutation"].tolist(), [0, 1])
            self.assertEqual(maxima["bundles"].tolist(), [1, 0])
            self.assertAlmostEqual(maxima.loc[0, "max_statistic"], 18.283772, places=4)
            self.assertEqual(maxima.loc[1, "max_statistic"], 0.0)

            observed = pd.read_csv(output_dir / "observed_bundles_fwer.csv")
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed.loc[0, "edge_count"], 2)
            self.assertEqual(observed.loc[0, "p_fwer"], 0.5)

    def test_df_aware_threshold_and_mass_use_edgewise_welch_df(self) -> None:
        n_voxels = 4
        n_edges = n_voxels * (n_voxels - 1) // 2
        group_a = np.array([2.0, 2.2, 1.8, 2.1, 1.9], dtype=np.float32)
        group_b = np.array([0.0, 0.1, -0.1, 0.2, -0.2, 0.05], dtype=np.float32)
        subject_effects = np.concatenate((group_a, group_b))

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            subject_paths = []
            for subject, effect in enumerate(subject_effects):
                values = np.zeros(n_edges, dtype=np.float32)
                values[:2] = effect
                path = temporary / f"subject_{subject}.ccmat"
                write_ccmat(path, values, n_voxels)
                subject_paths.append(path)

            filelist = temporary / "filelist.txt"
            filelist.write_text("".join(f"{path}\n" for path in subject_paths))
            permutations = temporary / "permutations.txt"
            permutations.write_text("0 1 2 3 4\n0 2 4 6 8\n")
            mask = temporary / "mask.dump"
            np.savetxt(
                mask,
                np.array(
                    [[0, 0, 0, 1], [10, 0, 0, 1], [10, 0, 1, 1], [30, 0, 0, 1]]
                ),
                fmt="%d",
            )
            output_dir = temporary / "results"

            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "run_bundle_fwer.py"),
                    str(filelist),
                    str(permutations),
                    str(output_dir),
                    "--mask", str(mask),
                    "--cluster-forming-p", "0.001",
                    "--min-size", "2",
                    "--min-cluster-voxels", "1",
                    "--batch-size", "2",
                    "--capacity", "100",
                    "--backend", str(BACKEND),
                    "--bundle-method", "bounded",
                    "--keep-sparse",
                ],
                check=True,
            )

            sparse = output_dir / "sparse_work/bundle_perm000000.bsp"
            header, records = read_sparse_edges(sparse)
            self.assertEqual(header["version"], DF_AWARE_VERSION)
            self.assertEqual(header["n_records"], 2)

            a = group_a.astype(np.float64)
            b = group_b.astype(np.float64)
            a_term = np.var(a, ddof=1) / len(a)
            b_term = np.var(b, ddof=1) / len(b)
            degrees_of_freedom = (a_term + b_term) ** 2 / (
                a_term**2 / (len(a) - 1) + b_term**2 / (len(b) - 1)
            )
            expected_t = stats.ttest_ind(a, b, equal_var=False).statistic
            expected_critical = stats.t.ppf(1 - 0.001 / 2, degrees_of_freedom)
            expected_excess = abs(expected_t) - expected_critical

            np.testing.assert_allclose(records["tstat"], expected_t, rtol=2e-6)
            np.testing.assert_allclose(
                records["excess"], expected_excess, rtol=2e-5, atol=2e-5
            )
            observed = pd.read_csv(output_dir / "observed_bundles_fwer.csv")
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed.loc[0, "edge_count"], 2)
            self.assertAlmostEqual(
                observed.loc[0, "mass"], 2 * expected_excess, places=3
            )

            grid_output = temporary / "grid_results"
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "run_bundle_fwer.py"),
                    str(filelist),
                    str(permutations),
                    str(grid_output),
                    "--mask", str(mask),
                    "--cluster-forming-p-grid", "0.001", "0.0001",
                    "--min-size", "2",
                    "--min-cluster-voxels", "1",
                    "--batch-size", "2",
                    "--capacity", "100",
                    "--backend", str(BACKEND),
                    "--keep-sparse",
                ],
                check=True,
            )
            grid_sparse = grid_output / "sparse_work/bundle_perm000000.bsp"
            grid_header, grid_records = read_sparse_edges(grid_sparse)
            self.assertEqual(grid_header["version"], DF_STORED_VERSION)
            np.testing.assert_allclose(
                grid_records["degrees_of_freedom"], degrees_of_freedom,
                rtol=2e-5,
            )
            grid_maxima = pd.read_csv(
                grid_output / "permutation_bundle_maxima_grid.csv"
            )
            self.assertEqual(len(grid_maxima), 4)
            self.assertEqual(
                sorted(grid_maxima["cluster_forming_p"].unique().tolist()),
                [0.0001, 0.001],
            )
            observed_grid = pd.read_csv(
                grid_output / "observed_bundles_grid_fwer.csv"
            )
            self.assertEqual(len(observed_grid), 2)
            np.testing.assert_allclose(observed_grid["p_grid_fwer"], 0.5)


if __name__ == "__main__":
    unittest.main()
