#!/usr/bin/env python3
"""End-to-end regression: the CUDA Freedman--Lane backend against the oracle.

`freedman_lane.py` is the reference implementation. This suite runs the real
CUDA backend on a small synthetic fixture and requires that it selects exactly
the same suprathreshold edges and reproduces the same statistics, so the GPU
path is never trusted on its own.

Skipped when no GPU or no built backend is available.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import freedman_lane as fl
from run_bundle_fwer import read_sparse_edges

BACKEND = HERE / "build/permutationTest_cuda_bundle"
CUDA_DEVICE = Path("/dev/nvidia0")
CCMAT_HEADER = struct.Struct("<IIQQ")
CCMAT_MAGIC = 0x43434D54


def write_ccmat(path: Path, values: np.ndarray, n_voxels: int) -> None:
    stored = np.asarray(values, dtype="<f4")
    with path.open("wb") as stream:
        stream.write(CCMAT_HEADER.pack(CCMAT_MAGIC, 1, n_voxels, stored.size))
        stored.tofile(stream)


@unittest.skipUnless(
    BACKEND.exists() and CUDA_DEVICE.exists(),
    "built CUDA bundle backend and a visible GPU are required",
)
class CudaFreedmanLaneTests(unittest.TestCase):
    N_VOXELS = 20
    N_SUBJECTS = 12
    N_GROUP_A = 5
    CLUSTER_P = 0.05

    def build_fixture(self, directory: Path, seed: int = 0, n_permutations: int = 6):
        generator = np.random.default_rng(seed)
        n_edges = self.N_VOXELS * (self.N_VOXELS - 1) // 2
        group = np.concatenate(
            [np.ones(self.N_GROUP_A), np.zeros(self.N_SUBJECTS - self.N_GROUP_A)]
        )
        age = generator.normal(35, 10, self.N_SUBJECTS)
        sex = generator.integers(0, 2, self.N_SUBJECTS).astype(float)
        design = np.column_stack(
            [np.ones(self.N_SUBJECTS), age - age.mean(), sex, group]
        )
        plan = fl.build_plan(design, 3)

        # heteroscedastic noise, genuine covariate signal, a planted effect
        data = (
            generator.normal(size=(self.N_SUBJECTS, n_edges))
            * np.where(group > 0, 1.5, 0.6)[:, None]
            + design[:, 1][:, None] * 0.03
            + design[:, 2][:, None] * 0.3
        ).astype(np.float32)
        data[:, :30] += (group[:, None] * 1.5).astype(np.float32)

        paths = []
        for subject in range(self.N_SUBJECTS):
            path = directory / f"s{100 + subject}_x.ccmat"
            write_ccmat(path, data[subject], self.N_VOXELS)
            paths.append(path)
        (directory / "filelist.txt").write_text(
            "".join(f"{path}\n" for path in paths)
        )

        permutations = np.array(
            [np.arange(self.N_SUBJECTS)]
            + [generator.permutation(self.N_SUBJECTS) for _ in range(n_permutations)]
        )
        np.savetxt(directory / "perm.txt", permutations, fmt="%d")
        fl.write_cuda_plan(plan, directory / "plan.flp")
        return plan, data, permutations

    def run_backend(self, directory: Path, count: int, *extra: str) -> None:
        subprocess.run(
            [
                str(BACKEND),
                str(directory / "filelist.txt"),
                str(directory / "perm.txt"),
                str(directory / "out"),
                "0",
                "--cluster-forming-p", str(self.CLUSTER_P),
                "--start-perm", "0", "--count", str(count),
                "--capacity", "100000", "--store-df",
                "--freedman-lane", str(directory / "plan.flp"),
                *extra,
            ],
            check=True,
            capture_output=True,
        )

    def test_cuda_matches_the_python_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            plan, data, permutations = self.build_fixture(directory)
            self.run_backend(directory, len(permutations))

            expected = fl.statistics_projector(
                plan, data.astype(np.float64), permutations
            )
            degrees = fl.effective_degrees_of_freedom(plan)
            critical = float(stats.t.ppf(1 - self.CLUSTER_P / 2, degrees))

            total = 0
            worst = 0.0
            for index in range(len(permutations)):
                _, records = read_sparse_edges(
                    directory / f"out_perm{index:06d}.bsp"
                )
                produced = {
                    int(record["edge_index"]): float(record["tstat"])
                    for record in records
                }
                oracle = {
                    edge: expected[index, edge]
                    for edge in range(expected.shape[1])
                    if abs(expected[index, edge]) >= critical
                }
                self.assertEqual(
                    set(produced), set(oracle),
                    msg=f"permutation {index}: suprathreshold edge sets differ",
                )
                for edge, value in oracle.items():
                    worst = max(worst, abs(produced[edge] - value))
                total += len(produced)

            self.assertGreater(total, 20, "fixture produced too few edges to test")
            self.assertLess(worst, 1e-5)

    def test_stored_degrees_of_freedom_are_the_fixed_residual_df(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            plan, _, permutations = self.build_fixture(directory, seed=1)
            self.run_backend(directory, len(permutations))
            degrees = fl.effective_degrees_of_freedom(plan)
            self.assertEqual(degrees, self.N_SUBJECTS - 4)
            for index in range(len(permutations)):
                _, records = read_sparse_edges(
                    directory / f"out_perm{index:06d}.bsp"
                )
                for record in records:
                    self.assertAlmostEqual(
                        float(record["degrees_of_freedom"]), degrees, places=3
                    )

    def test_observed_row_reproduces_the_unpermuted_statistic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            plan, data, _ = self.build_fixture(directory, seed=2)
            self.run_backend(directory, 1)
            _, records = read_sparse_edges(directory / "out_perm000000.bsp")
            for record in records:
                edge = int(record["edge_index"])
                self.assertAlmostEqual(
                    float(record["tstat"]),
                    fl.statistic_direct(plan, data[:, edge].astype(np.float64)),
                    places=4,
                )

    def test_group_membership_permutation_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.build_fixture(directory, seed=3)
            # overwrite with group-A membership rows of the wrong width
            np.savetxt(
                directory / "perm.txt",
                np.array([[0, 1, 2, 3, 4], [1, 3, 5, 7, 9]]),
                fmt="%d",
            )
            result = subprocess.run(
                [
                    str(BACKEND),
                    str(directory / "filelist.txt"),
                    str(directory / "perm.txt"),
                    str(directory / "out"), "0",
                    "--cluster-forming-p", str(self.CLUSTER_P),
                    "--start-perm", "0", "--count", "2",
                    "--capacity", "1000", "--store-df",
                    "--freedman-lane", str(directory / "plan.flp"),
                ],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("group-membership file cannot be used", result.stderr)

    def test_plan_requires_a_cluster_forming_p(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.build_fixture(directory, seed=4)
            result = subprocess.run(
                [
                    str(BACKEND),
                    str(directory / "filelist.txt"),
                    str(directory / "perm.txt"),
                    str(directory / "out"), "3.5",
                    "--start-perm", "0", "--count", "1",
                    "--freedman-lane", str(directory / "plan.flp"),
                ],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires --cluster-forming-p", result.stderr)

    def test_participant_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.build_fixture(directory, seed=5)
            # a plan built for a different participant count
            other = np.column_stack(
                [
                    np.ones(8),
                    np.arange(8.0) - 3.5,
                    np.array([0.0, 1, 0, 1, 0, 1, 0, 1]),
                    np.r_[np.ones(3), np.zeros(5)],
                ]
            )
            fl.write_cuda_plan(fl.build_plan(other, 3), directory / "wrong.flp")
            result = subprocess.run(
                [
                    str(BACKEND),
                    str(directory / "filelist.txt"),
                    str(directory / "perm.txt"),
                    str(directory / "out"), "0",
                    "--cluster-forming-p", "0.05",
                    "--start-perm", "0", "--count", "1",
                    "--freedman-lane", str(directory / "wrong.flp"),
                ],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("participants but the filelist", result.stderr)


@unittest.skipUnless(
    BACKEND.exists() and CUDA_DEVICE.exists(),
    "built CUDA bundle backend and a visible GPU are required",
)
class AdjustedPipelineTests(unittest.TestCase):
    """run_bundle_fwer.py end to end with the adjusted statistic."""

    def test_planted_effect_survives_bundle_fwer_with_held_out_nulls(self) -> None:
        n_voxels, n_subjects, n_group_a = 40, 16, 7
        n_edges = n_voxels * (n_voxels - 1) // 2
        generator = np.random.default_rng(7)
        group = np.concatenate([np.ones(n_group_a), np.zeros(n_subjects - n_group_a)])
        design = np.column_stack(
            [
                np.ones(n_subjects),
                generator.normal(0, 10, n_subjects),
                generator.integers(0, 2, n_subjects).astype(float),
                group,
            ]
        )
        plan = fl.build_plan(design, 3)

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            data = (
                generator.normal(size=(n_subjects, n_edges))
                * np.where(group > 0, 1.2, 0.6)[:, None]
                + design[:, 1][:, None] * 0.02
            ).astype(np.float32)
            rows, columns = np.triu_indices(n_voxels, 1)
            blob = (rows < 6) & (columns < 6)
            data[:, blob] += (group[:, None] * 2.5).astype(np.float32)

            paths = []
            for subject in range(n_subjects):
                path = directory / f"s{100 + subject}_p.ccmat"
                write_ccmat(path, data[subject], n_voxels)
                paths.append(path)
            (directory / "filelist.txt").write_text(
                "".join(f"{path}\n" for path in paths)
            )
            permutations = np.array(
                [np.arange(n_subjects)]
                + [generator.permutation(n_subjects) for _ in range(41)]
            )
            np.savetxt(directory / "perm.txt", permutations, fmt="%d")
            fl.write_cuda_plan(plan, directory / "plan.flp")
            coordinates = np.array(
                [[i, j, k, 1] for i in range(4) for j in range(5) for k in range(2)]
            )[:n_voxels]
            np.savetxt(directory / "mask.dump", coordinates, fmt="%d")

            subprocess.run(
                [
                    sys.executable, str(HERE / "run_bundle_fwer.py"),
                    str(directory / "filelist.txt"),
                    str(directory / "perm.txt"),
                    str(directory / "results"),
                    "--mask", str(directory / "mask.dump"),
                    "--cluster-forming-p", "0.01",
                    "--freedman-lane-plan", str(directory / "plan.flp"),
                    "--calibration-permutations", "1",
                    "--calibration-start-row", "1",
                    "--inference-permutations", "40",
                    "--inference-start-row", "2",
                    "--min-size", "2", "--min-cluster-voxels", "1",
                    "--neighbor-dist", "1.0", "--batch-size", "50",
                    "--capacity", "100000", "--bundle-threads", "4",
                ],
                check=True, capture_output=True,
            )

            import pandas as pd

            bundles = pd.read_csv(directory / "results/observed_bundles_fwer.csv")
            self.assertGreater(len(bundles), 0)
            best = bundles.sort_values("statistic", ascending=False).iloc[0]
            self.assertLess(float(best["p_fwer"]), 0.05)
            self.assertEqual(int(best["inference_exceedances"]), 0)

            config = json.loads(
                (directory / "results/bundle_fwer_config.json").read_text()
            )
            self.assertEqual(config["edge_statistic"], "hc2_freedman_lane_adjusted")
            self.assertEqual(config["permutation_representation"], "full-index")
            self.assertIsNotNone(config["freedman_lane_plan_sha256"])

            results = json.loads(
                (directory / "results/bundle_fwer_results.json").read_text()
            )
            self.assertEqual(results["p_fwer_denominator"], 41)
            self.assertEqual(results["inference_maxima_used"], 40)

            maxima = pd.read_csv(directory / "results/permutation_bundle_maxima.csv")
            computed = set(maxima["permutation"].astype(int))
            self.assertNotIn(1, computed)  # the calibration row is held out
            self.assertEqual(computed, {0} | set(range(2, 42)))

    def test_group_membership_file_is_rejected_by_the_orchestrator(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / "filelist.txt").write_text("/nonexistent.ccmat\n" * 12)
            np.savetxt(
                directory / "perm.txt",
                np.array([[0, 1, 2], [1, 3, 5], [2, 4, 6]]),
                fmt="%d",
            )
            design = np.column_stack(
                [
                    np.ones(12),
                    np.arange(12.0) - 5.5,
                    np.array([0.0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                    np.r_[np.ones(5), np.zeros(7)],
                ]
            )
            fl.write_cuda_plan(fl.build_plan(design, 3), directory / "plan.flp")
            result = subprocess.run(
                [
                    sys.executable, str(HERE / "run_bundle_fwer.py"),
                    str(directory / "filelist.txt"),
                    str(directory / "perm.txt"),
                    str(directory / "results"),
                    "--cluster-forming-p", "0.01",
                    "--freedman-lane-plan", str(directory / "plan.flp"),
                    "--calibration-permutations", "1",
                    "--calibration-start-row", "1",
                    "--inference-permutations", "1",
                    "--inference-start-row", "2",
                ],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
