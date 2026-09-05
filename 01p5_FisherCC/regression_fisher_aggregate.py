#!/usr/bin/env python3
"""Regression tests for the 01p5_FisherCC participant-aggregation stage.

These cover the validation requirements recorded in
``analysis_notes/ANALYSIS_DECISIONS.md`` for this stage:

1. transform and aggregation against a small float64 reference;
2. invariance to processing chunk size and run-file order;
3. ``raw-equal`` reproduces the current participant matrices;
4. participant ordering and group boundaries verified independently;
5. exact/near -1 and 1, non-finite inputs, one-run participants, and unequal
   run counts.

Items 6 (distributional comparison between modes) and 7 (rerunning threshold
calibration and inference) are production activities, not unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np


FISHER_DIR = Path(__file__).resolve().parent
CUDA_PERM_DIR = FISHER_DIR.parent / "02_cudaPerm"
for module_dir in (FISHER_DIR, CUDA_PERM_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import fisher_aggregate_ccmat as fisher
from fisher_aggregate_ccmat import (
    CCMAT_MAGIC,
    CCMAT_VERSION,
    CLAMP_R,
    HEADER_SIZE,
    HEADER_STRUCT,
    collect_group,
    main,
    read_header,
    sidecar_path,
)


# ── fixtures ────────────────────────────────────────────────────────────────
def write_ccmat(path: Path, values: np.ndarray, n_voxels: int) -> np.ndarray:
    """Write a binary CCMAT file and return the float32 values as stored."""
    stored = np.asarray(values, dtype="<f4")
    with path.open("wb") as stream:
        stream.write(
            HEADER_STRUCT.pack(CCMAT_MAGIC, CCMAT_VERSION, n_voxels, stored.size)
        )
        stored.tofile(stream)
    return stored


def read_ccmat(path: Path) -> np.ndarray:
    n_voxels, n_elements = read_header(path)
    return np.fromfile(path, dtype="<f4", offset=HEADER_SIZE, count=n_elements)


def reference_mean(
    runs: list[np.ndarray], weights: list[float], transform: str
) -> np.ndarray:
    """Independent float64 reference for one participant's aggregate."""
    total = np.zeros(runs[0].size, dtype=np.float64)
    for values, weight in zip(runs, weights):
        block = np.asarray(values, dtype=np.float64)
        if transform == "atanh":
            block = np.arctanh(np.clip(block, -CLAMP_R, CLAMP_R))
        total += weight * block
    return total / float(np.sum(weights))


class FixtureMixin:
    """Builds a two-group synthetic dataset with unequal run counts."""

    N_VOXELS = 6
    N_ELEMENTS = N_VOXELS * (N_VOXELS - 1) // 2  # 15

    # group A: two participants (3 runs, 1 run); group B: two (2 runs, 4 runs)
    LAYOUT = {
        "A": {"101": 3, "102": 1},
        "B": {"201": 2, "202": 4},
    }

    def build(self, directory: Path, seed: int = 7) -> dict:
        rng = np.random.default_rng(seed)
        runs_dir = directory / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        stored: dict[str, list[np.ndarray]] = {}
        group_paths: dict[str, list[Path]] = {"A": [], "B": []}
        for group, members in self.LAYOUT.items():
            for identifier, n_runs in members.items():
                stored[identifier] = []
                for run in range(1, n_runs + 1):
                    values = rng.uniform(-0.95, 0.95, size=self.N_ELEMENTS)
                    path = runs_dir / f"s{identifier}_{run}.ccmat"
                    stored[identifier].append(
                        write_ccmat(path, values, self.N_VOXELS)
                    )
                    group_paths[group].append(path)

        file_list = directory / "runs.txt"
        ordered = group_paths["A"] + group_paths["B"]
        file_list.write_text("".join(f"{path}\n" for path in ordered))
        return {
            "file_list": file_list,
            "group_a_runs": len(group_paths["A"]),
            "stored": stored,
            "paths": ordered,
        }

    def run_stage(self, directory: Path, fixture: dict, *extra: str) -> tuple[Path, Path]:
        output_dir = directory / f"out{len(list(directory.glob('out*')))}"
        output_list = output_dir / "participants.txt"
        exit_code = main(
            [
                "--file-list",
                str(fixture["file_list"]),
                "--group-a-runs",
                str(fixture["group_a_runs"]),
                "--output-dir",
                str(output_dir),
                "--output-file-list",
                str(output_list),
                *extra,
            ]
        )
        self.assertEqual(exit_code, 0)
        return output_dir, output_list


# ── 1. transform and aggregation against a float64 reference ────────────────
class TransformReferenceTests(FixtureMixin, unittest.TestCase):
    def test_fisher_equal_matches_float64_reference(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            output_dir, _ = self.run_stage(directory, fixture, "--mode", "fisher-equal")

            for identifier, runs in fixture["stored"].items():
                expected = reference_mean(runs, [1.0] * len(runs), "atanh")
                produced = read_ccmat(output_dir / f"s{identifier}_fisherz.ccmat")
                np.testing.assert_array_equal(
                    produced, expected.astype("<f4"), err_msg=identifier
                )

    def test_raw_equal_matches_float64_reference(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            output_dir, _ = self.run_stage(directory, fixture, "--mode", "raw-equal")

            for identifier, runs in fixture["stored"].items():
                expected = reference_mean(runs, [1.0] * len(runs), "identity")
                produced = read_ccmat(output_dir / f"s{identifier}_rawmean.ccmat")
                np.testing.assert_array_equal(produced, expected.astype("<f4"))

    def test_fisher_duration_uses_supplied_weights(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)

            weights = {}
            lines = ["run,weight,n_timepoints"]
            for index, path in enumerate(fixture["paths"]):
                weight = 1.0 + 0.5 * index
                weights[path.stem] = weight
                lines.append(f"{path.name},{weight},{120 + index}")
            table = directory / "weights.csv"
            table.write_text("\n".join(lines) + "\n")

            output_dir, _ = self.run_stage(
                directory,
                fixture,
                "--mode",
                "fisher-duration",
                "--weight-table",
                str(table),
                "--timepoint-column",
                "n_timepoints",
            )

            for identifier, runs in fixture["stored"].items():
                run_weights = [
                    weights[f"s{identifier}_{run}"]
                    for run in range(1, len(runs) + 1)
                ]
                expected = reference_mean(runs, run_weights, "atanh")
                produced = read_ccmat(output_dir / f"s{identifier}_fisherz_w.ccmat")
                np.testing.assert_array_equal(produced, expected.astype("<f4"))

    def test_fisher_z_is_not_back_transformed(self) -> None:
        """A Fisher z aggregate must differ from the raw-r mean it came from."""
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            fisher_dir, _ = self.run_stage(directory, fixture, "--mode", "fisher-equal")
            raw_dir, _ = self.run_stage(directory, fixture, "--mode", "raw-equal")

            z_values = read_ccmat(fisher_dir / "s101_fisherz.ccmat")
            r_values = read_ccmat(raw_dir / "s101_rawmean.ccmat")
            self.assertFalse(np.allclose(z_values, r_values))
            # tanh of the z mean is the interpretable correlation-scale value
            self.assertTrue(np.all(np.abs(np.tanh(z_values)) < 1.0))

            manifest = json.loads(
                (fisher_dir / "fisher_aggregation_manifest_fisher-equal.json").read_text()
            )
            self.assertEqual(manifest["output_scale"], "fisher_z")
            self.assertFalse(manifest["back_transformed"])

    def test_single_run_participant_is_the_transformed_run(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            output_dir, _ = self.run_stage(directory, fixture, "--mode", "fisher-equal")

            # s102 has exactly one run
            expected = np.arctanh(
                np.asarray(fixture["stored"]["102"][0], dtype=np.float64)
            )
            produced = read_ccmat(output_dir / "s102_fisherz.ccmat")
            np.testing.assert_array_equal(produced, expected.astype("<f4"))


# ── 2. invariance to chunk size and run-file order ──────────────────────────
class InvarianceTests(FixtureMixin, unittest.TestCase):
    def test_chunk_size_does_not_change_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            digests = []
            for chunk in ("1", "2", "7", "1000000"):
                output_dir, _ = self.run_stage(
                    directory, fixture, "--mode", "fisher-equal",
                    "--chunk-elements", chunk,
                )
                digests.append(
                    [
                        read_ccmat(output_dir / f"s{identifier}_fisherz.ccmat").tobytes()
                        for identifier in fixture["stored"]
                    ]
                )
            for other in digests[1:]:
                self.assertEqual(digests[0], other)

    def test_run_order_within_participant_does_not_change_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            output_dir, _ = self.run_stage(directory, fixture, "--mode", "fisher-equal")
            baseline = read_ccmat(output_dir / "s202_fisherz.ccmat")

            # Reverse the run order of s202 inside its group block.
            paths = list(fixture["paths"])
            block = [index for index, p in enumerate(paths) if p.name.startswith("s202_")]
            reordered = list(paths)
            for target, source in zip(block, reversed(block)):
                reordered[target] = paths[source]
            shuffled_list = directory / "runs_shuffled.txt"
            shuffled_list.write_text("".join(f"{path}\n" for path in reordered))

            shuffled = dict(fixture)
            shuffled["file_list"] = shuffled_list
            output_dir_b, _ = self.run_stage(
                directory, shuffled, "--mode", "fisher-equal"
            )
            np.testing.assert_array_equal(
                baseline, read_ccmat(output_dir_b / "s202_fisherz.ccmat")
            )

    def test_runs_are_canonically_ordered_within_participant(self) -> None:
        paths = [Path(f"/x/s5_{run}.ccmat") for run in (10, 2, 1)]
        grouped = collect_group(paths)
        self.assertEqual(
            [path.name for path in grouped["5"]],
            ["s5_1.ccmat", "s5_2.ccmat", "s5_10.ccmat"],
        )


# ── 3. raw-equal reproduces the existing participant matrices ───────────────
class LegacyEquivalenceTests(FixtureMixin, unittest.TestCase):
    def test_raw_equal_matches_average_ccmat_runs(self) -> None:
        import average_ccmat_runs

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)

            new_dir, _ = self.run_stage(directory, fixture, "--mode", "raw-equal")

            legacy_dir = directory / "legacy"
            legacy_list = directory / "legacy.txt"
            argv = sys.argv
            sys.argv = [
                "average_ccmat_runs.py",
                "--file-list", str(fixture["file_list"]),
                "--group-a-runs", str(fixture["group_a_runs"]),
                "--output-dir", str(legacy_dir),
                "--output-file-list", str(legacy_list),
            ]
            try:
                average_ccmat_runs.main()
            finally:
                sys.argv = argv

            for identifier in fixture["stored"]:
                legacy = read_ccmat(legacy_dir / f"s{identifier}_mean.ccmat")
                produced = read_ccmat(new_dir / f"s{identifier}_rawmean.ccmat")
                np.testing.assert_array_equal(produced, legacy, err_msg=identifier)


# ── 4. participant ordering and group boundaries ────────────────────────────
class OrderingTests(FixtureMixin, unittest.TestCase):
    def test_output_list_preserves_group_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            output_dir, output_list = self.run_stage(
                directory, fixture, "--mode", "fisher-equal"
            )

            written = [
                Path(line) for line in output_list.read_text().split() if line
            ]
            self.assertEqual(
                [path.name for path in written],
                [
                    "s101_fisherz.ccmat",
                    "s102_fisherz.ccmat",
                    "s201_fisherz.ccmat",
                    "s202_fisherz.ccmat",
                ],
            )

            manifest = json.loads(
                (output_dir / "fisher_aggregation_manifest_fisher-equal.json").read_text()
            )
            self.assertEqual(manifest["group_a_subjects"], 2)
            self.assertEqual(manifest["group_b_subjects"], 2)
            groups = [entry["group"] for entry in manifest["subjects"]]
            self.assertEqual(groups, ["A", "A", "B", "B"])
            # unequal run counts survive into the manifest
            self.assertEqual(
                [entry["n_runs"] for entry in manifest["subjects"]], [3, 1, 2, 4]
            )

    def test_participant_spanning_both_groups_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            with self.assertRaisesRegex(ValueError, "cross the group boundary"):
                self.run_stage(
                    directory,
                    {**fixture, "group_a_runs": 2},  # splits s101's runs
                    "--mode",
                    "fisher-equal",
                )


# ── 5. numerical edge cases ─────────────────────────────────────────────────
class EdgeCaseTests(unittest.TestCase):
    N_VOXELS = 4
    N_ELEMENTS = 6

    def _two_subject_fixture(self, directory: Path, values: np.ndarray) -> dict:
        runs_dir = directory / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        a = runs_dir / "s1_1.ccmat"
        b = runs_dir / "s2_1.ccmat"
        write_ccmat(a, values, self.N_VOXELS)
        write_ccmat(b, np.zeros(self.N_ELEMENTS), self.N_VOXELS)
        file_list = directory / "runs.txt"
        file_list.write_text(f"{a}\n{b}\n")
        return {"file_list": file_list, "group_a_runs": 1}

    def _run(self, directory: Path, fixture: dict, *extra: str) -> Path:
        output_dir = directory / "out"
        main(
            [
                "--file-list", str(fixture["file_list"]),
                "--group-a-runs", str(fixture["group_a_runs"]),
                "--output-dir", str(output_dir),
                "--output-file-list", str(output_dir / "participants.txt"),
                *extra,
            ]
        )
        return output_dir

    def test_exact_unit_correlations_are_clipped_and_counted(self) -> None:
        values = np.array([1.0, -1.0, 0.5, 0.0, 1.0, -0.25])
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self._two_subject_fixture(directory, values)
            output_dir = self._run(directory, fixture, "--mode", "fisher-equal")

            produced = read_ccmat(output_dir / "s1_fisherz.ccmat")
            self.assertTrue(np.all(np.isfinite(produced)))
            self.assertAlmostEqual(float(produced[0]), fisher.MAX_ABS_Z, places=3)
            self.assertAlmostEqual(float(produced[1]), -fisher.MAX_ABS_Z, places=3)
            # untouched values still go through the plain transform
            self.assertAlmostEqual(float(produced[2]), float(np.arctanh(0.5)), places=6)

            sidecar = json.loads(
                sidecar_path(output_dir / "s1_fisherz.ccmat").read_text()
            )
            clipping = sidecar["clipping"]
            self.assertEqual(clipping["clipped_total"], 3)
            self.assertEqual(clipping["clipped_at_unit"], 3)
            self.assertEqual(clipping["clipped_beyond_unit"], 0)
            self.assertEqual(clipping["clipped_min_input"], -1.0)
            self.assertEqual(clipping["clipped_max_input"], 1.0)

    def test_near_unit_correlations_are_not_clipped(self) -> None:
        near = float(np.float32(0.9999999))
        values = np.array([near, -near, 0.1, 0.2, 0.3, 0.4])
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self._two_subject_fixture(directory, values)
            output_dir = self._run(directory, fixture, "--mode", "fisher-equal")

            sidecar = json.loads(
                sidecar_path(output_dir / "s1_fisherz.ccmat").read_text()
            )
            self.assertEqual(sidecar["clipping"]["clipped_total"], 0)
            produced = read_ccmat(output_dir / "s1_fisherz.ccmat")
            self.assertTrue(np.all(np.isfinite(produced)))

    def test_out_of_range_values_are_counted_separately(self) -> None:
        values = np.array([1.5, -2.0, 0.5, 0.0, 1.0, -0.25])
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self._two_subject_fixture(directory, values)
            output_dir = self._run(directory, fixture, "--mode", "fisher-equal")

            clipping = json.loads(
                sidecar_path(output_dir / "s1_fisherz.ccmat").read_text()
            )["clipping"]
            self.assertEqual(clipping["clipped_at_unit"], 1)
            self.assertEqual(clipping["clipped_beyond_unit"], 2)
            self.assertEqual(clipping["clipped_min_input"], -2.0)
            self.assertEqual(clipping["clipped_max_input"], 1.5)

    def test_non_finite_input_is_rejected_not_coerced(self) -> None:
        for bad in (np.nan, np.inf, -np.inf):
            values = np.array([0.1, bad, 0.2, 0.3, 0.4, 0.5])
            with tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                fixture = self._two_subject_fixture(directory, values)
                with self.assertRaisesRegex(ValueError, "non-finite correlation"):
                    self._run(directory, fixture, "--mode", "fisher-equal")

    def test_non_finite_input_is_rejected_in_raw_mode_too(self) -> None:
        values = np.array([0.1, np.nan, 0.2, 0.3, 0.4, 0.5])
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self._two_subject_fixture(directory, values)
            with self.assertRaisesRegex(ValueError, "non-finite correlation"):
                self._run(directory, fixture, "--mode", "raw-equal")


# ── interface and resume safety ─────────────────────────────────────────────
class InterfaceTests(FixtureMixin, unittest.TestCase):
    def test_duration_mode_requires_a_weight_table(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            with self.assertRaisesRegex(ValueError, "requires --weight-table"):
                self.run_stage(directory, fixture, "--mode", "fisher-duration")

    def test_equal_weight_modes_reject_a_weight_table(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            table = directory / "weights.csv"
            table.write_text("run,weight\ns101_1.ccmat,1.0\n")
            with self.assertRaisesRegex(ValueError, "equal-weight mode"):
                self.run_stage(
                    directory, fixture, "--mode", "fisher-equal",
                    "--weight-table", str(table),
                )

    def test_missing_weight_entry_fails_before_writing_anything(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            table = directory / "weights.csv"
            table.write_text("run,weight\ns101_1.ccmat,1.0\n")
            output_dir = directory / "out0"
            with self.assertRaisesRegex(KeyError, "s101_2"):
                main(
                    [
                        "--file-list", str(fixture["file_list"]),
                        "--group-a-runs", str(fixture["group_a_runs"]),
                        "--output-dir", str(output_dir),
                        "--output-file-list", str(output_dir / "participants.txt"),
                        "--mode", "fisher-duration",
                        "--weight-table", str(table),
                    ]
                )
            written = list(output_dir.glob("*.ccmat")) if output_dir.exists() else []
            self.assertEqual(written, [])

    def test_non_positive_weight_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            table = Path(raw) / "weights.csv"
            table.write_text("run,weight\ns101_1.ccmat,0\n")
            with self.assertRaisesRegex(ValueError, "non-positive"):
                fisher.load_weight_table(table, "weight", None)

    def test_resume_skips_only_matching_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            output_dir, _ = self.run_stage(directory, fixture, "--mode", "fisher-equal")
            target = output_dir / "s101_fisherz.ccmat"
            expected = read_ccmat(target)

            # A stale output whose sidecar no longer matches must be rebuilt.
            sidecar = json.loads(sidecar_path(target).read_text())
            sidecar["runs"] = ["s101_1.ccmat"]
            sidecar_path(target).write_text(json.dumps(sidecar))
            write_ccmat(target, np.zeros(self.N_ELEMENTS), self.N_VOXELS)

            main(
                [
                    "--file-list", str(fixture["file_list"]),
                    "--group-a-runs", str(fixture["group_a_runs"]),
                    "--output-dir", str(output_dir),
                    "--output-file-list", str(output_dir / "participants.txt"),
                    "--mode", "fisher-equal",
                ]
            )
            np.testing.assert_array_equal(read_ccmat(target), expected)

    def test_output_container_is_readable_by_the_downstream_header_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            output_dir, _ = self.run_stage(directory, fixture, "--mode", "fisher-equal")
            path = output_dir / "s101_fisherz.ccmat"
            with path.open("rb") as stream:
                magic, version, n_voxels, n_elements = struct.unpack(
                    "<IIQQ", stream.read(HEADER_SIZE)
                )
            self.assertEqual(magic, CCMAT_MAGIC)
            self.assertEqual(version, CCMAT_VERSION)
            self.assertEqual(n_voxels, self.N_VOXELS)
            self.assertEqual(n_elements, self.N_ELEMENTS)
            self.assertEqual(path.stat().st_size, HEADER_SIZE + 4 * n_elements)

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            output_dir = directory / "dry"
            main(
                [
                    "--file-list", str(fixture["file_list"]),
                    "--group-a-runs", str(fixture["group_a_runs"]),
                    "--output-dir", str(output_dir),
                    "--output-file-list", str(output_dir / "participants.txt"),
                    "--mode", "fisher-equal",
                    "--dry-run",
                ]
            )
            self.assertFalse(output_dir.exists())

    def test_sha256_checksum_policy_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            output_dir, _ = self.run_stage(
                directory, fixture, "--mode", "fisher-equal", "--checksum", "sha256"
            )
            manifest = json.loads(
                (output_dir / "fisher_aggregation_manifest_fisher-equal.json").read_text()
            )
            self.assertEqual(manifest["checksum_policy"], "sha256")
            first = manifest["subjects"][0]["runs"][0]
            self.assertEqual(len(first["sha256"]), 64)
            self.assertEqual(len(manifest["subjects"][0]["output"]["sha256"]), 64)

    def test_free_space_guard_blocks_and_ignores_completed_outputs(self) -> None:
        each = HEADER_SIZE + 4 * self.N_ELEMENTS
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            output_dir, _ = self.run_stage(directory, fixture, "--mode", "fisher-equal")
            outputs = sorted(output_dir.glob("*.ccmat"))
            self.assertEqual(len(outputs), 4)

            # Nothing pending once every output is present at full size.
            record = fisher.check_free_space(
                output_dir, outputs, self.N_ELEMENTS, allow_low=False
            )
            self.assertEqual(record["pending_outputs"], 0)
            self.assertEqual(record["required_bytes"], 0)

            # A missing output is counted again.
            outputs[0].unlink()
            record = fisher.check_free_space(
                output_dir, outputs, self.N_ELEMENTS, allow_low=False
            )
            self.assertEqual(record["pending_outputs"], 1)
            self.assertEqual(record["required_bytes"], each)

            manifest = json.loads(
                (output_dir / "fisher_aggregation_manifest_fisher-equal.json").read_text()
            )
            self.assertEqual(manifest["disk_space_at_start"]["pending_outputs"], 4)

    def test_free_space_guard_raises_when_output_cannot_fit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            with self.assertRaisesRegex(RuntimeError, "--allow-low-space"):
                fisher.check_free_space(
                    directory, [directory / "huge.ccmat"], 2**60, allow_low=False
                )
            # the override lets an operator proceed anyway
            fisher.check_free_space(
                directory, [directory / "huge.ccmat"], 2**60, allow_low=True
            )

    def test_mismatched_matrix_dimensions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fixture = self.build(directory)
            write_ccmat(
                directory / "runs" / "s202_4.ccmat", np.zeros(21), 7
            )
            with self.assertRaisesRegex(ValueError, "matrix dimensions differ"):
                self.run_stage(directory, fixture, "--mode", "fisher-equal")


if __name__ == "__main__":
    unittest.main()
