#!/usr/bin/env python3
"""Average repeated-run binary CCMAT files into one matrix per participant.

The input file list must be group ordered. Participant IDs are extracted from
filenames of the form ``s<id>_<run>.ccmat``. Outputs retain the binary CCMAT
format and contain the arithmetic mean correlation for each connection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np


CCMAT_MAGIC = 0x43434D54
CCMAT_VERSION = 1
HEADER_SIZE = 24
SUBJECT_RE = re.compile(r"^s(\d+)_")


def read_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        raw = stream.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise ValueError(f"short CCMAT header: {path}")
    magic, version, n_voxels, n_elements = struct.unpack("<IIQQ", raw)
    if magic != CCMAT_MAGIC or version != CCMAT_VERSION:
        raise ValueError(f"unsupported CCMAT header: {path}")
    expected = HEADER_SIZE + 4 * n_elements
    if path.stat().st_size != expected:
        raise ValueError(
            f"CCMAT size mismatch for {path}: {path.stat().st_size} != {expected}"
        )
    return int(n_voxels), int(n_elements)


def subject_id(path: Path) -> str:
    match = SUBJECT_RE.match(path.name)
    if not match:
        raise ValueError(f"cannot extract participant ID from {path.name}")
    return match.group(1)


def collect_group(paths: list[Path]) -> OrderedDict[str, list[Path]]:
    subjects: OrderedDict[str, list[Path]] = OrderedDict()
    for path in paths:
        subjects.setdefault(subject_id(path), []).append(path)
    return subjects


def valid_output(path: Path, n_voxels: int, n_elements: int) -> bool:
    try:
        return read_header(path) == (n_voxels, n_elements)
    except (OSError, ValueError):
        return False


def average_subject(
    run_paths: list[Path],
    output_path: Path,
    n_voxels: int,
    n_elements: int,
    chunk_elements: int,
) -> None:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    run_arrays = [
        np.memmap(
            path,
            dtype="<f4",
            mode="r",
            offset=HEADER_SIZE,
            shape=(n_elements,),
        )
        for path in run_paths
    ]

    with temporary.open("wb") as stream:
        stream.write(
            struct.pack(
                "<IIQQ", CCMAT_MAGIC, CCMAT_VERSION, n_voxels, n_elements
            )
        )
        for start in range(0, n_elements, chunk_elements):
            stop = min(start + chunk_elements, n_elements)
            accumulator = np.zeros(stop - start, dtype=np.float64)
            for values in run_arrays:
                accumulator += values[start:stop]
            accumulator /= len(run_arrays)
            accumulator.astype("<f4").tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())

    if not valid_output(temporary, n_voxels, n_elements):
        raise RuntimeError(f"failed output validation: {temporary}")
    temporary.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-list", required=True, type=Path)
    parser.add_argument("--group-a-runs", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-file-list", required=True, type=Path)
    parser.add_argument(
        "--chunk-elements",
        type=int,
        default=8_000_000,
        help="Float elements processed per chunk (default: 8 million).",
    )
    args = parser.parse_args()

    paths = [Path(line.strip()) for line in args.file_list.read_text().splitlines() if line.strip()]
    if not 0 < args.group_a_runs < len(paths):
        raise ValueError("--group-a-runs must split the input file list")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} run matrices; first: {missing[0]}")

    group_a = collect_group(paths[: args.group_a_runs])
    group_b = collect_group(paths[args.group_a_runs :])
    overlap = set(group_a).intersection(group_b)
    if overlap:
        raise ValueError(f"participants cross the group boundary: {sorted(overlap)}")

    n_voxels, n_elements = read_header(paths[0])
    for path in paths[1:]:
        if read_header(path) != (n_voxels, n_elements):
            raise ValueError(f"matrix dimensions differ: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    manifest: dict[str, object] = {
        "method": "arithmetic mean of repeated-run correlation values",
        "source_file_list": str(args.file_list.resolve()),
        "group_a_runs": args.group_a_runs,
        "group_a_subjects": len(group_a),
        "group_b_subjects": len(group_b),
        "n_voxels": n_voxels,
        "n_elements": n_elements,
        "subjects": [],
    }

    all_subjects = [("A", key, value) for key, value in group_a.items()]
    all_subjects += [("B", key, value) for key, value in group_b.items()]
    for index, (group, identifier, run_paths) in enumerate(all_subjects, start=1):
        output = args.output_dir / f"s{identifier}_mean.ccmat"
        output_paths.append(output)
        if valid_output(output, n_voxels, n_elements):
            print(f"[{index}/{len(all_subjects)}] s{identifier}: existing output valid; skipping", flush=True)
        else:
            print(
                f"[{index}/{len(all_subjects)}] s{identifier}: averaging {len(run_paths)} runs",
                flush=True,
            )
            average_subject(
                run_paths, output, n_voxels, n_elements, args.chunk_elements
            )
        manifest["subjects"].append(
            {
                "group": group,
                "subject": identifier,
                "runs": [str(path) for path in run_paths],
                "output": str(output),
            }
        )

    args.output_file_list.write_text("".join(f"{path}\n" for path in output_paths))
    manifest_path = args.output_dir / "aggregation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {len(output_paths)} participant matrices: "
        f"{len(group_a)} group A, {len(group_b)} group B",
        flush=True,
    )
    print(f"File list: {args.output_file_list}", flush=True)


if __name__ == "__main__":
    main()
