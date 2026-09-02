#!/usr/bin/env python3
"""Build and validate the participant-level design matrix for the adjusted model.

Per ``manuscript/ANALYSIS_DECISIONS.md`` (2026-09-02, "covariate-adjusted
control--TLE analysis"), the minimal primary model for participant ``i`` and
edge ``e`` is::

    r_ie = beta_0e + beta_Ge * group_i
                   + beta_Ae * centered_age_i
                   + beta_Se * sex_i + error_ie

with ``beta_Ge`` the contrast of interest.  The exact coding and centering of
every design column must be written to the run manifest, and covariates must
not be selected according to their sample p-values -- so this module takes the
model as a declaration, records precisely what it built, and refuses to guess.

Covariate audit encoded here
----------------------------
- **Age**: centered continuous covariate in the primary adjusted model.
- **Sex**: categorical covariate in the primary adjusted model.
- **Handedness**: *not* an automatic primary-model covariate.  All six
  left-handed participants are in the TLE group, so group and handedness
  cannot be reliably distinguished.  ``--restrict-handedness R`` runs the
  preferred sensitivity analysis on the 62 right-handed participants instead.
  Passing handedness as a primary covariate requires ``--allow-confounded``
  and is recorded as such.
- **Run count**: a measurement-precision/data-availability issue, not
  automatically a biological confound.  Available as an explicit opt-in
  covariate only; never added by default.
- **TLE laterality and diagnosis subtype**: nested within the patient group,
  so they are never available as nuisance covariates for this contrast.
- **Motion**: no participant- or run-level motion summary was delivered with
  this dataset.  There is nothing to include, and nothing is inferred.  See
  the decision log's data-provenance limitation.

Group coding follows the rest of the pipeline: the first ``--group-a-subjects``
filelist entries are group A and get ``group = 1``, so a positive ``beta_G``
means group A greater -- the same sign convention as the Welch pipeline's
``t = mean(A) - mean(B)``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from permutation_rows import sha256_file


SUBJECT_RE = re.compile(r"^s(\d+)_")

# Column names are part of the recorded model definition; do not rename these
# without recording a decision, because manifests refer to them by name.
GROUP_COLUMN = "group"
INTERCEPT_COLUMN = "intercept"

SEX_REFERENCE = "m"
SEX_COLUMN = "sex_female"
HAND_REFERENCE = "R"
HAND_COLUMN = "hand_left"


@dataclass
class Design:
    """A built design matrix plus everything needed to reproduce it."""

    matrix: np.ndarray  # (n_subjects, n_columns), float64
    column_names: list[str]
    subject_ids: list[str]
    group_labels: list[str]
    group_a_subjects: int
    nuisance_columns: list[str]
    coding: dict[str, object] = field(default_factory=dict)
    excluded: list[dict[str, object]] = field(default_factory=list)

    @property
    def n_subjects(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def group_index(self) -> int:
        return self.column_names.index(GROUP_COLUMN)

    @property
    def nuisance_indices(self) -> list[int]:
        return [self.column_names.index(name) for name in self.nuisance_columns]

    def contrast(self) -> np.ndarray:
        """The contrast vector selecting beta_G."""
        contrast = np.zeros(len(self.column_names), dtype=np.float64)
        contrast[self.group_index] = 1.0
        return contrast

    def to_manifest(self) -> dict[str, object]:
        matrix = self.matrix
        _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
        rank = int(np.linalg.matrix_rank(matrix))
        return {
            "model": (
                "r_ie = " + " + ".join(
                    f"beta_{name}e * {name}" for name in self.column_names
                ) + " + error_ie"
            ),
            "contrast_of_interest": GROUP_COLUMN,
            "contrast_vector": self.contrast().tolist(),
            "sign_convention": (
                "group = 1 for the first group_a_subjects filelist entries, so "
                "a positive beta_group means group A greater, matching the "
                "Welch pipeline's t = mean(A) - mean(B)"
            ),
            "column_names": list(self.column_names),
            "nuisance_columns": list(self.nuisance_columns),
            "coding": self.coding,
            "n_subjects": self.n_subjects,
            "group_a_subjects": self.group_a_subjects,
            "group_b_subjects": self.n_subjects - self.group_a_subjects,
            "subject_ids": list(self.subject_ids),
            "group_labels": list(self.group_labels),
            "design_rank": rank,
            "design_full_rank": rank == matrix.shape[1],
            "design_condition_number": float(
                singular_values[0] / singular_values[-1]
            ),
            "excluded_subjects": self.excluded,
        }


def subject_id(path: Path) -> str:
    match = SUBJECT_RE.match(path.name)
    if not match:
        raise ValueError(f"cannot extract participant ID from {path.name}")
    return match.group(1)


def read_filelist(path: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_covariates(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    for required in ("serial", "tag", "gender", "Hand", "age"):
        if required not in table.columns:
            raise ValueError(
                f"{path} has no '{required}' column (found {sorted(table.columns)})"
            )
    if table["serial"].duplicated().any():
        repeated = table.loc[table["serial"].duplicated(), "serial"].tolist()
        raise ValueError(f"{path} lists participants more than once: {repeated}")
    table["serial"] = table["serial"].astype(str)
    return table.set_index("serial", drop=False)


def build_design(
    filelist: list[Path],
    group_a_subjects: int,
    covariates: pd.DataFrame,
    *,
    include_age: bool = True,
    include_sex: bool = True,
    include_handedness: bool = False,
    include_run_count: bool = False,
    run_counts: dict[str, int] | None = None,
    restrict_handedness: str | None = None,
    allow_confounded: bool = False,
) -> Design:
    if not 0 < group_a_subjects < len(filelist):
        raise ValueError("--group-a-subjects must split the participant file list")

    identifiers = [subject_id(path) for path in filelist]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("the participant file list repeats a participant")

    missing = [key for key in identifiers if key not in covariates.index]
    if missing:
        raise ValueError(
            f"{len(missing)} participant(s) have no covariate row; first: s{missing[0]}"
        )

    labels = ["A"] * group_a_subjects + ["B"] * (len(filelist) - group_a_subjects)
    frame = covariates.loc[identifiers].copy()
    frame["group_label"] = labels
    frame["group"] = [1.0] * group_a_subjects + [0.0] * (
        len(filelist) - group_a_subjects
    )

    for column in ("age", "gender", "Hand"):
        if frame[column].isna().any():
            blank = frame.loc[frame[column].isna(), "serial"].tolist()
            raise ValueError(f"missing '{column}' for participant(s) {blank}")

    # ── optional sensitivity restriction ────────────────────────────────────
    excluded: list[dict[str, object]] = []
    if restrict_handedness is not None:
        keep = frame["Hand"] == restrict_handedness
        excluded = [
            {
                "subject": row.serial,
                "group": row.group_label,
                "reason": f"Hand != {restrict_handedness}",
                "hand": row.Hand,
            }
            for row in frame.loc[~keep].itertuples()
        ]
        frame = frame.loc[keep]
        if frame.empty:
            raise ValueError(f"no participant has Hand == {restrict_handedness!r}")
        group_a_subjects = int((frame["group_label"] == "A").sum())
        if group_a_subjects in (0, len(frame)):
            raise ValueError(
                f"restricting to Hand == {restrict_handedness!r} leaves only one group"
            )
        if list(frame["group_label"]) != ["A"] * group_a_subjects + ["B"] * (
            len(frame) - group_a_subjects
        ):
            raise RuntimeError("group ordering broke during handedness restriction")

    # ── columns ─────────────────────────────────────────────────────────────
    columns: list[tuple[str, np.ndarray]] = [
        (INTERCEPT_COLUMN, np.ones(len(frame), dtype=np.float64))
    ]
    coding: dict[str, object] = {
        INTERCEPT_COLUMN: {"type": "constant", "value": 1.0},
    }
    nuisance = [INTERCEPT_COLUMN]

    if include_age:
        age = frame["age"].to_numpy(dtype=np.float64)
        mean_age = float(age.mean())
        columns.append(("age_centered", age - mean_age))
        coding["age_centered"] = {
            "type": "continuous",
            "source_column": "age",
            "centering": "mean of the analysis sample",
            "mean_subtracted": mean_age,
            "units": "years",
        }
        nuisance.append("age_centered")

    if include_sex:
        gender = frame["gender"].astype(str).str.strip().str.lower()
        levels = sorted(gender.unique())
        if not set(levels).issubset({"m", "f"}):
            raise ValueError(f"unexpected 'gender' values: {levels}")
        columns.append((SEX_COLUMN, (gender == "f").to_numpy(dtype=np.float64)))
        coding[SEX_COLUMN] = {
            "type": "categorical indicator",
            "source_column": "gender",
            "reference_level": SEX_REFERENCE,
            "indicator_level": "f",
            "encoding": "1 = female, 0 = male",
            "counts": gender.value_counts().to_dict(),
        }
        nuisance.append(SEX_COLUMN)

    if include_handedness:
        if not allow_confounded:
            raise ValueError(
                "Handedness is not an automatic primary-model covariate: all "
                "left-handed participants are in one group, so group and "
                "handedness cannot be reliably distinguished. Use "
                "--restrict-handedness R for the preferred sensitivity "
                "analysis, or pass --allow-confounded to override deliberately."
            )
        hand = frame["Hand"].astype(str).str.strip().str.upper()
        columns.append((HAND_COLUMN, (hand == "L").to_numpy(dtype=np.float64)))
        coding[HAND_COLUMN] = {
            "type": "categorical indicator",
            "source_column": "Hand",
            "reference_level": HAND_REFERENCE,
            "indicator_level": "L",
            "encoding": "1 = left-handed, 0 = right-handed",
            "counts": hand.value_counts().to_dict(),
            "warning": (
                "included over the documented objection that handedness is "
                "confounded with group in this sample"
            ),
        }
        nuisance.append(HAND_COLUMN)

    if include_run_count:
        if run_counts is None:
            raise ValueError("--include-run-count requires --run-file-list")
        counts = np.array(
            [float(run_counts[key]) for key in frame["serial"]], dtype=np.float64
        )
        mean_count = float(counts.mean())
        columns.append(("run_count_centered", counts - mean_count))
        coding["run_count_centered"] = {
            "type": "continuous",
            "source": "count of run matrices per participant in --run-file-list",
            "centering": "mean of the analysis sample",
            "mean_subtracted": mean_count,
            "note": (
                "data-availability / measurement-precision covariate, included "
                "only on explicit request; not a biological confound"
            ),
        }
        nuisance.append("run_count_centered")

    columns.append((GROUP_COLUMN, frame["group"].to_numpy(dtype=np.float64)))
    coding[GROUP_COLUMN] = {
        "type": "categorical indicator",
        "reference_level": "B",
        "indicator_level": "A",
        "encoding": "1 = group A (leading filelist entries), 0 = group B",
    }

    names = [name for name, _ in columns]
    matrix = np.column_stack([values for _, values in columns])

    rank = int(np.linalg.matrix_rank(matrix))
    if rank < matrix.shape[1]:
        raise ValueError(
            f"design matrix is rank deficient (rank {rank} < {matrix.shape[1]} "
            f"columns: {names}). A covariate is collinear with the group term "
            "or with another covariate; the group effect is not identifiable."
        )

    return Design(
        matrix=matrix,
        column_names=names,
        subject_ids=[str(value) for value in frame["serial"]],
        group_labels=list(frame["group_label"]),
        group_a_subjects=group_a_subjects,
        nuisance_columns=nuisance,
        coding=coding,
        excluded=excluded,
    )


def count_runs(run_file_list: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in read_filelist(run_file_list):
        key = subject_id(path)
        counts[key] = counts.get(key, 0) + 1
    return counts


def save_design(design: Design, output_dir: Path, extra: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "design.npz",
        matrix=design.matrix,
        column_names=np.array(design.column_names),
        subject_ids=np.array(design.subject_ids),
        group_labels=np.array(design.group_labels),
        contrast=design.contrast(),
        nuisance_indices=np.array(design.nuisance_indices, dtype=np.int64),
        group_index=np.array(design.group_index, dtype=np.int64),
    )
    frame = pd.DataFrame(design.matrix, columns=design.column_names)
    frame.insert(0, "group_label", design.group_labels)
    frame.insert(0, "subject", design.subject_ids)
    frame.to_csv(output_dir / "design_matrix.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "02_cudaPerm/design_matrix.py",
        "command_line": sys.argv,
        **extra,
        **design.to_manifest(),
    }
    (output_dir / "design_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--file-list", required=True, type=Path,
        help="participant-level connectivity matrix list, group A first",
    )
    parser.add_argument(
        "--group-a-subjects", required=True, type=int,
        help="number of leading file-list entries belonging to group A",
    )
    parser.add_argument(
        "--covariates", type=Path,
        default=project / "data/share_with_KI/KI_shared_subjects_list.csv",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--no-age", action="store_true", help="drop the centered age covariate"
    )
    parser.add_argument(
        "--no-sex", action="store_true", help="drop the sex covariate"
    )
    parser.add_argument(
        "--include-handedness", action="store_true",
        help="add handedness as a primary covariate (requires --allow-confounded)",
    )
    parser.add_argument(
        "--include-run-count", action="store_true",
        help="add centered run count as a covariate (requires --run-file-list)",
    )
    parser.add_argument(
        "--run-file-list", type=Path,
        help="run-level file list, used only to derive per-participant run counts",
    )
    parser.add_argument(
        "--restrict-handedness", default=None, metavar="LEVEL",
        help="sensitivity analysis restricted to participants with this Hand "
             "value (e.g. R for the 62 right-handed participants)",
    )
    parser.add_argument(
        "--allow-confounded", action="store_true",
        help="permit a covariate documented as confounded with group",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    filelist = read_filelist(args.file_list)
    covariates = load_covariates(args.covariates)
    run_counts = count_runs(args.run_file_list) if args.run_file_list else None

    design = build_design(
        filelist,
        args.group_a_subjects,
        covariates,
        include_age=not args.no_age,
        include_sex=not args.no_sex,
        include_handedness=args.include_handedness,
        include_run_count=args.include_run_count,
        run_counts=run_counts,
        restrict_handedness=args.restrict_handedness,
        allow_confounded=args.allow_confounded,
    )

    save_design(
        design,
        args.output_dir,
        {
            "source_file_list": str(args.file_list.resolve()),
            "source_covariates": str(args.covariates.resolve()),
            "source_covariates_sha256": sha256_file(args.covariates),
            "run_file_list": (
                str(args.run_file_list.resolve()) if args.run_file_list else None
            ),
            "restrict_handedness": args.restrict_handedness,
            "motion_covariate": None,
            "motion_covariate_note": (
                "No participant- or run-level motion summary was delivered with "
                "this dataset; none is inferred. Reported as a data-provenance "
                "limitation."
            ),
        },
    )

    print(f"design: {design.n_subjects} participants x {len(design.column_names)} columns")
    print(f"columns: {', '.join(design.column_names)}")
    print(
        f"group A = {design.group_a_subjects}, "
        f"group B = {design.n_subjects - design.group_a_subjects}"
    )
    if design.excluded:
        print(f"excluded {len(design.excluded)} participant(s) by --restrict-handedness")
    manifest = design.to_manifest()
    print(f"rank {manifest['design_rank']}/{len(design.column_names)}, "
          f"condition number {manifest['design_condition_number']:.2f}")
    print(f"written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
