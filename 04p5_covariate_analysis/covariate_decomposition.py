"""Post-hoc decomposition: which covariate removes the group effect, and where.

The 2x2 design in `manuscript/APPENDIX_COVARIATE_ADJUSTED_ANALYSIS.md` isolated
*covariate adjustment* as the change that removes significance, but it adjusted
for age and sex together and so cannot say which. This tool answers that by
refitting nested models on the edges of the previously significant bundles and
comparing the group statistic across them.

**This is descriptive, not inferential, and the distinction is not a
formality.** The bundles were selected because they showed a large group
effect, so every quantity computed inside them is conditioned on that
selection. Group statistics recomputed here cannot be given p-values, and an
interaction tested here is tested in a region chosen for its main effect --
circular by construction. A valid interaction test requires the interaction in
the model of a fresh whole-brain permutation run.

What the comparison *can* support is a relative statement: holding the edge set
fixed, adding age alone versus sex alone versus both shows which term absorbs
the group contrast. The background edge set (suprathreshold edges in other
bundles from the same run) gives that statement a reference, so "the
interaction is large here" is not read off an absolute scale that has none.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent / "02_cudaPerm"
for directory in (HERE, PIPELINE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from design_matrix import build_design, load_covariates, read_filelist

CCMAT_HEADER = 24


def condensed_index(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Row-major upper-triangular index, matching ccmat_io.h."""

    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    return lo * (2 * n - lo - 1) // 2 + hi - lo - 1


def read_edges(paths: list[Path], edge_indices: np.ndarray) -> np.ndarray:
    """Random-access read of selected edges from every participant matrix.

    The matrices are ~7 GB each; a full scan of 68 of them is ~480 GB, but the
    format is a flat float32 array behind a fixed header, so the few thousand
    edges actually wanted can be seeked to directly.
    """

    order = np.argsort(edge_indices)
    ordered = edge_indices[order]
    values = np.empty((len(paths), edge_indices.size), dtype=np.float64)
    for subject, path in enumerate(paths):
        with open(path, "rb") as stream:
            descriptor = stream.fileno()
            # Without this the kernel treats each 4-byte seek as the start of a
            # sequential scan and reads a full window ahead: measured at ~111 GB
            # of device traffic for ~5 MB of wanted data across 68 files.
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_RANDOM)
            except (AttributeError, OSError):
                pass
            buffer = np.empty(ordered.size, dtype=np.float32)
            for position, edge in enumerate(ordered):
                raw = os.pread(descriptor, 4, CCMAT_HEADER + 4 * int(edge))
                buffer[position] = np.frombuffer(raw, dtype="<f4")[0]
        restored = np.empty_like(buffer)
        restored[order] = buffer
        values[subject] = restored
    return values


def hc2_statistics(design: np.ndarray, values: np.ndarray, column: int) -> np.ndarray:
    """HC2-studentized coefficient on `column`, vectorized over edges.

    `values` is (subjects, edges). Equivalent to freedman_lane.statistic_direct
    for one edge; the regression suite pins that equivalence.
    """

    gram_inverse = np.linalg.inv(design.T @ design)
    hat = design @ gram_inverse @ design.T
    leverage = np.diag(hat)
    beta = gram_inverse @ design.T @ values
    residuals = values - design @ beta
    omega = residuals**2 / (1.0 - leverage)[:, None]
    contrast = np.zeros(design.shape[1])
    contrast[column] = 1.0
    weights = gram_inverse @ contrast              # (p,)
    projected = design @ weights                   # (n,)
    variance = (projected[:, None] ** 2 * omega).sum(axis=0)
    return (contrast @ beta) / np.sqrt(variance)


def model_summary(
    design: np.ndarray, values: np.ndarray, column: int, name: str,
    cluster_forming_p: float,
) -> dict[str, object]:
    statistics = hc2_statistics(design, values, column)
    residual_df = values.shape[0] - np.linalg.matrix_rank(design)
    critical = float(stats.t.isf(cluster_forming_p / 2.0, residual_df))
    excess = np.abs(statistics) - critical
    surviving = excess > 0
    return {
        "model": name,
        "residual_df": int(residual_df),
        "critical_t": critical,
        "mean_abs_t": float(np.abs(statistics).mean()),
        "median_abs_t": float(np.median(np.abs(statistics))),
        "max_abs_t": float(np.abs(statistics).max()),
        "edges_surviving": int(surviving.sum()),
        "fraction_surviving": float(surviving.mean()),
        "mass": float(excess[surviving].sum()) if surviving.any() else 0.0,
    }


def parse_args() -> argparse.Namespace:
    project = HERE.parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file_list", type=Path, help="participant matrices, group A first")
    parser.add_argument("edges_bundled", type=Path, help="observed_edges_bundled.csv")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--group-a-subjects", type=int, required=True)
    parser.add_argument(
        "--bundles", type=int, nargs="+", required=True,
        help="bundle labels to decompose (the previously significant ones)",
    )
    parser.add_argument("--mask", type=Path, default=project / "templates/mask3mm.dump")
    parser.add_argument(
        "--covariates", type=Path,
        default=project / "data/share_with_KI/KI_shared_subjects_list.csv",
    )
    parser.add_argument("--cluster-forming-p", type=float, default=5e-6)
    parser.add_argument(
        "--background-edges", type=int, default=10000,
        help="suprathreshold edges sampled from the other bundles as reference",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = read_filelist(args.file_list.resolve())
    covariates = load_covariates(args.covariates.resolve())
    mask = np.loadtxt(args.mask, usecols=(0, 1, 2), dtype=np.int64, ndmin=2)
    n_voxels = mask.shape[0]
    index_of = {tuple(row): index for index, row in enumerate(mask)}

    frame = pd.read_csv(args.edges_bundled)
    def to_edges(subset: pd.DataFrame) -> np.ndarray:
        first = np.array([index_of[tuple(r)] for r in
                          np.rint(subset[["i1", "j1", "k1"]].to_numpy()).astype(int)])
        second = np.array([index_of[tuple(r)] for r in
                           np.rint(subset[["i2", "j2", "k2"]].to_numpy()).astype(int)])
        return condensed_index(first, second, n_voxels)

    target = frame[frame["bundle"].isin(args.bundles)]
    others = frame[~frame["bundle"].isin(args.bundles)]
    generator = np.random.default_rng(args.seed)
    if len(others) > args.background_edges:
        others = others.iloc[
            generator.choice(len(others), args.background_edges, replace=False)
        ]
    print(f"target edges {len(target):,} (bundles {args.bundles}) | "
          f"background edges {len(others):,}", flush=True)

    sets = {"target": to_edges(target), "background": to_edges(others)}
    data = {name: read_edges(paths, edges) for name, edges in sets.items()}
    print("edge values read", flush=True)

    models = {
        "group only": dict(include_age=False, include_sex=False),
        "group + age": dict(include_age=True, include_sex=False),
        "group + sex": dict(include_age=False, include_sex=True),
        "group + age + sex": dict(include_age=True, include_sex=True),
    }
    designs = {}
    for name, options in models.items():
        design = build_design(paths, args.group_a_subjects, covariates, **options)
        designs[name] = (design.matrix, design.group_index, design.column_names)

    rows: list[dict[str, object]] = []
    for set_name, values in data.items():
        for name, (matrix, group_index, _) in designs.items():
            summary = model_summary(
                matrix, values, group_index, name, args.cluster_forming_p
            )
            summary["edge_set"] = set_name
            summary["edges"] = int(values.shape[1])
            rows.append(summary)
    main_effects = pd.DataFrame(rows)

    # Interactions, in the full model, with the background set as the only
    # available yardstick. Circular for the target set; see the module docstring.
    full_matrix, group_index, column_names = designs["group + age + sex"]
    interaction_rows: list[dict[str, object]] = []
    for term in ("age_centered", "sex_female"):
        if term not in column_names:
            continue
        position = column_names.index(term)
        augmented = np.hstack([
            full_matrix,
            (full_matrix[:, group_index] * full_matrix[:, position])[:, None],
        ])
        for set_name, values in data.items():
            statistics = hc2_statistics(augmented, values, augmented.shape[1] - 1)
            residual_df = values.shape[0] - np.linalg.matrix_rank(augmented)
            p_values = stats.t.sf(np.abs(statistics), residual_df) * 2
            interaction_rows.append({
                "interaction": f"group x {term}",
                "edge_set": set_name,
                "edges": int(values.shape[1]),
                "mean_abs_t": float(np.abs(statistics).mean()),
                "median_abs_t": float(np.median(np.abs(statistics))),
                "max_abs_t": float(np.abs(statistics).max()),
                "fraction_p_lt_0.05": float((p_values < 0.05).mean()),
                "fraction_p_lt_0.001": float((p_values < 0.001).mean()),
            })
    interactions = pd.DataFrame(interaction_rows)

    main_effects.to_csv(output_dir / "covariate_decomposition_main.csv", index=False)
    interactions.to_csv(output_dir / "covariate_decomposition_interactions.csv", index=False)
    (output_dir / "covariate_decomposition.json").write_text(json.dumps({
        "file_list": str(args.file_list),
        "edges_bundled": str(args.edges_bundled),
        "bundles": args.bundles,
        "cluster_forming_p": args.cluster_forming_p,
        "target_edges": int(sets["target"].size),
        "background_edges": int(sets["background"].size),
        "caveat": (
            "descriptive only; the bundles were selected on the group effect, "
            "so statistics recomputed inside them are conditioned on that "
            "selection and interaction tests here are circular"
        ),
        "main_effects": main_effects.to_dict(orient="records"),
        "interactions": interactions.to_dict(orient="records"),
    }, indent=2) + "\n")

    pd.set_option("display.width", 200)
    print("\n=== group statistic by model ===")
    print(main_effects[[
        "edge_set", "model", "residual_df", "mean_abs_t", "max_abs_t",
        "edges_surviving", "fraction_surviving", "mass",
    ]].to_string(index=False))
    print("\n=== interactions (descriptive; see caveat) ===")
    print(interactions.to_string(index=False))
    print(f"\nComplete: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
