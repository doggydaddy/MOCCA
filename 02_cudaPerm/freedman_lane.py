#!/usr/bin/env python3
"""Covariate-adjusted Freedman--Lane permutation inference for the group term.

Implements the nuisance-aware permutation scheme required by
``analysis_notes/ANALYSIS_DECISIONS.md`` (2026-09-02, "covariate-adjusted
control--TLE analysis"):

1. Let ``Z = [intercept, centered age, sex]`` and fit the reduced nuisance
   model for each edge.
2. Retain its fitted values and residuals.
3. For every null draw, apply one full n-participant permutation to the
   residuals and add the nuisance fitted values back.
4. Fit/evaluate the full model ``X = [Z, group]`` and calculate the
   studentized group statistic.
5. Use the same participant permutation for every edge.

Methodological basis: Winkler AM, Ridgway GR, Webster MA, Smith SM, Nichols TE.
Permutation inference for the general linear model. *NeuroImage* 2014;
92:381--397.  https://doi.org/10.1016/j.neuroimage.2014.01.060

The studentized statistic
-------------------------
The existing pipeline deliberately uses Welch's unequal-variance t, so the
adjusted statistic must stay heteroscedasticity-robust rather than silently
becoming a pooled-variance statistic.  **HC2** is the right generalization,
because for a two-group design with no covariates the HC2-studentized group
coefficient equals Welch's t *exactly* (verified to floating-point equality in
``regression_freedman_lane.py``).  Adding covariates then changes the model
without changing the statistic's variance assumptions.

    t = c'beta / sqrt( c' (X'X)^-1 X' diag(e_i^2/(1-h_ii)) X (X'X)^-1 c )

Why this is cheap enough to run 1.78 billion times
--------------------------------------------------
Written naively, every (edge, permutation) pair needs its own regression.  The
algebra collapses to two dense matrix products instead.  With
``M_Z = I - Z Z^+``, ``M_X = I - X X^+``, ``H_Z = I - M_Z``, and the
Frisch--Waugh contrast vector ``a = M_Z g / (g' M_Z g)`` so that
``beta_G = a'y``:

* ``a'Z = 0`` and ``M_X H_Z = 0``, so for a Freedman--Lane draw
  ``y* = P M_Z y + H_Z y`` both halves of the statistic depend on the data
  only through the **nuisance residuals** ``u = M_Z y``, which are computed
  once per edge and reused by every permutation:

      numerator   = a' P u                     (a length-n dot product)
      residuals   = M_X y* = M_X P u
      denominator^2 = sum_i w_i (M_X P u)_i^2  = u' (P' K P) u

  with permutation-independent ``w_i = a_i^2 / (1 - h_ii)`` and
  ``K = M_X' diag(w) M_X``.

* The quadratic form is a dot product against the packed upper triangle of
  ``u u'``, so per edge-chunk the whole permutation set is two GEMMs:

      numerators   = W  @ U        W: (n_perm, n)          U:  (n, n_edges)
      denominators = KP @ UU       KP: (n_perm, n(n+1)/2)  UU: (n(n+1)/2, n_edges)

``S_p = P' K P`` is only a symmetric relabelling of one fixed ``K``, so the
whole permutation set costs one 94 MB float32 table at n=68, B=10001.  ``K``
is positive semidefinite and the packed form is well conditioned, so float32
is safe: measured over 2000 edges x 51 permutations the absolute error on
``t`` stays below 2.2e-6, and the relative error below 1e-6 wherever
``|t| > 1``.  Relative error does grow near ``t = 0`` -- the numerator
cancels there -- but those edges are nowhere near any cluster-forming
threshold, and the absolute bound is what governs threshold crossing.

This module is the reference implementation and the precomputation stage.  It
is exact, not fast: it is the oracle a CUDA backend regresses against, and it
emits the ``W``/``KP`` tables that such a backend would consume.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from permutation_rows import (
    add_partition_arguments,
    partition_from_args,
    sha256_file,
    validate_permutation_file,
)


HC_KINDS = ("HC0", "HC2", "HC3")
DEFAULT_HC = "HC2"


@dataclass(frozen=True)
class FreedmanLanePlan:
    """Permutation-independent algebra for one design matrix."""

    design: np.ndarray            # (n, p)
    group_index: int
    hc_kind: str
    nuisance_residual_maker: np.ndarray   # M_Z, (n, n)
    nuisance_hat: np.ndarray              # H_Z, (n, n)
    full_residual_maker: np.ndarray       # M_X, (n, n)
    contrast_weights: np.ndarray          # a, (n,)
    variance_weights: np.ndarray          # w_i, (n,)
    kernel: np.ndarray                    # K = M_X' diag(w) M_X, (n, n)

    @property
    def n_subjects(self) -> int:
        return int(self.design.shape[0])

    @property
    def n_packed(self) -> int:
        n = self.n_subjects
        return n * (n + 1) // 2


def build_plan(
    design: np.ndarray, group_index: int, hc_kind: str = DEFAULT_HC
) -> FreedmanLanePlan:
    """Precompute everything that does not depend on the permutation or data."""
    if hc_kind not in HC_KINDS:
        raise ValueError(f"hc_kind must be one of {HC_KINDS}, got {hc_kind!r}")
    design = np.asarray(design, dtype=np.float64)
    if design.ndim != 2:
        raise ValueError("design matrix must be two-dimensional")
    n, p = design.shape
    if not 0 <= group_index < p:
        raise ValueError(f"group_index {group_index} outside design columns")
    if np.linalg.matrix_rank(design) < p:
        raise ValueError("design matrix is rank deficient; beta_G is not identifiable")

    identity = np.eye(n)
    group = design[:, group_index]
    nuisance = np.delete(design, group_index, axis=1)

    nuisance_hat = nuisance @ np.linalg.pinv(nuisance)
    nuisance_residual_maker = identity - nuisance_hat
    full_residual_maker = identity - design @ np.linalg.pinv(design)

    # Frisch-Waugh: beta_G = a'y with a = M_Z g / (g' M_Z g).
    residualized_group = nuisance_residual_maker @ group
    denominator = float(residualized_group @ residualized_group)
    if denominator <= 0.0:
        raise ValueError(
            "the group column is entirely explained by the nuisance columns; "
            "beta_G is not identifiable"
        )
    contrast_weights = residualized_group / denominator

    leverage = np.diag(design @ np.linalg.inv(design.T @ design) @ design.T)
    if np.any(leverage >= 1.0):
        raise ValueError("a participant has leverage 1; the HC weight is undefined")
    exponent = {"HC0": 0.0, "HC2": 1.0, "HC3": 2.0}[hc_kind]
    variance_weights = contrast_weights**2 / (1.0 - leverage) ** exponent

    kernel = full_residual_maker.T @ np.diag(variance_weights) @ full_residual_maker
    kernel = 0.5 * (kernel + kernel.T)  # symmetrize against round-off

    return FreedmanLanePlan(
        design=design,
        group_index=group_index,
        hc_kind=hc_kind,
        nuisance_residual_maker=nuisance_residual_maker,
        nuisance_hat=nuisance_hat,
        full_residual_maker=full_residual_maker,
        contrast_weights=contrast_weights,
        variance_weights=variance_weights,
        kernel=kernel,
    )


# ── direct (slow, obviously correct) evaluation ─────────────────────────────
def statistic_direct(
    plan: FreedmanLanePlan, values: np.ndarray, permutation: np.ndarray | None = None
) -> float:
    """Fit the full model on one Freedman--Lane draw and studentize, literally.

    Kept deliberately naive: this is what the fast path is checked against.
    """
    design = plan.design
    values = np.asarray(values, dtype=np.float64)
    if permutation is None:
        draw = values
    else:
        draw = (
            plan.nuisance_residual_maker @ values
        )[permutation] + plan.nuisance_hat @ values

    gram_inverse = np.linalg.inv(design.T @ design)
    beta = gram_inverse @ design.T @ draw
    residuals = draw - design @ beta
    leverage = np.diag(design @ gram_inverse @ design.T)
    exponent = {"HC0": 0.0, "HC2": 1.0, "HC3": 2.0}[plan.hc_kind]
    omega = residuals**2 / (1.0 - leverage) ** exponent
    covariance = gram_inverse @ design.T @ np.diag(omega) @ design @ gram_inverse
    contrast = np.zeros(design.shape[1])
    contrast[plan.group_index] = 1.0
    return float(
        contrast @ beta / np.sqrt(contrast @ covariance @ contrast)
    )


# ── packed quadratic form ───────────────────────────────────────────────────
def packed_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n)


def pack_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Pack a symmetric matrix so that ``packed @ pack_outer(u) == u' M u``."""
    rows, columns = packed_indices(matrix.shape[0])
    packed = matrix[rows, columns].astype(np.float64, copy=True)
    packed[rows != columns] *= 2.0
    return packed


def pack_outer(values: np.ndarray) -> np.ndarray:
    """Pack the upper triangle of ``u u'`` for one or many edges.

    ``values`` is ``(n,)`` or ``(n, n_edges)``; the result is ``(n_packed,)``
    or ``(n_packed, n_edges)``.
    """
    values = np.asarray(values, dtype=np.float64)
    rows, columns = packed_indices(values.shape[0])
    if values.ndim == 1:
        return values[rows] * values[columns]
    return values[rows, :] * values[columns, :]


def nuisance_residuals(plan: FreedmanLanePlan, data: np.ndarray) -> np.ndarray:
    """``u = M_Z y``: computed once per edge, reused by every permutation."""
    return plan.nuisance_residual_maker @ np.asarray(data, dtype=np.float64)


def permutation_tables(
    plan: FreedmanLanePlan, permutations: np.ndarray, dtype=np.float32
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``W`` and ``KP`` tables a GPU backend consumes.

    ``permutations`` is ``(n_perm, n)``; row ``b`` maps output position ``i``
    to input participant ``permutations[b, i]``, matching
    ``draw = u[permutation] + H_Z y``.  Returns ``W`` of shape
    ``(n_perm, n)`` and ``KP`` of shape ``(n_perm, n_packed)``.
    """
    permutations = np.asarray(permutations, dtype=np.int64)
    if permutations.ndim != 2 or permutations.shape[1] != plan.n_subjects:
        raise ValueError(
            f"permutations must have shape (n_perm, {plan.n_subjects}) and each "
            f"row must be a full permutation of 0..{plan.n_subjects - 1}, but "
            f"got {permutations.shape}. The existing subject-level permutation "
            "files store only the membership of group A, so they cannot be "
            "used as-is for Freedman-Lane; generate full-index rows instead."
        )
    for index, row in enumerate(permutations):
        if not np.array_equal(np.sort(row), np.arange(plan.n_subjects)):
            raise ValueError(
                f"permutation row {index} is not a full permutation of "
                f"0..{plan.n_subjects - 1}; Freedman--Lane needs a complete "
                "participant reordering, not a group-membership row"
            )

    n_perm = permutations.shape[0]
    weights = np.empty((n_perm, plan.n_subjects), dtype=np.float64)
    packed = np.empty((n_perm, plan.n_packed), dtype=np.float64)
    scattered = np.empty((plan.n_subjects, plan.n_subjects), dtype=np.float64)
    for index, row in enumerate(permutations):
        # The draw is v_i = u[row[i]], so both halves are scatters onto `row`:
        #   a'v      = sum_i a_i u[row[i]]        -> weights[row[i]] = a_i
        #   v'K v    = sum_ij K_ij u[row[i]]u[row[j]] -> S[row[i], row[j]] = K_ij
        # which keeps u (and therefore the packed outer product) in the
        # original participant order, so it is computed once per edge.
        weights[index, row] = plan.contrast_weights
        scattered[np.ix_(row, row)] = plan.kernel
        packed[index] = pack_symmetric(scattered)
    return weights.astype(dtype), packed.astype(dtype)


def statistics_from_tables(
    weights: np.ndarray, packed_kernel: np.ndarray, residuals: np.ndarray
) -> np.ndarray:
    """Evaluate every (permutation, edge) statistic as two matrix products.

    ``residuals`` is ``u`` with shape ``(n, n_edges)``.  Returns
    ``(n_perm, n_edges)``.
    """
    numerators = weights @ residuals
    denominators = packed_kernel @ pack_outer(residuals).astype(
        packed_kernel.dtype, copy=False
    )
    return numerators / np.sqrt(denominators)


def statistics(
    plan: FreedmanLanePlan,
    data: np.ndarray,
    permutations: np.ndarray,
    dtype=np.float64,
) -> np.ndarray:
    """Convenience wrapper: raw data in, ``(n_perm, n_edges)`` statistics out."""
    weights, packed = permutation_tables(plan, permutations, dtype=dtype)
    return statistics_from_tables(
        weights, packed, nuisance_residuals(plan, data).astype(dtype, copy=False)
    )


def effective_degrees_of_freedom(plan: FreedmanLanePlan) -> float:
    """Residual degrees of freedom of the full model, ``n - rank(X)``.

    The df-aware cluster-forming threshold needs a df.  Unlike Welch's
    edge-specific Satterthwaite df, the HC-studentized coefficient has no
    exact small-sample null distribution, so this fixed residual df is a
    documented approximation used only to convert a cluster-forming p into a
    ``|t|`` threshold -- never to report a p-value.  Family-wise error control
    comes from the permutation distribution, which does not rely on it.
    """
    return float(plan.n_subjects - np.linalg.matrix_rank(plan.design))


# ── CUDA plan file ──────────────────────────────────────────────────────────
# The GPU never materializes K. Instead it uses the projector form
#     M_X v = v - X d,   d = (X'X)^-1 X' v
# so the quadratic form costs O(n*p) rather than O(n^2/2) per (edge,
# permutation): ~600 flops instead of ~2350 at n=68, p=4. Likewise the
# nuisance residual is u = y - Z (Z'Z)^-1 Z' y, needing only p_z temporaries
# and allowing the input buffer to be updated in place.
#
# Gathering v_i = u[perm[i]] would make every access a scatter. Substituting
# i = perm^-1(m) instead moves the permutation into four tiny host-side
# tables, leaving the kernel with sequential reads of u.
CUDA_PLAN_MAGIC = 0x464C504E
CUDA_PLAN_VERSION = 1
CUDA_PLAN_HEADER = struct.Struct("<IIIIIf")


def write_cuda_plan(plan: FreedmanLanePlan, path: Path) -> dict[str, object]:
    """Write the fixed algebra the CUDA backend needs, as float32."""
    design = plan.design
    n, p = design.shape
    nuisance = np.delete(design, plan.group_index, axis=1)
    n_nuisance = nuisance.shape[1]

    gram_nuisance = np.linalg.pinv(nuisance)          # (p_z, n): d_z = Gz y
    gram_full = np.linalg.pinv(design)                # (p,   n): d   = Gx v

    with path.open("wb") as stream:
        stream.write(
            CUDA_PLAN_HEADER.pack(
                CUDA_PLAN_MAGIC, CUDA_PLAN_VERSION, n, p, n_nuisance,
                effective_degrees_of_freedom(plan),
            )
        )
        for block in (
            nuisance,            # Z   (n, p_z)
            gram_nuisance,       # Gz  (p_z, n)
            design,              # X   (n, p)
            gram_full,           # Gx  (p, n)
            plan.contrast_weights,   # a  (n,)
            plan.variance_weights,   # w  (n,)
        ):
            np.ascontiguousarray(block, dtype="<f4").tofile(stream)

    return {
        "cuda_plan": str(path.resolve()),
        "cuda_plan_sha256": sha256_file(path),
        "cuda_plan_bytes": path.stat().st_size,
        "cuda_plan_layout": (
            "header(magic,version,n,p,p_z,residual_df) then float32 "
            "Z(n,p_z), Gz(p_z,n), X(n,p), Gx(p,n), a(n), w(n)"
        ),
    }


def statistics_projector(
    plan: FreedmanLanePlan, data: np.ndarray, permutations: np.ndarray
) -> np.ndarray:
    """Evaluate statistics exactly the way the CUDA kernel does.

    Kept beside the GEMM path so the two can be cross-checked without a GPU:
    same algebra, different factorization.
    """
    design = plan.design
    gram_full = np.linalg.pinv(design)
    residuals = nuisance_residuals(plan, np.asarray(data, dtype=np.float64))
    output = np.empty((len(permutations), residuals.shape[1]))
    for index, row in enumerate(permutations):
        permuted = residuals[row, :]                       # v_i = u[perm[i]]
        fitted = design @ (gram_full @ permuted)           # X d
        full_residuals = permuted - fitted                 # M_X v
        output[index] = (plan.contrast_weights @ permuted) / np.sqrt(
            plan.variance_weights @ (full_residuals**2)
        )
    return output


# ── CLI: emit the precomputed tables ────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--design", required=True, type=Path,
        help="design.npz written by design_matrix.py",
    )
    parser.add_argument(
        "--permutations", required=True, type=Path,
        help="full-index permutation file; row 0 must be the identity",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hc-kind", choices=HC_KINDS, default=DEFAULT_HC)
    parser.add_argument(
        "--rows", choices=("inference", "calibration", "all", "custom"),
        default="inference",
        help="which permutation rows to tabulate. 'inference' is row 0 plus "
             "the held-out inference range (the default, and what FWER "
             "inference consumes); 'calibration' is the calibration range "
             "only; 'custom' uses --start-row/--count.",
    )
    parser.add_argument(
        "--start-row", type=int, default=0,
        help="first permutation row to tabulate, with --rows custom",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="number of permutation rows to tabulate, with --rows custom",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32",
        help="table precision; float32 keeps the absolute error on t below "
             "about 2e-6 (see the module docstring)",
    )
    add_partition_arguments(parser, stage="inference")
    return parser.parse_args(argv)


def select_rows(args, partition, total_rows: int) -> list[int]:
    """Resolve --rows into explicit permutation row indices."""
    if args.rows == "inference":
        rows = partition.inference_rows_with_observed
    elif args.rows == "calibration":
        rows = partition.calibration_rows
    elif args.rows == "all":
        rows = list(range(total_rows))
    else:
        stop = total_rows if args.count is None else args.start_row + args.count
        rows = list(range(args.start_row, stop))
    if not rows:
        raise ValueError("no permutation rows selected")
    if rows[0] < 0 or rows[-1] >= total_rows:
        raise ValueError(
            f"requested rows {rows[0]}..{rows[-1]} but the file has {total_rows}"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = np.load(args.design, allow_pickle=False)
    design = bundle["matrix"]
    group_index = int(bundle["group_index"])
    column_names = [str(name) for name in bundle["column_names"]]

    plan = build_plan(design, group_index, args.hc_kind)

    permutations = np.loadtxt(args.permutations, dtype=np.int64, ndmin=2)
    partition = partition_from_args(args)
    if args.rows != "custom":
        # Same validation the other stages apply, plus the full-index contract.
        validate_permutation_file(
            args.permutations, partition,
            allow_extra_rows=args.allow_extra_permutation_rows,
            representation="full-index", n_subjects=plan.n_subjects,
        )
    row_indices = select_rows(args, partition, permutations.shape[0])
    selected = permutations[row_indices]
    if row_indices[0] == 0 and not np.array_equal(
        selected[0], np.arange(plan.n_subjects)
    ):
        raise ValueError(
            "row 0 of the permutation file must be the identity permutation "
            "(the observed, unpermuted assignment)"
        )

    dtype = np.float32 if args.dtype == "float32" else np.float64
    weights, packed = permutation_tables(plan, selected, dtype=dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cuda_plan = write_cuda_plan(plan, args.output_dir / "freedman_lane_plan.flp")
    np.savez(
        args.output_dir / "freedman_lane_tables.npz",
        weights=weights,
        packed_kernel=packed,
        nuisance_residual_maker=plan.nuisance_residual_maker.astype(dtype),
        contrast_weights=plan.contrast_weights,
        variance_weights=plan.variance_weights,
        kernel=plan.kernel,
        permutation_rows=np.asarray(row_indices, dtype=np.int64),
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "02_cudaPerm/freedman_lane.py",
        "command_line": sys.argv,
        "design": str(args.design.resolve()),
        "design_columns": column_names,
        "group_column": column_names[group_index],
        "permutation_file": str(args.permutations.resolve()),
        "permutation_file_sha256": sha256_file(args.permutations),
        "permutation_representation": "full participant index order per row",
        "permutation_row_selection": args.rows,
        "permutation_rows_first_last": [row_indices[0], row_indices[-1]],
        "permutation_rows_contiguous": (
            row_indices == list(range(row_indices[0], row_indices[-1] + 1))
        ),
        **partition.to_manifest(),
        "n_permutations": int(selected.shape[0]),
        "n_subjects": plan.n_subjects,
        "n_packed": plan.n_packed,
        "statistic": f"{args.hc_kind}-studentized group coefficient",
        "statistic_note": (
            "HC2 reproduces Welch's unequal-variance t exactly for a two-group "
            "design with no covariates, so the adjusted model retains the "
            "existing pipeline's heteroscedasticity assumptions"
        ),
        "permutation_scheme": "Freedman-Lane residual permutation under the reduced model",
        "reference": (
            "Winkler AM et al. Permutation inference for the general linear "
            "model. NeuroImage 2014;92:381-397"
        ),
        "residual_degrees_of_freedom": effective_degrees_of_freedom(plan),
        "table_dtype": args.dtype,
        "table_bytes": int(weights.nbytes + packed.nbytes),
        **cuda_plan,
    }
    (args.output_dir / "freedman_lane_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"design: {plan.n_subjects} participants x {design.shape[1]} columns "
          f"({', '.join(column_names)})")
    print(f"statistic: {args.hc_kind}-studentized {column_names[group_index]} coefficient")
    print(f"tabulated {len(row_indices)} rows ({args.rows}): "
          f"{row_indices[0]}..{row_indices[-1]}")
    print(f"W {weights.shape} + KP {packed.shape} = "
          f"{(weights.nbytes + packed.nbytes) / 2**20:.1f} MiB ({args.dtype})")
    print(f"CUDA plan: {cuda_plan['cuda_plan']} ({cuda_plan['cuda_plan_bytes']} bytes)")
    print(f"written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
