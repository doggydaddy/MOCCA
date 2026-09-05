"""Measure the false-positive behaviour of bundle-level FDR on null data.

Whether Benjamini-Hochberg's positive-dependence assumption (PRDS) holds for a
data-dependent partition of edges into bundles is not something either theory
or intuition settles cleanly here. It is, however, directly measurable.

Under the **complete null** every rejection is false, so the false discovery
proportion is 1 whenever anything is rejected and 0 otherwise, and therefore

    FDR = E[FDP] = P(at least one rejection).

A procedure controlling FDR at ``q`` must therefore declare at least one bundle
in at most a fraction ``q`` of pure-null permutations. That is a prediction
this pipeline's own permutations can check without appealing to PRDS at all.

Each null permutation is treated in turn as if it were the observed row: its
bundles are ranked against the pooled null built from *every other* null
permutation -- exact leave-one-out, so a permutation never contributes to the
distribution judging it -- then BH and BY are applied and the rejections
counted. The reported rate is what the guarantee promises to bound.

Two limits worth carrying into the write-up: this tests control under the
complete null only (FDR under a partial null is not directly checkable this
way), and it inherits whatever the bundling and threshold settings of the run
that produced the input were. It is the standard and most sensitive diagnostic
for the failure mode in question, not a proof of validity in general.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import false_discovery


def leave_one_out_p_values(
    statistics: np.ndarray, permutation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Uncorrected p-values for every bundle against all *other* permutations.

    Counting is done once against the whole pool and corrected per permutation
    by subtracting that permutation's own contribution, which is exact and
    avoids rebuilding a null per permutation.
    """

    statistics = np.asarray(statistics, dtype=float)
    permutation = np.asarray(permutation)
    if statistics.shape != permutation.shape:
        raise ValueError("statistics and permutation labels must align")
    if statistics.size == 0:
        raise ValueError("no bundle statistics supplied")

    total = statistics.size
    pooled = np.sort(statistics)
    against_all = total - np.searchsorted(pooled, statistics, side="left")

    exceedances = np.empty(total, dtype=np.int64)
    denominators = np.empty(total, dtype=np.int64)
    order = np.argsort(permutation, kind="stable")
    boundaries = np.flatnonzero(
        np.r_[True, permutation[order][1:] != permutation[order][:-1], True]
    )
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        members = order[start:stop]
        own = np.sort(statistics[members])
        within = own.size - np.searchsorted(own, statistics[members], side="left")
        exceedances[members] = against_all[members] - within
        denominators[members] = total - own.size

    if np.any(denominators <= 0):
        raise ValueError(
            "at least two permutations with bundles are required to build a "
            "leave-one-out null"
        )
    p_values = (1.0 + exceedances) / (denominators + 1.0)
    return exceedances, p_values


def calibrate(
    frame: pd.DataFrame, q: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-permutation rejection counts and the complete-null rates."""

    statistics = frame["statistic"].to_numpy(float)
    permutation = frame["permutation"].to_numpy(np.int64)
    _, p_values = leave_one_out_p_values(statistics, permutation)

    rows: list[dict[str, object]] = []
    order = np.argsort(permutation, kind="stable")
    ordered = permutation[order]
    boundaries = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        members = order[start:stop]
        own = p_values[members]
        bh = false_discovery.benjamini_hochberg(own)
        by = false_discovery.benjamini_yekutieli(own)
        rows.append({
            "permutation": int(permutation[members[0]]),
            "bundles": int(own.size),
            "min_p_uncorrected": float(own.min()),
            "rejections_bh": int(np.count_nonzero(bh <= q)),
            "rejections_by": int(np.count_nonzero(by <= q)),
        })

    per_permutation = pd.DataFrame(rows).sort_values("permutation")
    count = len(per_permutation)
    summary = {
        "q": q,
        "permutations": count,
        "pooled_bundles": int(statistics.size),
        "any_rejection_rate_bh": float(
            (per_permutation["rejections_bh"] > 0).mean()
        ),
        "any_rejection_rate_by": float(
            (per_permutation["rejections_by"] > 0).mean()
        ),
        "mean_rejections_bh": float(per_permutation["rejections_bh"].mean()),
        "mean_rejections_by": float(per_permutation["rejections_by"].mean()),
        "max_rejections_bh": int(per_permutation["rejections_bh"].max()),
        "max_rejections_by": int(per_permutation["rejections_by"].max()),
    }
    # Wilson interval, so a rate near q is not over-read at finite permutations.
    for method in ("bh", "by"):
        successes = float(summary[f"any_rejection_rate_{method}"]) * count
        summary[f"any_rejection_ci_{method}"] = wilson_interval(successes, count)
    return per_permutation, summary


def wilson_interval(successes: float, trials: int, z: float = 1.96) -> list[float]:
    if trials == 0:
        return [0.0, 1.0]
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    spread = z * np.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return [float(max(0.0, centre - spread)), float(min(1.0, centre + spread))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "bundle_statistics", type=Path,
        help="permutation_bundle_statistics.csv from a --fdr run",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--fdr-q", type=float, nargs="+", default=[0.05],
        help="target rate(s) to check (default 0.05)",
    )
    parser.add_argument(
        "--include-observed", action="store_true",
        help=("also treat row 0 as a null draw. Off by default: row 0 is the "
              "real data, so including it would not be a complete-null check"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.bundle_statistics.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    for column in ("permutation", "statistic"):
        if column not in frame.columns:
            raise ValueError(f"{path.name} has no '{column}' column")

    observed_bundles = int((frame["permutation"] == 0).sum())
    if not args.include_observed:
        frame = frame.loc[frame["permutation"] != 0]
    if frame.empty:
        raise ValueError("no null bundle statistics to calibrate on")
    if frame["permutation"].nunique() < 2:
        raise ValueError(
            "at least two permutations with bundles are required to build a "
            "leave-one-out null"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float]] = []
    for q in args.fdr_q:
        if not 0 < q < 1:
            raise ValueError("--fdr-q values must lie strictly between 0 and 1")
        per_permutation, summary = calibrate(frame, q)
        per_permutation.to_csv(
            output_dir / f"fdr_null_rejections_q{q:g}.csv", index=False
        )
        summaries.append(summary)
        for method in ("bh", "by"):
            rate = summary[f"any_rejection_rate_{method}"]
            low, high = summary[f"any_rejection_ci_{method}"]
            verdict = (
                "exceeds q" if low > q
                else ("at or below q" if high <= q else "indistinguishable from q")
            )
            print(
                f"[q={q:g}] {method.upper()}: any-rejection rate "
                f"{rate:.4f} (95% CI {low:.4f}-{high:.4f}) -- {verdict}; "
                f"mean {summary[f'mean_rejections_{method}']:.2f} / max "
                f"{summary[f'max_rejections_{method}']} bundles declared",
                flush=True,
            )

    results = {
        "source": str(path),
        "null_permutations": int(frame["permutation"].nunique()),
        "observed_bundles_present": observed_bundles,
        "observed_included_as_null": bool(args.include_observed),
        "identity": (
            "under the complete null FDP is 1 whenever anything is rejected, "
            "so FDR equals P(at least one rejection); a valid procedure at q "
            "must not exceed q here"
        ),
        "summaries": summaries,
    }
    (output_dir / "fdr_null_calibration_results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(f"Complete: {output_dir / 'fdr_null_calibration_results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
