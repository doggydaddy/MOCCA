#!/usr/bin/env python3
"""Monte Carlo precision report for a completed run_bundle_fwer.py result.

Every FWER p-value this pipeline reports is a permutation-count ratio r/m
(m = null_permutations + 1 trials, including the observed row itself. Under the
disjoint row partition the config's `null_permutations` is the *inference* null
count, so m excludes the calibration rows exactly as the p-value does; r =
how many of those m trials are at least as extreme as observed). That ratio
is a binomial proportion, so it carries sampling uncertainty from having run
only m permutations -- this script attaches an exact (Clopper-Pearson)
confidence interval to each reported p-value and flags bundles whose CI
straddles --alpha, i.e. bundles where more permutations could plausibly flip
the significance call.

Supports both run_bundle_fwer.py output layouts:
  - single-threshold: observed_bundles_fwer.csv + null_max_bundle_statistics.npy
  - grid:             observed_bundles_grid_fwer.csv + permutation_bundle_maxima_grid.csv

Usage:
    .venv/bin/python 02_cudaPerm/bundle_fwer_precision.py \\
        /path/to/bundle_fwer_result_dir --alpha 0.05
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


def clopper_pearson(r: np.ndarray, m: int, confidence: float):
    """Exact binomial CI for r successes out of m trials (vectorized over r)."""
    alpha = 1 - confidence
    r = np.asarray(r, dtype=float)
    lower = np.where(r > 0, beta.ppf(alpha / 2, r, m - r + 1), 0.0)
    upper = np.where(r < m, beta.ppf(1 - alpha / 2, r + 1, m - r), 1.0)
    return lower, upper


def annotate(p_hat: np.ndarray, r: np.ndarray, m: int, alpha: float, confidence: float) -> pd.DataFrame:
    ci_low, ci_high = clopper_pearson(r, m, confidence)
    se = np.sqrt(p_hat * (1 - p_hat) / m)
    return pd.DataFrame({
        "p_hat_recomputed": p_hat,
        "n_at_least_as_extreme": r.astype(int),
        "n_trials": m,
        "resolution_floor": 1.0 / m,
        "se_normal_approx": se,
        f"ci{int(confidence * 100)}_low": ci_low,
        f"ci{int(confidence * 100)}_high": ci_high,
        "ci_crosses_alpha": (ci_low <= alpha) & (alpha <= ci_high),
    })


def process_single_threshold(result_dir: Path, alpha: float, confidence: float):
    config = json.loads((result_dir / "bundle_fwer_config.json").read_text())
    m = config["null_permutations"] + 1

    null_maxima = np.load(result_dir / "null_max_bundle_statistics.npy")
    if len(null_maxima) != m - 1:
        raise RuntimeError(
            f"null_max_bundle_statistics.npy has {len(null_maxima)} rows, "
            f"expected {m - 1} from bundle_fwer_config.json's null_permutations"
        )

    observed = pd.read_csv(result_dir / "observed_bundles_fwer.csv")
    values = observed["statistic"].to_numpy(float)
    r = 1 + (null_maxima[None, :] >= values[:, None]).sum(axis=1)
    p_hat = r / m

    mismatch = np.abs(p_hat - observed["p_fwer"].to_numpy(float)) > 1e-9
    if mismatch.any():
        raise RuntimeError(
            f"{mismatch.sum()} bundle(s) recomputed to a different p_fwer than "
            "on disk -- result directory may be stale or from a different pipeline version."
        )

    annotated = pd.concat([observed, annotate(p_hat, r, m, alpha, confidence)], axis=1)
    return annotated, m


def process_grid(result_dir: Path, alpha: float, confidence: float):
    config = json.loads((result_dir / "bundle_fwer_config.json").read_text())
    m = config["null_permutations"] + 1

    maxima = pd.read_csv(result_dir / "permutation_bundle_maxima_grid.csv")
    if maxima["permutation"].nunique() != m:
        raise RuntimeError(
            f"permutation_bundle_maxima_grid.csv has {maxima['permutation'].nunique()} "
            f"distinct permutations, expected {m} from bundle_fwer_config.json's null_permutations"
        )
    all_min_rank = maxima.groupby("permutation")["min_rank_p"].first().to_numpy(float)

    observed = pd.read_csv(result_dir / "observed_bundles_grid_fwer.csv")

    threshold_r = np.empty(len(observed), dtype=int)
    for idx in observed["threshold_index"].unique():
        pool = maxima.loc[maxima["threshold_index"] == idx, "max_statistic"].to_numpy(float)
        rows = observed["threshold_index"].to_numpy() == idx
        values = observed.loc[rows, "statistic"].to_numpy(float)
        threshold_r[rows] = (pool[None, :] >= values[:, None]).sum(axis=1)
    threshold_p_hat = threshold_r / m

    grid_values = observed["p_threshold_fwer"].to_numpy(float)
    grid_r = (all_min_rank[None, :] <= grid_values[:, None]).sum(axis=1)
    grid_p_hat = grid_r / m

    mismatch = (
        (np.abs(threshold_p_hat - observed["p_threshold_fwer"].to_numpy(float)) > 1e-9)
        | (np.abs(grid_p_hat - observed["p_grid_fwer"].to_numpy(float)) > 1e-9)
    )
    if mismatch.any():
        raise RuntimeError(
            f"{mismatch.sum()} bundle(s) recomputed to a different p_threshold_fwer/"
            "p_grid_fwer than on disk -- result directory may be stale or from a "
            "different pipeline version."
        )

    threshold_ci = annotate(threshold_p_hat, threshold_r, m, alpha, confidence)
    threshold_ci = threshold_ci.add_prefix("threshold_")
    grid_ci = annotate(grid_p_hat, grid_r, m, alpha, confidence)
    grid_ci = grid_ci.add_prefix("grid_")
    annotated = pd.concat([observed, threshold_ci, grid_ci], axis=1)
    return annotated, m


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("result_dir", type=Path, help="run_bundle_fwer.py output directory")
    p.add_argument("--alpha", type=float, default=0.05, help="significance threshold to check CI coverage against (default 0.05)")
    p.add_argument("--confidence", type=float, default=0.95, help="confidence level for the Clopper-Pearson interval (default 0.95)")
    p.add_argument("--output", type=Path, default=None, help="output CSV path (default: <result_dir>/<observed csv name>_precision.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    is_grid = (args.result_dir / "observed_bundles_grid_fwer.csv").exists()
    is_single = (args.result_dir / "observed_bundles_fwer.csv").exists()
    if is_grid and is_single:
        raise RuntimeError(f"{args.result_dir} has both single-threshold and grid output; ambiguous.")
    if not is_grid and not is_single:
        raise RuntimeError(f"{args.result_dir} has neither observed_bundles_fwer.csv nor observed_bundles_grid_fwer.csv.")

    if is_single:
        annotated, m = process_single_threshold(args.result_dir, args.alpha, args.confidence)
        output = args.output or args.result_dir / "observed_bundles_fwer_precision.csv"
        crossing = annotated[annotated["ci_crosses_alpha"]]
        crossing_view = crossing[["bundle", "p_fwer", f"ci{int(args.confidence * 100)}_low", f"ci{int(args.confidence * 100)}_high"]]
    else:
        annotated, m = process_grid(args.result_dir, args.alpha, args.confidence)
        output = args.output or args.result_dir / "observed_bundles_grid_fwer_precision.csv"
        crossing = annotated[annotated["grid_ci_crosses_alpha"]]
        crossing_view = crossing[[
            "cluster_forming_p", "threshold_index", "bundle", "p_grid_fwer",
            f"grid_ci{int(args.confidence * 100)}_low", f"grid_ci{int(args.confidence * 100)}_high",
        ]]

    annotated.to_csv(output, index=False)
    print(f"wrote {output}")
    print(f"n_trials (null_permutations + 1) = {m}; resolution floor = 1/{m} = {1 / m:.6g}")
    if len(crossing_view):
        print(
            f"\n{len(crossing_view)} bundle(s) whose {int(args.confidence * 100)}% CI "
            f"straddles alpha={args.alpha:g} -- significance call is not resolved at "
            f"m={m} permutations:"
        )
        print(crossing_view.to_string(index=False))
    else:
        print(f"\nno bundle's {int(args.confidence * 100)}% CI straddles alpha={args.alpha:g}.")


if __name__ == "__main__":
    main()
