"""False-discovery-rate correction across bundles.

The FWER path in ``run_bundle_fwer.py`` compares each observed bundle against
the distribution of per-permutation *maxima*: one number per permutation, so a
bundle must out-rank the largest bundle almost every null permutation produced.
FDR instead asks how extreme a bundle is against the pooled distribution of
*individual* null bundle statistics, and then adjusts those uncorrected
p-values with a step-up procedure.

The two answer different questions and give different guarantees. FWER bounds
the probability of *any* false bundle; FDR bounds the expected proportion of
false bundles among those declared. An FDR-declared bundle can be large, so
"false bundle" is not a mild failure -- see the Chumbley & Friston (2009)
discussion of topological FDR before reporting these as if they were FWER
results.

Both Benjamini-Hochberg and Benjamini-Yekutieli adjusted p-values are always
computed. BH assumes positive regression dependence (PRDS); bundle statistics
within one permutation are dependent in a way that is not proven to satisfy it,
so BY -- valid under arbitrary dependence at the cost of a ``log m`` factor --
is reported alongside rather than instead.
"""

from __future__ import annotations

import numpy as np


def pooled_null_p_values(
    observed: np.ndarray, null: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Uncorrected permutation p-values against a pooled null.

    ``null`` holds every bundle statistic from every null permutation, not the
    per-permutation maxima. The ``+1`` in numerator and denominator counts the
    observed bundle itself as one draw, which keeps the p-value strictly
    positive and matches the convention used by the FWER path.
    """

    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    if observed.ndim != 1 or null.ndim != 1:
        raise ValueError("observed and null statistics must be 1-dimensional")
    if null.size == 0:
        raise ValueError("the pooled null bundle distribution is empty")
    if not np.all(np.isfinite(null)):
        raise ValueError("the pooled null bundle distribution is not finite")
    if observed.size and not np.all(np.isfinite(observed)):
        raise ValueError("observed bundle statistics are not finite")

    ordered = np.sort(null)
    # searchsorted(..., "left") is the index of the first null >= value, so the
    # difference from the end is the count of nulls >= value, ties included.
    exceedances = null.size - np.searchsorted(ordered, observed, side="left")
    p_values = (1.0 + exceedances) / (null.size + 1.0)
    return exceedances.astype(np.int64), p_values


def _step_up_adjust(p_values: np.ndarray, penalty: float) -> np.ndarray:
    """Step-up adjusted p-values: ``min_{j>=i} penalty * m / j * p_(j)``."""

    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p-values must be 1-dimensional")
    if p_values.size == 0:
        return np.empty(0, dtype=float)
    if not np.all(np.isfinite(p_values)):
        raise ValueError("p-values must be finite")
    if np.any(p_values < 0.0) or np.any(p_values > 1.0):
        raise ValueError("p-values must lie in [0, 1]")

    count = p_values.size
    order = np.argsort(p_values, kind="stable")
    ranks = np.arange(1, count + 1, dtype=float)
    scaled = penalty * count / ranks * p_values[order]
    # Enforce monotonicity from the largest p-value downwards, so that a
    # bundle is never declared while a more extreme one is not.
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    result = np.empty(count, dtype=float)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH adjusted p-values; valid under independence or PRDS."""

    return _step_up_adjust(p_values, 1.0)


def benjamini_yekutieli(p_values: np.ndarray) -> np.ndarray:
    """BY adjusted p-values; valid under arbitrary dependence."""

    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.empty(0, dtype=float)
    harmonic = float(np.sum(1.0 / np.arange(1, p_values.size + 1)))
    return _step_up_adjust(p_values, harmonic)


def harmonic_penalty(count: int) -> float:
    """The ``c(m) = sum_{k=1}^{m} 1/k`` factor BY pays for dependence."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, count + 1)))
