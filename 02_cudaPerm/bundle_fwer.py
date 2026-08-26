"""Bundle-level statistics for future permutation-based FWER inference.

This module deliberately leaves the established COFFEE-DAC pipelines
untouched.  It composes their existing, tested spatial routines through the
end of bundle pruning, stops before hierarchical network clustering, and
reduces the surviving bundles to one maximum statistic.

The function in this module is not, by itself, an FWER correction.  It is the
deterministic statistic that must be evaluated for the observed grouping and
for every permuted grouping.  The resulting permutation maxima can then be
used to assign bundle-level corrected p-values.
"""

from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal
import io
import sys

import numpy as np
import pandas as pd

# The established COFFEE-DAC routines remain in their original directory.
COFFEE_DAC_DIR = Path(__file__).resolve().parents[1] / "04_coffee-dac"
if str(COFFEE_DAC_DIR) not in sys.path:
    sys.path.insert(0, str(COFFEE_DAC_DIR))

from coffee_dac_pipeline import BUNDLE_COL, NETWORK_COL
from coffee_dac_pipeline_v2 import (
    TSTAT_COL,
    assign_bundle_labels_cc,
    assign_bundle_labels_strict,
    filter_isolated_edges,
    filter_small_networks,
    prune_intra_network_isolated,
    prune_small_endpoint_clusters,
)


BundleStatistic = Literal["mass", "extent"]
EdgeInput = np.ndarray | pd.DataFrame | str | PathLike[str]


@dataclass(frozen=True)
class BundleSummary:
    """Statistic and basic provenance for one surviving bundle."""

    bundle: int
    sign: int
    edge_count: int
    mass: float
    statistic: float


@dataclass(frozen=True)
class BundleStatisticResult:
    """Full deterministic output underlying ``max_bundle_statistic``."""

    max_statistic: float
    bundles: tuple[BundleSummary, ...]
    edges_bundled: np.ndarray
    input_edge_count: int
    threshold_edge_count: int

    @property
    def retained_edge_count(self) -> int:
        return int(self.edges_bundled.shape[0])

    @property
    def bundle_count(self) -> int:
        return len(self.bundles)


def _coerce_edges(edges: EdgeInput) -> np.ndarray:
    if isinstance(edges, (str, PathLike)):
        array = pd.read_csv(Path(edges)).to_numpy()
    elif isinstance(edges, pd.DataFrame):
        array = edges.to_numpy()
    else:
        array = np.asarray(edges)

    if array.ndim != 2 or array.shape[1] != 8:
        raise ValueError(
            "Expected raw edges with exactly 8 columns: "
            "i1,j1,k1,i2,j2,k2,pvalue,tstat."
        )
    if not np.all(np.isfinite(array[:, :6])):
        raise ValueError("Endpoint coordinates must be finite.")
    if not np.all(np.isfinite(array[:, TSTAT_COL])):
        raise ValueError("T-statistics must be finite.")
    return np.asarray(array, dtype=np.float64)


def _empty_bundled_edges() -> np.ndarray:
    return np.empty((0, NETWORK_COL + 1), dtype=np.float64)


def _bundle_one_sign(
    edges: np.ndarray,
    *,
    neighbor_dist: float,
    min_size: int,
    min_cluster_voxels: int,
    strict_bundles: bool,
) -> np.ndarray:
    """Run the unchanged v2 spatial stages for one effect direction."""

    if edges.shape[0] == 0:
        return _empty_bundled_edges()

    edges_filtered, _ = filter_isolated_edges(
        edges, neighbor_dist=neighbor_dist
    )
    if edges_filtered.shape[0] == 0:
        return _empty_bundled_edges()

    bundler = (
        assign_bundle_labels_strict
        if strict_bundles
        else assign_bundle_labels_cc
    )
    edges_bundled, _ = bundler(
        edges_filtered, neighbor_dist=neighbor_dist
    )

    edges_bundled, _, _ = prune_intra_network_isolated(
        edges_bundled, neighbor_dist=neighbor_dist
    )
    if edges_bundled.shape[0] == 0:
        return _empty_bundled_edges()

    edges_bundled, _, _ = filter_small_networks(
        edges_bundled, min_size=min_size
    )
    if edges_bundled.shape[0] == 0:
        return _empty_bundled_edges()

    edges_bundled, _, _ = prune_small_endpoint_clusters(
        edges_bundled,
        min_cluster_voxels=min_cluster_voxels,
        neighbor_dist=neighbor_dist,
    )
    if edges_bundled.shape[0] == 0:
        return _empty_bundled_edges()

    edges_bundled, _, _ = filter_small_networks(
        edges_bundled, min_size=min_size
    )
    return edges_bundled


def _combine_sign_results(results: list[np.ndarray]) -> np.ndarray:
    nonempty = [edges for edges in results if edges.shape[0] > 0]
    if not nonempty:
        return _empty_bundled_edges()

    combined: list[np.ndarray] = []
    label_offset = 0
    for edges in nonempty:
        current = edges.copy()
        current[:, BUNDLE_COL] += label_offset
        current[:, NETWORK_COL] += label_offset
        label_offset += len(np.unique(current[:, BUNDLE_COL].astype(int)))
        combined.append(current)
    return np.vstack(combined)


def compute_bundle_statistics(
    edges: EdgeInput,
    *,
    cluster_forming_threshold: float = 0.0,
    statistic: BundleStatistic = "mass",
    neighbor_dist: float = 1.0,
    min_size: int = 10,
    min_cluster_voxels: int = 6,
    strict_bundles: bool = True,
    split_signs: bool = True,
    verbose: bool = False,
) -> BundleStatisticResult:
    """Form pruned bundles and calculate their extent or excess-t mass.

    Parameters match the established v2 spatial stages.  When
    ``cluster_forming_threshold`` is positive, only edges satisfying
    ``abs(tstat) >= threshold`` enter bundle formation and bundle mass is
    ``sum(abs(tstat) - threshold)``.  A zero threshold is useful for replaying
    existing pre-thresholded CSVs and makes mass equal ``sum(abs(tstat))``.

    Positive and negative effects are processed independently by default, but
    their bundle labels share one output namespace and their maximum statistic
    is taken jointly.  This is the appropriate deterministic statistic for a
    future two-sided bundle-max permutation distribution.
    """

    if statistic not in ("mass", "extent"):
        raise ValueError("statistic must be 'mass' or 'extent'.")
    if cluster_forming_threshold < 0:
        raise ValueError("cluster_forming_threshold must be non-negative.")
    if neighbor_dist < 0:
        raise ValueError("neighbor_dist must be non-negative.")
    if min_size < 1 or min_cluster_voxels < 1:
        raise ValueError("Bundle and endpoint-cluster sizes must be >= 1.")

    raw = _coerce_edges(edges)
    input_edge_count = int(raw.shape[0])
    threshold_mask = (
        np.abs(raw[:, TSTAT_COL]) >= cluster_forming_threshold
    )
    thresholded = raw[threshold_mask]

    if split_signs:
        sign_inputs = [
            thresholded[thresholded[:, TSTAT_COL] > 0],
            thresholded[thresholded[:, TSTAT_COL] < 0],
        ]
    else:
        sign_inputs = [thresholded]

    output_context = nullcontext() if verbose else redirect_stdout(io.StringIO())
    with output_context:
        sign_results = [
            _bundle_one_sign(
                sign_edges,
                neighbor_dist=neighbor_dist,
                min_size=min_size,
                min_cluster_voxels=min_cluster_voxels,
                strict_bundles=strict_bundles,
            )
            for sign_edges in sign_inputs
        ]

    edges_bundled = _combine_sign_results(sign_results)
    summaries: list[BundleSummary] = []

    for label in np.unique(edges_bundled[:, BUNDLE_COL].astype(int)):
        bundle_edges = edges_bundled[
            edges_bundled[:, BUNDLE_COL].astype(int) == label
        ]
        tstats = bundle_edges[:, TSTAT_COL]
        sign = int(np.sign(tstats[0])) if split_signs else 0
        edge_count = int(bundle_edges.shape[0])
        mass = float(
            np.sum(np.abs(tstats) - cluster_forming_threshold, dtype=np.float64)
        )
        value = mass if statistic == "mass" else float(edge_count)
        summaries.append(
            BundleSummary(
                bundle=int(label),
                sign=sign,
                edge_count=edge_count,
                mass=mass,
                statistic=value,
            )
        )

    max_statistic = max(
        (summary.statistic for summary in summaries), default=0.0
    )
    return BundleStatisticResult(
        max_statistic=float(max_statistic),
        bundles=tuple(summaries),
        edges_bundled=edges_bundled,
        input_edge_count=input_edge_count,
        threshold_edge_count=int(thresholded.shape[0]),
    )


def max_bundle_statistic(edges: EdgeInput, **kwargs: object) -> float:
    """Return the maximum surviving bundle statistic for one label grouping."""

    return compute_bundle_statistics(edges, **kwargs).max_statistic
