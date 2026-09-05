"""Threshold-free cluster enhancement over the bundle edge graph.

Reference implementation for the ``tfce`` statistic in ``bundle_fwer_omp.cpp``.
Python stays the oracle: this module is written for clarity and independent
numerics (``scipy.stats.t`` rather than the backend's tabulated Student-t
quantiles), and ``regression_bundle_tfce.py`` holds the two to agreement.

Following Smith & Nichols (*NeuroImage* 2009;44:83-98), each element accrues

    TFCE(e) = sum_h  extent(e, h)^E * h^H * dh

with three adaptations to this pipeline, all deliberate:

* **Elements are edges, not voxels**, and the adjacency integrated over is the
  strict bundler's own: two edges join when they share an endpoint voxel *and*
  their free endpoints are within Chebyshev ``neighbor_dist``. Nothing new is
  invented for TFCE -- it enhances the bundle geometry already in use.

* **Height is the z-equivalent of the edge's two-sided p-value, not |t|.**
  Welch's Satterthwaite df varies per edge, so raw |t| is not comparable
  across the map; thresholding at height ``z`` means thresholding each edge at
  its own critical t for ``p = 2(1 - Phi(z))``.

* **Extent is the count of distinct voxels a bundle touches, not its edge
  count.** A densely connected region of V voxels carries on the order of
  V^2/2 edges, so an edge-count extent would act like a doubled exponent and
  restore exactly the giant-component domination TFCE is meant to damp.

No pruning (isolated-edge removal, ``min_size``, ``min_cluster_voxels``) is
applied during the integration. Those stages are legibility filters and are
applied only to the bundles finally reported, so the enhanced statistic -- and
therefore the permutation null it is ranked against -- is not shaped by them.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import io
import sys

import numpy as np
from scipy import stats

COFFEE_DIR = Path(__file__).resolve().parents[1] / "04_coffee-dac"
if str(COFFEE_DIR) not in sys.path:
    sys.path.insert(0, str(COFFEE_DIR))

from coffee_dac_pipeline import BUNDLE_COL
from coffee_dac_pipeline_v2 import assign_bundle_labels_strict


def _strict_labels(
    coordinates: np.ndarray, tstat: np.ndarray, neighbor_dist: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bundle labels from the validated strict bundler.

    That function writes its labels at the pipeline's fixed column positions,
    so it must be handed the pipeline's full edge layout (six coordinates, a
    p-value column, a statistic column) rather than coordinates alone. It also
    reports to stdout, which is suppressed here as it is in ``bundle_fwer``.
    """

    payload = np.empty((coordinates.shape[0], 8), dtype=float)
    payload[:, 0:6] = coordinates[:, 0:6]
    payload[:, 6] = np.nan
    payload[:, 7] = tstat
    with redirect_stdout(io.StringIO()):
        labelled, _ = assign_bundle_labels_strict(
            payload, neighbor_dist=neighbor_dist
        )
    return labelled, labelled[:, BUNDLE_COL].astype(int)


def two_sided_p_from_z(z: np.ndarray | float) -> np.ndarray | float:
    """The two-sided normal p-value whose z-equivalent height is ``z``."""

    return stats.norm.sf(z) * 2.0


def z_grid(z_min: float, z_max: float, z_step: float) -> np.ndarray:
    """Integration heights, inclusive of ``z_min``.

    Mirrors the backend's grid construction exactly, including its tolerance,
    so the two integrate over the same heights rather than merely similar ones.
    """

    if not z_step > 0:
        raise ValueError("z_step must be positive")
    if not z_max > z_min:
        raise ValueError("z_max must exceed z_min")
    if not z_min > 0:
        raise ValueError("z_min must be positive")
    steps = int(np.floor((z_max - z_min) / z_step + 1e-9)) + 1
    return z_min + np.arange(steps, dtype=float) * z_step


def critical_t(degrees_of_freedom: np.ndarray, z: float) -> np.ndarray:
    """Per-edge |t| that corresponds to height ``z`` at that edge's df."""

    return stats.t.isf(two_sided_p_from_z(z) / 2.0, degrees_of_freedom)


def tfce_scores(
    edges: np.ndarray,
    tstat: np.ndarray,
    degrees_of_freedom: np.ndarray,
    *,
    z_values: np.ndarray,
    z_step: float,
    extent_exponent: float = 0.5,
    height_exponent: float = 2.0,
    neighbor_dist: float = 1.0,
) -> np.ndarray:
    """Per-edge TFCE for one effect direction.

    ``edges`` is an ``(N, >=6)`` array whose first six columns are the two
    endpoint coordinates, matching the pipeline's edge layout everywhere else.
    Sign splitting is the caller's job, exactly as in the backend.
    """

    edges = np.asarray(edges, dtype=float)
    tstat = np.asarray(tstat, dtype=float)
    degrees_of_freedom = np.asarray(degrees_of_freedom, dtype=float)
    if edges.ndim != 2 or edges.shape[1] < 6:
        raise ValueError("edges must be (N, >=6) with two endpoint triplets")
    if tstat.shape != (edges.shape[0],) or degrees_of_freedom.shape != tstat.shape:
        raise ValueError("tstat and degrees_of_freedom must be one per edge")

    scores = np.zeros(edges.shape[0], dtype=float)
    if edges.shape[0] == 0:
        return scores

    magnitude = np.abs(tstat)
    for height in z_values:
        selected = np.nonzero(magnitude >= critical_t(degrees_of_freedom, height))[0]
        if selected.size == 0:
            # Heights are nested, so nothing survives above this one either.
            break
        labelled, labels = _strict_labels(
            edges[selected], tstat[selected], neighbor_dist
        )

        # Distinct voxels cannot be summed across a union: two bundles are
        # separate exactly when they fail the shared-voxel-plus-neighbouring-
        # free-end test, which does not stop them touching a voxel in common.
        extent = {}
        for label in np.unique(labels):
            member = labelled[labels == label, :6]
            touched = set(map(tuple, np.rint(member[:, 0:3]).astype(int)))
            touched.update(map(tuple, np.rint(member[:, 3:6]).astype(int)))
            extent[label] = len(touched)

        weight = height ** height_exponent * z_step
        scores[selected] += np.array(
            [extent[label] ** extent_exponent for label in labels]
        ) * weight
    return scores
