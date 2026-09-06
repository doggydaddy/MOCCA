"""Regression tests for the v3 (divisive) pipeline's parent-bundle awareness.

A single bundle-level FWER visualization export can legitimately contain
more than one independently significant parent bundle (e.g. two bundles both
survive FWER at alpha=0.05 in the same run). These tests guard the
correctness property the rest of the v3 rewrite depends on: edges from
different parent bundles must never share a linkage tree, never influence
each other's distances, and never get mixed into the same displayed
sub-bundle -- see coffee_dac_pipeline_v3.py's module docstring.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from coffee_dac_pipeline import BUNDLE_COL, NETWORK_COL
from coffee_dac_pipeline_v3 import (
    _resolve_parent_ids,
    build_edge_linkage_per_parent,
    process_edge_data_v3,
    recut_subbundles,
)


def _cluster(rng, center, n, spread=0.3):
    ep1 = rng.normal(loc=center, scale=spread, size=(n, 3))
    ep2 = rng.normal(loc=np.asarray(center) + 5.0, scale=spread, size=(n, 3))
    return np.round(np.c_[ep1, ep2]).astype(int)


def _make_two_parent_edges():
    """Two parent bundles, well separated in space, each with its own
    internal sub-cluster structure -- parent 0 has 2 natural sub-clusters,
    parent 1 has 3."""
    rng = np.random.default_rng(0)
    parent0 = np.vstack([
        _cluster(rng, [0, 0, 0], 6),
        _cluster(rng, [50, 50, 50], 6),
    ])
    parent1 = np.vstack([
        _cluster(rng, [200, 200, 200], 5),
        _cluster(rng, [250, 250, 250], 5),
        _cluster(rng, [300, 300, 300], 5),
    ])

    def rows(coords, bundle, pvalue, tstat, network):
        n = coords.shape[0]
        extra = np.column_stack([
            np.full(n, pvalue), np.full(n, tstat),
            np.full(n, bundle), np.full(n, network),
        ])
        return np.hstack([coords, extra])

    all_rows = np.vstack([
        rows(parent0, 0, 0.042, -3.0, 0),
        rows(parent1, 1, 0.045, -3.0, 1),
    ])
    cols = ['i1', 'j1', 'k1', 'i2', 'j2', 'k2', 'pvalue', 'tstat',
            'bundle', 'network']
    df = pd.DataFrame(all_rows, columns=cols)
    df.iloc[:, :6] = df.iloc[:, :6].astype(int)
    df['bundle'] = df['bundle'].astype(int)
    df['network'] = df['network'].astype(int)
    return df


class ParentIdResolutionTests(unittest.TestCase):
    def test_uses_input_bundle_column_when_present(self):
        df = _make_two_parent_edges()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "singleFWER_v2_processed.csv"
            df.to_csv(csv_path, index=False)
            edges = df.to_numpy()
            parent_ids, source = _resolve_parent_ids(str(csv_path), edges)
            self.assertEqual(source, 'input_bundle_column')
            np.testing.assert_array_equal(parent_ids, edges[:, BUNDLE_COL].astype(int))

    def test_uses_sibling_v2_processed_csv_when_input_has_no_bundle_column(self):
        df = _make_two_parent_edges()
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "dataset_singleFWER.csv"
            df.iloc[:, :8].to_csv(raw_path, index=False)
            df.to_csv(raw_path.with_name("dataset_singleFWER_v2_processed.csv"), index=False)
            edges = df.iloc[:, :8].to_numpy()
            parent_ids, source = _resolve_parent_ids(str(raw_path), edges)
            self.assertEqual(source, 'sibling_v2_processed_csv')
            np.testing.assert_array_equal(parent_ids, df['bundle'].to_numpy())

    def test_falls_back_to_one_implicit_parent_with_no_grouping_info(self):
        df = _make_two_parent_edges()
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "dataset_singleFWER.csv"
            df.iloc[:, :8].to_csv(raw_path, index=False)
            edges = df.iloc[:, :8].to_numpy()
            parent_ids, source = _resolve_parent_ids(str(raw_path), edges)
            self.assertEqual(source, 'implicit_single_parent')
            self.assertTrue(np.all(parent_ids == 0))

    def test_mismatched_sibling_row_count_falls_back_safely(self):
        df = _make_two_parent_edges()
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "dataset_singleFWER.csv"
            df.iloc[:, :8].to_csv(raw_path, index=False)
            # Sibling exists but has a different row count -- must not be
            # used positionally, since that would silently mislabel edges.
            df.iloc[:-1].to_csv(
                raw_path.with_name("dataset_singleFWER_v2_processed.csv"), index=False
            )
            edges = df.iloc[:, :8].to_numpy()
            parent_ids, source = _resolve_parent_ids(str(raw_path), edges)
            self.assertEqual(source, 'implicit_single_parent')


class IndependentSubdivisionTests(unittest.TestCase):
    def test_two_parent_bundles_never_share_a_linkage_tree(self):
        df = _make_two_parent_edges()
        edges = df.to_numpy()
        parent_ids = edges[:, BUNDLE_COL].astype(int)
        linkage_matrices = build_edge_linkage_per_parent(edges, parent_ids)
        self.assertEqual(set(linkage_matrices), {0, 1})
        n_parent0 = int((parent_ids == 0).sum())
        n_parent1 = int((parent_ids == 1).sum())
        # N-1 merges for N leaves -- if the trees were pooled, both entries
        # would report a merge count matching the COMBINED edge count instead.
        self.assertEqual(linkage_matrices[0].shape[0], n_parent0 - 1)
        self.assertEqual(linkage_matrices[1].shape[0], n_parent1 - 1)

    def test_recutting_one_parent_does_not_change_the_other(self):
        df = _make_two_parent_edges()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "dataset_v2_processed.csv"
            df.to_csv(csv_path, index=False)
            result = process_edge_data_v3(
                str(csv_path), nr_bundles={0: 2, 1: 3}, invocation='test',
            )
        edges_out = result['edges_net']
        parent0_labels_before = edges_out[edges_out[:, NETWORK_COL] == 0, BUNDLE_COL].copy()

        edges_recut, nr_out_map = recut_subbundles(
            edges_out, result['linkage_matrices'], {0: 2, 1: 5},
        )
        self.assertEqual(nr_out_map, {0: 2, 1: 5})
        parent0_labels_after = edges_recut[edges_recut[:, NETWORK_COL] == 0, BUNDLE_COL]
        np.testing.assert_array_equal(parent0_labels_before, parent0_labels_after)

    def test_no_edge_crosses_between_parent_bundles_after_subdivision(self):
        df = _make_two_parent_edges()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "dataset_v2_processed.csv"
            df.to_csv(csv_path, index=False)
            result = process_edge_data_v3(
                str(csv_path), nr_bundles=2, invocation='test',
            )
        edges_out = result['edges_net']
        original_parent_ids = df['bundle'].to_numpy()
        # NETWORK_COL must still exactly match the original parent-bundle
        # assignment -- subdivision must never reassign which parent an
        # edge belongs to, only its display sub-bundle within that parent.
        np.testing.assert_array_equal(
            edges_out[:, NETWORK_COL].astype(int), original_parent_ids
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
