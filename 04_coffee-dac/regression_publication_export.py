"""Regression tests for standardized publication-export data contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from mocca_gui.publication_exporter import (
    COLOR_MODE_GUI,
    PublicationExporter,
    SUBDIVISION_SAFEGUARD,
    _standard_camera_states,
    endpoint_incidence,
    resolve_parent_bundles,
)


class EndpointIncidenceTests(unittest.TestCase):
    def test_counts_both_ends_at_each_voxel(self):
        edges = np.asarray([
            [1, 2, 3, 4, 5, 6, .01, 3, 0, 0],
            [1, 2, 3, 4, 5, 6, .01, 4, 0, 0],
            [4, 5, 6, 7, 8, 9, .01, 5, 0, 0],
        ], dtype=float)
        observed = {
            tuple(row[:3]): int(row[3]) for row in endpoint_incidence(edges)
        }
        self.assertEqual(observed[(1, 2, 3)], 2)
        self.assertEqual(observed[(4, 5, 6)], 3)
        self.assertEqual(observed[(7, 8, 9)], 1)


class ParentResolutionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "inference": {
                "correction": "single_threshold_permutation_fwer",
                "alpha": 0.05,
                "positive_effect": "controls > patients",
                "negative_effect": "patients > controls",
                "selected_source_bundles": [
                    {"bundle": 95, "sign": -1, "edge_count": 2,
                     "mass": 8.0, "p_fwer": .04},
                    {"bundle": 94, "sign": 1, "edge_count": 3,
                     "mass": 12.0, "p_fwer": .02},
                ],
            }
        }

    def test_v2_compact_ids_map_back_to_source_inferential_bundles(self):
        edges = np.asarray([
            [0, 0, 0, 1, 1, 1, .02, 3, 0, 0],
            [0, 0, 1, 1, 1, 2, .02, 4, 0, 0],
            [0, 0, 2, 1, 1, 3, .02, 5, 0, 0],
            [2, 2, 2, 3, 3, 3, .04, -3, 1, 1],
            [2, 2, 3, 3, 3, 4, .04, -4, 1, 1],
        ], dtype=float)
        parents, manifest = resolve_parent_bundles(
            edges,
            [{"fcn": 0, "bundle": "All"}, {"fcn": 1, "bundle": "All"}],
            "v2",
            provenance=self.manifest,
        )
        self.assertIs(manifest, self.manifest)
        self.assertEqual([parent.identifier for parent in parents], ["94", "95"])
        self.assertEqual(parents[0].summary["mass"], 12.0)
        self.assertEqual(parents[0].summary["source_p_value"], .02)
        self.assertEqual(parents[0].summary["fwer_corrected_p"], .02)
        self.assertEqual(parents[1].summary["direction"], "patients > controls")

    def test_v3_partial_selection_exports_complete_parent_without_subdivision_inference(self):
        edges = np.asarray([
            [0, 0, 0, 1, 1, 1, .02, 3, 0, 0],
            [0, 0, 1, 1, 1, 2, .02, 4, 1, 0],
            [0, 0, 2, 1, 1, 3, .02, 5, 2, 0],
        ], dtype=float)
        one_parent_manifest = {
            "inference": {
                **self.manifest["inference"],
                "selected_source_bundles": [
                    {"bundle": 94, "sign": 1, "edge_count": 3,
                     "mass": 12.0, "p_fwer": .02},
                ],
            }
        }
        parents, _ = resolve_parent_bundles(
            edges,
            [{"fcn": 0, "bundle": 1}],
            "v3",
            provenance=one_parent_manifest,
        )
        self.assertEqual(len(parents), 1)
        self.assertEqual(len(parents[0].edges), 3)
        self.assertEqual(parents[0].selection, [{"fcn": 0, "bundle": "All"}])
        self.assertEqual(parents[0].summary["display_subdivision_count"], 3)
        self.assertEqual(len(parents[0].colors), 3)
        self.assertIn("not tested independently", SUBDIVISION_SAFEGUARD)

    def test_unprovenanced_pvalue_is_not_labeled_fwer_corrected(self):
        edges = np.asarray([
            [0, 0, 0, 1, 1, 1, .025, 3, 0, 0],
        ], dtype=float)
        parents, _ = resolve_parent_bundles(
            edges, [{"fcn": 0, "bundle": 0}], "v2"
        )
        self.assertEqual(parents[0].summary["source_p_value"], .025)
        self.assertIsNone(parents[0].summary["fwer_corrected_p"])
        self.assertIn(
            "provenance unavailable", parents[0].summary["inferential_status"]
        )

    def test_v3_two_parent_bundles_stay_independent(self):
        # A single v3 array can legitimately contain more than one
        # independently FWER-significant parent bundle (e.g. bundle 94 and
        # bundle 95 both survive FWER at alpha=0.05). Each parent bundle is
        # its own NETWORK_COL group and must resolve to its own ParentBundle,
        # with its own edges, subdivisions, and source p-value -- never
        # pooled with the other parent's.
        edges = np.asarray([
            # parent bundle 94 (NETWORK_COL 0): 2 display subdivisions
            [0, 0, 0, 1, 1, 1, .02, 3, 0, 0],
            [0, 0, 1, 1, 1, 2, .02, 4, 1, 0],
            # parent bundle 95 (NETWORK_COL 1): 1 display subdivision
            [5, 5, 5, 6, 6, 6, .04, -3, 0, 1],
        ], dtype=float)
        parents, manifest = resolve_parent_bundles(
            edges,
            [{"fcn": 0, "bundle": "All"}, {"fcn": 1, "bundle": "All"}],
            "v3",
            provenance=self.manifest,
        )
        self.assertIs(manifest, self.manifest)
        self.assertEqual(len(parents), 2)
        by_identifier = {parent.identifier: parent for parent in parents}
        self.assertEqual(set(by_identifier), {"94", "95"})
        self.assertEqual(len(by_identifier["94"].edges), 2)
        self.assertEqual(by_identifier["94"].summary["fwer_corrected_p"], .02)
        self.assertEqual(
            by_identifier["94"].summary["display_subdivision_count"], 2
        )
        self.assertEqual(len(by_identifier["95"].edges), 1)
        self.assertEqual(by_identifier["95"].summary["fwer_corrected_p"], .04)
        self.assertEqual(
            by_identifier["95"].summary["display_subdivision_count"], 1
        )

    def test_gui_color_mode_freezes_resolved_bundle_and_network_colors(self):
        edges = np.asarray([
            [0, 0, 0, 1, 1, 1, .02, 3, 0, 4],
            [0, 0, 1, 1, 1, 2, .02, 4, 1, 4],
        ], dtype=float)

        class Plotter:
            def resolve_bundle_color(self, fcn, bundle):
                return {
                    (4, 0): np.asarray((.1, .2, .3, 1.0)),
                    (4, 1): (.8, .7, .6, .5),
                }[(fcn, bundle)]

        parents, _ = resolve_parent_bundles(
            edges,
            [{"fcn": 4, "bundle": "All"}],
            "v3",
            network_plotter=Plotter(),
            color_mode=COLOR_MODE_GUI,
        )
        self.assertEqual(
            parents[0].colors,
            {(4, 0): (.1, .2, .3, 1.0), (4, 1): (.8, .7, .6, .5)},
        )


class CameraTests(unittest.TestCase):
    def test_all_standard_views_share_parallel_scale_and_distance(self):
        class Plotter:
            bounds = (0, 60, 0, 72, 0, 60)

        states = _standard_camera_states(Plotter())
        scales = {state["parallel_scale"] for state in states.values()}
        distances = {
            round(float(np.linalg.norm(
                state["position"] - state["focal_point"]
            )), 8)
            for state in states.values()
        }
        self.assertEqual(len(scales), 1)
        self.assertEqual(len(distances), 1)
        self.assertGreater(states["left_lateral"]["position"][0], 30)
        self.assertLess(states["right_lateral"]["position"][0], 30)


class PublicationSetTests(unittest.TestCase):
    def test_export_set_contains_primary_supplementary_and_manifest_outputs(self):
        edges = np.asarray([
            [1, 2, 3, 4, 5, 6, .025, 3, 0, 0],
        ], dtype=float)

        class NetworkPlotter:
            _edge_actors = [object()]
            last_selection = [{"fcn": 0, "bundle": 0}]
            _brain_meshes = []
            _live_brain_actors = []
            brain_opacity_scale = 1.0
            wm_visible = False

            @staticmethod
            def resolve_bundle_color(fcn, bundle):
                return (.2, .4, .6, 1.0)

        panel = Image.new("RGB", (80, 80), "white")
        views = {
            "left_lateral": panel,
            "superior_dorsal": panel,
            "right_lateral": panel,
        }
        cameras = {
            name: {"position": [0, 0, 1], "focal_point": [0, 0, 0],
                   "view_up": [0, 1, 0], "parallel_scale": 1}
            for name in views
        }
        density = {
            "representation": "anatomical maximum-intensity projection",
            "projection_axis": "superior-inferior (K)",
            "values": "raw endpoint-voxel incidence counts",
            "normalization": "none",
        }

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "mocca_gui.publication_exporter._render_brain_views",
            return_value=(views, cameras),
        ), mock.patch(
            "mocca_gui.publication_exporter._endpoint_density_image",
            return_value=(panel, np.asarray([[1, 2, 3, 1]]), density),
        ):
            outputs = PublicationExporter().export(
                directory, NetworkPlotter(), edges, pipeline="v2",
                color_mode=COLOR_MODE_GUI,
            )
            root = Path(outputs["directory"])
            expected = {
                "bundle_summary.csv",
                "figure_captions.txt",
                "figure_parent_bundles.pdf",
                "figure_parent_bundles.png",
                "parent_bundle_0_endpoint_density.csv",
                "parent_bundle_0_endpoints.png",
                "parent_bundle_0_full_edges.png",
                "parent_bundle_0_views.png",
                "publication_export_manifest.json",
            }
            self.assertEqual({path.name for path in root.iterdir()}, expected)
            manifest = json.loads(
                (root / "publication_export_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["centroid_line_width_mapping"]["edge_count_range"],
                [1, 1],
            )
            self.assertTrue(manifest["palette"]["fixed_across_all_outputs"])
            self.assertEqual(manifest["palette"]["mode"], COLOR_MODE_GUI)
            self.assertEqual(
                manifest["parent_bundles"][0]["colors"][
                    "network_0_display_group_0"
                ],
                [.2, .4, .6, 1.0],
            )
            self.assertIsNone(
                manifest["parent_bundles"][0]["summary"]["fwer_corrected_p"]
            )
            self.assertTrue(all(len(item["sha256"]) == 64
                                for item in manifest["outputs"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
