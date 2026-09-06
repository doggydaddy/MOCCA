"""Regression tests for publication-oriented current-view export."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mocca_gui.figure_exporter import (
    FigureExporter,
    PUBLICATION_DPI,
    SUBDIVISION_SAFEGUARD,
    trim_uniform_background,
)


class _Camera:
    def GetPosition(self):
        return (1.0, 2.0, 3.0)

    def GetFocalPoint(self):
        return (0.0, 0.0, 0.0)

    def GetViewUp(self):
        return (0.0, 0.0, 1.0)

    def GetViewAngle(self):
        return 30.0

    def GetParallelProjection(self):
        return False

    def GetParallelScale(self):
        return 1.0

    def GetClippingRange(self):
        return (0.1, 100.0)


class _Plotter:
    window_size = (100, 60)
    camera_position = (
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    camera = _Camera()

    def screenshot(self, **kwargs):
        self.screenshot_kwargs = kwargs
        image = np.full((60, 100, 3), 32, dtype=np.uint8)
        image[10:50, 20:80] = (180, 100, 40)
        return image


class _NetworkPlotter:
    def __init__(self):
        self.plotter = _Plotter()
        self._edge_actors = [object()]
        self.last_selection = [{"fcn": 0, "bundle": "All"}]
        self.last_endpoint_visible = True
        self.bundle_colors = {(0, 1): 2}
        self.centroid_flags = {(0, 1): True}
        self.thicknesses = {(0, 1): 4}
        self.curvatures = {(0, 1): 1.5}
        self.opacities = {(0, 1): 0.9}
        self.endpoint_sizes = {(0, 1): 1.25}
        self._brain_meshes = [
            ("brain.stl", 0.35, "grey", True, {}, False),
        ]
        self.brain_opacity_scale = 1.0
        self.wm_visible = True
        self._live_brain_actors = []
        self._orientation_widget = object()

    def resolve_bundle_color(self, fcn, bundle):
        from mocca_gui.colormap import my_colormap
        index = self.bundle_colors.get((fcn, bundle), fcn)
        return my_colormap.colors[index]


class CropTests(unittest.TestCase):
    def test_uniform_border_is_trimmed_with_padding(self):
        image = np.full((100, 120, 3), 10, dtype=np.uint8)
        image[30:70, 40:80] = 200
        cropped = trim_uniform_background(image, padding_fraction=0.0)
        self.assertEqual(cropped.shape, (44, 44, 3))

    def test_blank_image_is_left_intact(self):
        image = np.full((12, 16, 3), 10, dtype=np.uint8)
        self.assertEqual(trim_uniform_background(image).shape, image.shape)


class FigureExporterTests(unittest.TestCase):
    def test_png_export_writes_figure_caption_and_strict_metadata(self):
        edges = np.asarray([
            [0, 0, 0, 1, 1, 1, 0.012, 3.0, 1, 0],
            [0, 0, 1, 1, 1, 2, 0.012, 2.0, 1, 0],
        ], dtype=float)
        network_plotter = _NetworkPlotter()

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            input_path.write_text("example\n", encoding="utf-8")
            output_path = Path(directory) / "figure.png"
            outputs = FigureExporter().export(
                output_path,
                network_plotter,
                edges,
                pipeline="v3",
                input_path=input_path,
            )

            for path in outputs.values():
                self.assertTrue(Path(path).is_file())

            with Image.open(output_path) as exported:
                self.assertLess(exported.width, 100)
                self.assertLess(exported.height, 60)
                self.assertAlmostEqual(exported.info["dpi"][0], PUBLICATION_DPI,
                                       delta=0.1)

            metadata_path = Path(outputs["metadata"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["statistical_safeguard"],
                             SUBDIVISION_SAFEGUARD)
            self.assertEqual(metadata["rendered_groups"][0]["edge_count"], 2)
            self.assertEqual(
                metadata["rendered_groups"][0]["display_subdivision_id"], 1
            )
            self.assertEqual(
                metadata["rendered_groups"][0]["rendering"], "centroid"
            )
            self.assertFalse(
                metadata["rendered_groups"][0]["independently_tested"]
            )
            self.assertNotIn("source_p_values", metadata["rendered_groups"][0])
            self.assertEqual(
                metadata["inferential_context"]["parent_bundles"][0][
                    "source_parent_bundle_p_values"
                ],
                [0.012],
            )
            self.assertEqual(metadata["output"]["dpi"], PUBLICATION_DPI)
            self.assertTrue(metadata["anatomical_orientation_marker"]["visible"])
            self.assertEqual(len(metadata["output"]["sha256"]), 64)
            self.assertIn(
                SUBDIVISION_SAFEGUARD,
                Path(outputs["caption"]).read_text(encoding="utf-8"),
            )
            self.assertIn(
                "display subdivisions",
                Path(outputs["caption"]).read_text(encoding="utf-8"),
            )
            self.assertEqual(
                network_plotter.plotter.screenshot_kwargs["window_size"],
                (4200, 2520),
            )

    def test_two_parent_bundles_get_separate_inferential_entries(self):
        # Two independently FWER-significant bundles (NETWORK_COL 0 and 1,
        # distinct p-values) loaded together in one v3 edge array must never
        # be pooled into one aggregate inferential-context entry.
        edges = np.asarray([
            [0, 0, 0, 1, 1, 1, 0.042, 3.0, 0, 0],
            [0, 0, 1, 1, 1, 2, 0.042, 2.0, 1, 0],
            [5, 5, 5, 6, 6, 6, 0.045, -3.0, 0, 1],
        ], dtype=float)
        network_plotter = _NetworkPlotter()
        network_plotter.last_selection = [
            {"fcn": 0, "bundle": "All"}, {"fcn": 1, "bundle": "All"}
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "figure.png"
            outputs = FigureExporter().export(
                output_path, network_plotter, edges, pipeline="v3",
            )
            metadata = json.loads(
                Path(outputs["metadata"]).read_text(encoding="utf-8")
            )
            parents = metadata["inferential_context"]["parent_bundles"]
            self.assertEqual(len(parents), 2)
            by_id = {entry["parent_bundle_id"]: entry for entry in parents}
            self.assertEqual(
                by_id[0]["source_parent_bundle_p_values"], [0.042]
            )
            self.assertEqual(by_id[0]["complete_parent_bundle_edge_count"], 2)
            self.assertEqual(
                by_id[1]["source_parent_bundle_p_values"], [0.045]
            )
            self.assertEqual(by_id[1]["complete_parent_bundle_edge_count"], 1)

    def test_export_requires_rendered_edges(self):
        network_plotter = _NetworkPlotter()
        network_plotter._edge_actors = []
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Plot a selection"):
                FigureExporter().export(
                    Path(directory) / "figure.png",
                    network_plotter,
                    np.empty((0, 10)),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
