"""Publication-oriented export of the scene currently shown in the GUI."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
from PIL import Image

from coffee_dac_pipeline import BUNDLE_COL, NETWORK_COL
from mocca_gui.plotter import (
    ANATOMICAL_ORIENTATION_LABELS,
    CENTROID_OPACITY_MAX,
    CENTROID_OPACITY_MIN,
    CENTROID_WIDTH_MAX_ABS,
    CENTROID_WIDTH_MAX_MULT,
    CENTROID_WIDTH_MIN_MULT,
)


PUBLICATION_DPI = 600
PUBLICATION_WIDTH_INCHES = 7.0
RASTER_EXTENSIONS = {".png", ".tif", ".tiff"}
VECTOR_EXTENSIONS = {".pdf", ".svg"}
SUPPORTED_EXTENSIONS = RASTER_EXTENSIONS | VECTOR_EXTENSIONS

SUBDIVISION_SAFEGUARD = (
    "Colors identify post hoc visual subdivisions of the parent bundle. "
    "Statistical testing was performed on the complete parent bundle; the "
    "subdivisions were not tested independently."
)


def _jsonable(value):
    """Convert NumPy/VTK-friendly containers into strict JSON values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution):
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_commit():
    repository = Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "-C", os.fspath(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_json(path, payload):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_text(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _background_color(image):
    """Estimate a uniform renderer background from the image border."""
    rgb = image[..., :3].astype(np.uint8, copy=False)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    # Quantization keeps antialiasing/noise from splitting one background into
    # many nearly identical colors. The median of the winning bin restores a
    # representative unquantized value.
    quantized = border // 4
    colors, inverse, counts = np.unique(
        quantized, axis=0, return_inverse=True, return_counts=True
    )
    winning_bin = int(np.argmax(counts))
    return np.median(border[inverse == winning_bin], axis=0)


def trim_uniform_background(image, tolerance=6, padding_fraction=0.015):
    """Tightly crop a screenshot while retaining a small, even margin."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA screenshot array")

    background = _background_color(array)
    difference = np.max(
        np.abs(array[..., :3].astype(np.int16) - background.astype(np.int16)),
        axis=2,
    )
    foreground = difference > tolerance
    if array.shape[2] == 4:
        foreground &= array[..., 3] > 0
    rows, columns = np.nonzero(foreground)
    if not len(rows):
        return array

    padding = max(2, int(round(max(array.shape[:2]) * padding_fraction)))
    top = max(0, int(rows.min()) - padding)
    bottom = min(array.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(columns.min()) - padding)
    right = min(array.shape[1], int(columns.max()) + padding + 1)
    return array[top:bottom, left:right]


def _camera_metadata(plotter):
    camera = plotter.camera
    result = {
        "position_focal_point_view_up": [
            list(vector) for vector in plotter.camera_position
        ],
    }
    vtk_getters = {
        "position": "GetPosition",
        "focal_point": "GetFocalPoint",
        "view_up": "GetViewUp",
        "view_angle_degrees": "GetViewAngle",
        "parallel_projection": "GetParallelProjection",
        "parallel_scale": "GetParallelScale",
        "clipping_range": "GetClippingRange",
    }
    for name, getter in vtk_getters.items():
        method = getattr(camera, getter, None)
        if method is not None:
            result[name] = method()
    return result


def _direction(t_statistics):
    finite = np.asarray(t_statistics, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return "unavailable"
    if np.all(finite > 0):
        return "positive"
    if np.all(finite < 0):
        return "negative"
    if np.all(finite == 0):
        return "zero"
    return "mixed"


def _expanded_rendered_groups(edges_net, selection, network_plotter, pipeline):
    groups = []
    seen = set()
    for item in selection:
        fcn = int(item["fcn"])
        if item["bundle"] == "All":
            bundle_ids = np.unique(
                edges_net[edges_net[:, NETWORK_COL] == fcn, BUNDLE_COL]
            ).astype(int)
        else:
            bundle_ids = [int(item["bundle"])]

        for bundle in bundle_ids:
            key = (fcn, int(bundle))
            if key in seen:
                continue
            seen.add(key)
            rows = edges_net[
                (edges_net[:, NETWORK_COL] == fcn)
                & (edges_net[:, BUNDLE_COL] == bundle)
            ]
            t_statistics = rows[:, 7] if rows.shape[1] > 7 else []
            color = network_plotter.resolve_bundle_color(fcn, bundle)
            color = tuple(color.tolist()) if hasattr(color, "tolist") else tuple(color)

            group = {
                "network_id": fcn,
                "edge_count": int(len(rows)),
                "direction_from_t_statistics": _direction(t_statistics),
                "rendering": (
                    "centroid" if network_plotter.centroid_flags.get(key, False)
                    else "full_edges"
                ),
                "rgba": color,
                "line_thickness": network_plotter.thicknesses.get(key, 3),
                "curvature": network_plotter.curvatures.get(key, 1.0),
                "opacity": network_plotter.opacities.get(key, 0.8),
                "endpoint_size": network_plotter.endpoint_sizes.get(key, 1.0),
            }
            if pipeline == "v3":
                group["display_subdivision_id"] = int(bundle)
                group["independently_tested"] = False
            else:
                group["bundle_id"] = int(bundle)
                p_values = []
                if rows.shape[1] > 6:
                    p_values = sorted({
                        float(value) for value in rows[:, 6]
                        if np.isfinite(float(value))
                    })
                group["source_p_values"] = p_values
            groups.append(group)
    return groups


def _inferential_context(edges_net, pipeline):
    """Keep parent-level inference separate from descriptive v3 groups.

    A single loaded v3 edge array can legitimately span more than one
    parent (inferential) bundle -- e.g. two bundles both survive FWER at
    alpha=0.05 in the same visualization export (see
    coffee_dac_pipeline_v3.py). Each parent bundle's edges are identified by
    its own NETWORK_COL id and get their own entry here; they must never be
    pooled into one aggregate p-value/edge-count, since that would blur two
    independent statistical findings into what looks like one.
    """
    if pipeline != "v3":
        return None
    parent_ids = (
        sorted(np.unique(edges_net[:, NETWORK_COL]).astype(int).tolist())
        if len(edges_net) else []
    )
    parent_bundles = []
    for parent_id in parent_ids:
        rows = edges_net[edges_net[:, NETWORK_COL] == parent_id]
        p_values = []
        if rows.shape[1] > 6:
            p_values = sorted({
                float(value) for value in rows[:, 6]
                if np.isfinite(float(value))
            })
        t_statistics = rows[:, 7] if rows.shape[1] > 7 else []
        parent_bundles.append({
            "unit": "complete parent bundle",
            "parent_bundle_id": parent_id,
            "complete_parent_bundle_edge_count": int(len(rows)),
            "direction_from_t_statistics": _direction(t_statistics),
            "source_parent_bundle_p_values": p_values,
            "mass": None,
            "mass_note": "Bundle mass is not present in the loaded GUI edge array.",
            "display_subdivisions_independently_tested": False,
        })
    return {
        "parent_bundle_identifier_note": (
            "parent_bundle_id is this array's NETWORK_COL value, not "
            "necessarily the upstream FWER bundle table's id; cross-"
            "reference the input filename and upstream manifest for that."
        ),
        "parent_bundles": parent_bundles,
    }


def _brain_mesh_metadata(network_plotter):
    meshes = []
    live_actors = getattr(network_plotter, "_live_brain_actors", [])
    for index, entry in enumerate(network_plotter._brain_meshes):
        mesh_path = Path(entry[0]).expanduser().resolve()
        actual_opacity = None
        if index < len(live_actors):
            actor = live_actors[index][0]
            actual_opacity = actor.GetProperty().GetOpacity()
        meshes.append({
            "path": os.fspath(mesh_path),
            "sha256": _sha256(mesh_path) if mesh_path.is_file() else None,
            "base_opacity": entry[1],
            "color": entry[2],
            "smooth_shading": entry[3] if len(entry) > 3 else False,
            "lighting": True,
            "render_kwargs": entry[4] if len(entry) > 4 else {},
            "is_white_matter": entry[5] if len(entry) > 5 else False,
            "rendered_opacity": actual_opacity,
        })
    return meshes


def _orientation_marker_visible(network_plotter):
    widget = getattr(network_plotter, "_orientation_widget", None)
    if widget is None:
        return False
    get_enabled = getattr(widget, "GetEnabled", None)
    return bool(get_enabled()) if get_enabled is not None else True


class FigureExporter:
    """Export the live renderer with journal-friendly defaults and provenance."""

    def export(self, filename, network_plotter, edges_net, pipeline=None,
               input_path=None):
        output_path = Path(filename).expanduser().resolve()
        extension = output_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                "Figure filename must end in .png, .tif, .tiff, .pdf, or .svg"
            )
        if not network_plotter._edge_actors or not network_plotter.last_selection:
            raise ValueError("Plot a selection before exporting a figure")

        plotter = network_plotter.plotter
        selection = list(network_plotter.last_selection)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if extension in RASTER_EXTENSIONS:
            image_details = self._export_raster(plotter, output_path, extension)
        else:
            image_details = self._export_vector(plotter, output_path, extension)

        caption = self._caption(selection, pipeline)
        caption_path = output_path.with_name(
            f"{output_path.stem}_caption.txt"
        )
        metadata_path = output_path.with_name(
            f"{output_path.stem}_publication_metadata.json"
        )
        _atomic_text(caption_path, caption)

        input_record = None
        if input_path:
            input_file = Path(input_path).expanduser().resolve()
            input_record = {"path": os.fspath(input_file)}
            if input_file.is_file():
                input_record["sha256"] = _sha256(input_file)

        groups = _expanded_rendered_groups(
            edges_net, selection, network_plotter, pipeline
        )
        metadata = {
            "schema_version": 1,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "export_type": "current_view",
            "output": {
                "path": os.fspath(output_path),
                "sha256": _sha256(output_path),
                **image_details,
            },
            "input": input_record,
            "pipeline": pipeline,
            "camera": _camera_metadata(plotter),
            "rendered_selection": selection,
            "rendered_groups": groups,
            "inferential_context": _inferential_context(edges_net, pipeline),
            "endpoint_glyphs_visible": network_plotter.last_endpoint_visible,
            "anatomical_orientation_marker": {
                "visible": _orientation_marker_visible(network_plotter),
                "coordinate_frame": "brain3mm voxel-index (LAS)",
                "face_labels": ANATOMICAL_ORIENTATION_LABELS,
            },
            "brain_meshes": _brain_mesh_metadata(network_plotter),
            "brain_opacity_scale": network_plotter.brain_opacity_scale,
            "white_matter_visible": network_plotter.wm_visible,
            "centroid_line_width_mapping": {
                "quantity": "constituent edge count",
                "normalization": "log1p within the rendered view",
                "width_multiplier_range": [
                    CENTROID_WIDTH_MIN_MULT, CENTROID_WIDTH_MAX_MULT
                ],
                "absolute_width_cap": CENTROID_WIDTH_MAX_ABS,
                "opacity_range": [CENTROID_OPACITY_MIN, CENTROID_OPACITY_MAX],
                "encodes_significance": False,
            },
            "statistical_safeguard": (
                SUBDIVISION_SAFEGUARD if pipeline == "v3" else None
            ),
            "software": {
                "python": platform.python_version(),
                "pyvista": _package_version("pyvista"),
                "vtk": _package_version("vtk"),
                "pillow": _package_version("pillow"),
                "git_commit": _git_commit(),
            },
            "caption_file": {
                "path": os.fspath(caption_path),
                "sha256": _sha256(caption_path),
            },
        }
        _atomic_json(metadata_path, metadata)
        return {
            "figure": os.fspath(output_path),
            "caption": os.fspath(caption_path),
            "metadata": os.fspath(metadata_path),
        }

    @staticmethod
    def _export_raster(plotter, output_path, extension):
        width = int(round(PUBLICATION_DPI * PUBLICATION_WIDTH_INCHES))
        window_width, window_height = [int(value) for value in plotter.window_size]
        if window_width <= 0 or window_height <= 0:
            raise RuntimeError("The renderer has no valid window size")
        height = max(1, int(round(width * window_height / window_width)))
        screenshot = plotter.screenshot(
            return_img=True,
            window_size=(width, height),
            transparent_background=False,
        )
        cropped = trim_uniform_background(screenshot)
        image = Image.fromarray(np.asarray(cropped, dtype=np.uint8))

        fd, temporary = tempfile.mkstemp(
            prefix=f".{output_path.stem}.", suffix=extension,
            dir=output_path.parent,
        )
        os.close(fd)
        try:
            if extension == ".png":
                image.save(temporary, format="PNG", dpi=(PUBLICATION_DPI,) * 2,
                           optimize=True)
            else:
                image.save(temporary, format="TIFF", dpi=(PUBLICATION_DPI,) * 2,
                           compression="tiff_lzw")
            os.replace(temporary, output_path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

        final_width, final_height = image.size
        return {
            "format": extension.lstrip("."),
            "pixel_size": [final_width, final_height],
            "dpi": PUBLICATION_DPI,
            "print_size_inches": [
                final_width / PUBLICATION_DPI,
                final_height / PUBLICATION_DPI,
            ],
            "tightly_cropped": True,
        }

    @staticmethod
    def _export_vector(plotter, output_path, extension):
        fd, temporary = tempfile.mkstemp(
            prefix=f".{output_path.stem}.", suffix=extension,
            dir=output_path.parent,
        )
        os.close(fd)
        try:
            plotter.save_graphic(
                temporary,
                title="COFFEE-DAC publication figure",
                raster=True,
                painter=True,
            )
            os.replace(temporary, output_path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return {
            "format": extension.lstrip("."),
            "vector_container": True,
            "three_dimensional_props_rasterized": True,
            "text_and_overlays_vector_when_supported": True,
            "tightly_cropped": False,
        }

    @staticmethod
    def _caption(selection, pipeline):
        selected_parts = []
        for item in selection:
            if pipeline == "v3":
                group = (
                    "all display subdivisions" if item["bundle"] == "All"
                    else f"display subdivision {int(item['bundle'])}"
                )
            else:
                group = (
                    "all displayed bundles" if item["bundle"] == "All"
                    else f"bundle {int(item['bundle'])}"
                )
            selected_parts.append(f"network {int(item['fcn'])}, {group}")
        selected = ", ".join(selected_parts)
        lines = [
            "COFFEE-DAC connectivity rendering from the interactively selected "
            f"camera view ({selected or 'current rendered selection'}).",
            "Centroid line width, where enabled, encodes constituent edge count "
            "and does not encode statistical significance.",
        ]
        if pipeline == "v3":
            lines.append(SUBDIVISION_SAFEGUARD)
        lines.append(
            "Camera, rendering settings, group edge counts, source p-values, "
            "checksums, and software versions are recorded in the accompanying "
            "publication metadata JSON file."
        )
        return "\n\n".join(lines)
