"""Reproducible multi-panel publication export for COFFEE-DAC."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

from coffee_dac_pipeline import BUNDLE_COL, NETWORK_COL
from mocca_gui.figure_exporter import (
    PUBLICATION_DPI,
    SUBDIVISION_SAFEGUARD,
    _brain_mesh_metadata,
    _git_commit,
    _jsonable,
    _package_version,
    _sha256,
)
from mocca_gui.plotter import (
    ANATOMICAL_ORIENTATION_LABELS,
    CENTROID_OPACITY_MAX,
    CENTROID_OPACITY_MIN,
    CENTROID_WIDTH_MAX_ABS,
    CENTROID_WIDTH_MAX_MULT,
    CENTROID_WIDTH_MIN_MULT,
    NetworkPlotter,
)


FIGURE_WIDTH_PX = 4200
PANEL_SIZE_PX = 990
ROW_HEADER_PX = 150
PANEL_GAP_PX = 30
ROW_GAP_PX = 24

# Okabe-Ito: a widely used color-vision-deficiency-safe qualitative palette.
# Yellow is placed late because it has low contrast on the white background.
OKABE_ITO = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
    "#F0E442",  # yellow
)

COLOR_MODE_GUI = "gui"
COLOR_MODE_PUBLICATION = "publication"
COLOR_MODES = {COLOR_MODE_GUI, COLOR_MODE_PUBLICATION}


class ExportCancelled(RuntimeError):
    """Raised when the GUI cancellation flag is set during an export."""


@dataclass
class ParentBundle:
    identifier: str
    slug: str
    network_id: int | None
    display_bundle_id: int | None
    edges: np.ndarray
    selection: list[dict]
    summary: dict
    colors: dict


def endpoint_incidence(edges):
    """Return ``(i, j, k, raw_count)`` for every incident endpoint voxel."""
    if len(edges) == 0:
        return np.empty((0, 4), dtype=np.int64)
    endpoints = np.rint(
        np.vstack((edges[:, 0:3], edges[:, 3:6]))
    ).astype(np.int64)
    voxels, counts = np.unique(endpoints, axis=0, return_counts=True)
    return np.column_stack((voxels, counts)).astype(np.int64, copy=False)


def _direction(t_statistics):
    values = np.asarray(t_statistics, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return "unavailable"
    if np.all(values > 0):
        return "positive"
    if np.all(values < 0):
        return "negative"
    if np.all(values == 0):
        return "zero"
    return "mixed"


def _safe_slug(value):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return slug or "unknown"


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _inference_manifest(input_path, provenance):
    if isinstance(provenance, dict) and provenance.get("inference"):
        return provenance
    if input_path:
        source = Path(input_path).expanduser().resolve()
        candidate = source.with_name(f"{source.stem}_v2_params.json")
        manifest = _read_json(candidate)
        if isinstance(manifest, dict) and manifest.get("inference"):
            return manifest
    return None


def _ordered_source_records(manifest):
    if not manifest:
        return []
    records = manifest.get("inference", {}).get("selected_source_bundles", [])

    def sort_key(record):
        return (
            -int(record.get("sign", 0)),
            -int(record.get("edge_count", 0)),
            int(record.get("bundle", 0)),
        )

    return sorted((dict(record) for record in records), key=sort_key)


def _source_record_p_value(record):
    for name in ("p_grid_fwer", "p_fwer"):
        if name in record and record[name] is not None:
            return float(record[name]), name
    return None, None


def _selection_keys(edges_net, selection):
    keys = []
    seen = set()
    for item in selection:
        fcn = int(item["fcn"])
        if item["bundle"] == "All":
            bundles = np.unique(
                edges_net[edges_net[:, NETWORK_COL] == fcn, BUNDLE_COL]
            ).astype(int)
        else:
            bundles = [int(item["bundle"])]
        for bundle in bundles:
            key = (fcn, int(bundle))
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _rgba(hex_color):
    text = hex_color.lstrip("#")
    return tuple(int(text[index:index + 2], 16) / 255.0 for index in (0, 2, 4)) + (1.0,)


def _subdivision_palette(group_keys):
    keys = sorted(set(group_keys))
    if len(keys) <= len(OKABE_ITO):
        colors = [_rgba(color) for color in OKABE_ITO[:len(keys)]]
    else:
        from matplotlib import colormaps
        colors = [
            tuple(colormaps["viridis"](value))
            for value in np.linspace(0.08, 0.92, len(keys))
        ]
    return dict(zip(keys, colors))


def _gui_palette(group_keys, network_plotter):
    """Freeze the live GUI's resolved color for each rendered group."""
    resolver = getattr(network_plotter, "resolve_bundle_color", None)
    if not callable(resolver):
        raise ValueError(
            "GUI color mode requires a plotter with resolve_bundle_color()"
        )
    colors = {}
    for fcn, bundle in sorted(set(group_keys)):
        color = resolver(int(fcn), int(bundle))
        if hasattr(color, "tolist"):
            color = color.tolist()
        color = tuple(float(channel) for channel in color)
        if len(color) == 3:
            color += (1.0,)
        if len(color) != 4:
            raise ValueError(
                f"Resolved GUI color for ({fcn}, {bundle}) is not RGB/RGBA"
            )
        colors[(int(fcn), int(bundle))] = color
    return colors


def _effect_label(record, manifest, fallback):
    if not record:
        return fallback
    sign = int(record.get("sign", 0))
    inference = (manifest or {}).get("inference", {})
    if sign > 0:
        return inference.get("positive_effect") or "positive"
    if sign < 0:
        return inference.get("negative_effect") or "negative"
    return fallback


def _summary_for_rows(rows, record, manifest, identifier, network_id,
                      display_bundle_id, subdivision_count):
    fallback_direction = _direction(rows[:, 7] if rows.shape[1] > 7 else [])
    p_value, p_field = _source_record_p_value(record or {})
    source_p_value = p_value
    if source_p_value is None and rows.shape[1] > 6:
        unique = np.unique(rows[:, 6].astype(float))
        finite = unique[np.isfinite(unique)]
        if len(finite) == 1:
            source_p_value = float(finite[0])
            p_field = "source_pvalue_column"

    inference = (manifest or {}).get("inference", {})
    alpha = inference.get("alpha")
    status = "unavailable"
    if p_value is not None and alpha is not None:
        status = "passes recorded alpha" if p_value <= float(alpha) else "exploratory"
    elif source_p_value is not None:
        status = "source p-value present; correction provenance unavailable"

    return {
        "parent_bundle_id": identifier,
        "network_id": network_id,
        "display_bundle_id": display_bundle_id,
        "direction": _effect_label(record, manifest, fallback_direction),
        "edge_count": int(len(rows)),
        "mass": (float(record["mass"]) if record and record.get("mass") is not None else None),
        "source_p_value": source_p_value,
        "fwer_corrected_p": p_value,
        "p_value_source": p_field,
        "correction": inference.get("correction"),
        "alpha": alpha,
        "inferential_status": status,
        "display_subdivision_count": int(subdivision_count),
    }


def resolve_parent_bundles(edges_net, selection, pipeline, input_path=None,
                           provenance=None, network_plotter=None,
                           color_mode=COLOR_MODE_PUBLICATION):
    """Resolve rendered data into the roadmap's parent/inferential units."""
    if color_mode not in COLOR_MODES:
        raise ValueError(
            f"Unknown color mode {color_mode!r}; expected one of "
            f"{sorted(COLOR_MODES)}"
        )
    manifest = _inference_manifest(input_path, provenance)
    records = _ordered_source_records(manifest)

    if pipeline == "v3":
        # v3 divides each parent (inferential) bundle into post hoc display
        # subdivisions independently -- a single loaded v3 array can
        # legitimately contain more than one parent bundle at once (e.g. two
        # bundles both survive FWER at alpha=0.05 in the same visualization
        # export; see coffee_dac_pipeline_v3.py). Each parent bundle is its
        # own NETWORK_COL group and gets its own ParentBundle here, never
        # pooled with another parent's edges, p-value, or subdivisions. The
        # standardized export always restores every subdivision within a
        # parent, even when the interactive view showed only a subset.
        network_ids = sorted(np.unique(edges_net[:, NETWORK_COL]).astype(int))
        parents = []
        for network_id in network_ids:
            rows = edges_net[edges_net[:, NETWORK_COL] == network_id]
            record = (
                records[network_id] if 0 <= network_id < len(records) else None
            )
            identifier = str(record.get("bundle")) if record else (
                f"{Path(input_path).stem}_parent{network_id}" if input_path
                else f"loaded_parent_{network_id}"
            )
            group_keys = [
                (int(row[NETWORK_COL]), int(row[BUNDLE_COL])) for row in rows
            ]
            colors = (
                _gui_palette(group_keys, network_plotter)
                if color_mode == COLOR_MODE_GUI
                else _subdivision_palette(group_keys)
            )
            summary = _summary_for_rows(
                rows, record, manifest, identifier, network_id, None,
                len(set(group_keys)),
            )
            parents.append(ParentBundle(
                identifier=identifier,
                slug=_safe_slug(identifier),
                network_id=summary["network_id"],
                display_bundle_id=None,
                edges=rows,
                selection=[{"fcn": network_id, "bundle": "All"}],
                summary=summary,
                colors=colors,
            ))
        return parents, manifest

    parents = []
    for fcn, bundle in _selection_keys(edges_net, selection):
        rows = edges_net[
            (edges_net[:, NETWORK_COL] == fcn)
            & (edges_net[:, BUNDLE_COL] == bundle)
        ]
        record = records[bundle] if 0 <= bundle < len(records) else None
        identifier = str(record.get("bundle")) if record else str(bundle)
        fallback = _direction(rows[:, 7] if rows.shape[1] > 7 else [])
        direction = _effect_label(record, manifest, fallback)
        is_negative = (
            int(record.get("sign", 0)) < 0 if record
            else fallback == "negative"
        )
        key = (fcn, bundle)
        color = (
            _gui_palette([key], network_plotter)[key]
            if color_mode == COLOR_MODE_GUI
            else _rgba(OKABE_ITO[1] if is_negative else OKABE_ITO[0])
        )
        summary = _summary_for_rows(
            rows, record, manifest, identifier, fcn, bundle, 0
        )
        parents.append(ParentBundle(
            identifier=identifier,
            slug=_safe_slug(identifier),
            network_id=fcn,
            display_bundle_id=bundle,
            edges=rows,
            selection=[{"fcn": fcn, "bundle": bundle}],
            summary=summary,
            colors={key: color},
        ))
    return parents, manifest


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _save_png(path, image, dpi=PUBLICATION_DPI):
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    try:
        image.convert("RGB").save(
            temporary, format="PNG", dpi=(dpi, dpi), optimize=True
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.csv")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_render_state(source, target):
    for name in (
        "thicknesses", "curvatures", "opacities", "endpoint_sizes",
        "bundle_colors", "bundle_rgba_overrides",
    ):
        setattr(target, name, dict(getattr(source, name, {})))
    target.wm_visible = source.wm_visible
    target.brain_opacity_scale = source.brain_opacity_scale
    for index, live in enumerate(getattr(source, "_live_brain_actors", [])):
        if index < len(target._live_brain_actors):
            opacity = live[0].GetProperty().GetOpacity()
            target._live_brain_actors[index][0].GetProperty().SetOpacity(opacity)


def _standard_camera_states(plotter, bounds=None):
    bounds = np.asarray(plotter.bounds if bounds is None else bounds, dtype=float)
    center = np.asarray([
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    ])
    extent = max(
        bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
    )
    distance = max(extent * 4.0, 1.0)
    scale = max(extent * 0.54, 1.0)
    return {
        "left_lateral": {
            "position": center + (distance, 0, 0),
            "focal_point": center,
            "view_up": (0, 0, 1),
            "parallel_scale": scale,
        },
        "superior_dorsal": {
            "position": center + (0, 0, distance),
            "focal_point": center,
            "view_up": (0, 1, 0),
            "parallel_scale": scale,
        },
        "right_lateral": {
            "position": center - (distance, 0, 0),
            "focal_point": center,
            "view_up": (0, 0, 1),
            "parallel_scale": scale,
        },
    }


def _capture_views(plotter, size=PANEL_SIZE_PX, stop_flag=None, bounds=None):
    images = {}
    metadata = {}
    for name, state in _standard_camera_states(plotter, bounds=bounds).items():
        if stop_flag and stop_flag():
            raise ExportCancelled("Publication export cancelled")
        plotter.camera_position = [
            state["position"], state["focal_point"], state["view_up"]
        ]
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = state["parallel_scale"]
        plotter.reset_camera_clipping_range()
        plotter.render()
        images[name] = Image.fromarray(np.asarray(plotter.screenshot(
            return_img=True,
            window_size=(size, size),
            transparent_background=False,
        ), dtype=np.uint8)).convert("RGB")
        metadata[name] = _jsonable(state)
    return images, metadata


def _brain_bounds(renderer):
    meshes = [entry[0] for entry in renderer._brain_mesh_actors]
    if not meshes:
        return None
    all_bounds = np.asarray([mesh.bounds for mesh in meshes], dtype=float)
    return (
        float(np.min(all_bounds[:, 0])), float(np.max(all_bounds[:, 1])),
        float(np.min(all_bounds[:, 2])), float(np.max(all_bounds[:, 3])),
        float(np.min(all_bounds[:, 4])), float(np.max(all_bounds[:, 5])),
    )


def _render_brain_views(parent, live_plotter, centroids,
                        centroid_count_range=None, stop_flag=None):
    plotter = pv.Plotter(off_screen=True, window_size=(PANEL_SIZE_PX, PANEL_SIZE_PX))
    renderer = NetworkPlotter(plotter, brain_meshes=live_plotter._brain_meshes)
    try:
        _copy_render_state(live_plotter, renderer)
        renderer.centroid_count_range = centroid_count_range
        renderer.bundle_rgba_overrides.update(parent.colors)
        for row in parent.edges:
            key = (int(row[NETWORK_COL]), int(row[BUNDLE_COL]))
            renderer.centroid_flags[key] = bool(centroids)
            renderer.thicknesses[key] = 3.0
            renderer.curvatures[key] = 1.0
            renderer.opacities[key] = 0.85
            renderer.endpoint_sizes[key] = 1.0
        completed = renderer.draw_selection(
            parent.edges,
            parent.selection,
            endpoint_visible=False,
            stop_flag=stop_flag,
        )
        if not completed:
            raise ExportCancelled("Publication export cancelled")
        return _capture_views(
            plotter, stop_flag=stop_flag, bounds=_brain_bounds(renderer)
        )
    finally:
        plotter.close()


@lru_cache(maxsize=1)
def _template_mip():
    template_path = Path(__file__).resolve().parents[2] / "templates" / "brain3mm.nii"
    try:
        import nibabel as nib
        volume = nib.load(template_path).get_fdata(dtype=np.float32)
        mip = np.max(volume, axis=2).T
        positive = mip[mip > 0]
        if len(positive):
            low, high = np.percentile(positive, (2, 99))
            mip = np.clip((mip - low) / max(high - low, 1e-6), 0, 1)
        return mip, os.fspath(template_path), _sha256(template_path)
    except (OSError, ValueError, ImportError):
        return None, None, None


def _endpoint_density_image(edges, size):
    incidence = endpoint_incidence(edges)
    maximums = np.max(incidence[:, :3], axis=0) + 1 if len(incidence) else (1, 1, 1)
    shape = tuple(max(int(value), 1) for value in maximums)
    anatomical, template_path, template_sha = _template_mip()
    if anatomical is None:
        anatomical = np.zeros((shape[1], shape[0]), dtype=np.float32)
    shape = (
        max(shape[0], anatomical.shape[1]),
        max(shape[1], anatomical.shape[0]),
        shape[2],
    )
    counts = np.zeros(shape, dtype=np.int64)
    for i, j, k, count in incidence:
        if i >= 0 and j >= 0 and k >= 0:
            counts[i, j, k] = count
    projection = np.max(counts, axis=2).T

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(
        figsize=(size / PUBLICATION_DPI, size / PUBLICATION_DPI),
        dpi=PUBLICATION_DPI,
        facecolor="white",
    )
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0.08, 0.08, 0.73, 0.84))
    axes.imshow(
        anatomical,
        origin="lower",
        extent=(-0.5, shape[0] - 0.5, -0.5, shape[1] - 0.5),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )
    masked = np.ma.masked_where(projection == 0, projection)
    maximum_count = max(int(projection.max()), 1)
    heatmap = axes.imshow(
        masked,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        vmin=0.5,
        vmax=maximum_count + 0.5,
        alpha=0.92,
    )
    axes.set_xticks([])
    axes.set_yticks([])
    for label, x, y, horizontal, vertical in (
        ("R", 0.01, 0.50, "left", "center"),
        ("L", 0.99, 0.50, "right", "center"),
        ("P", 0.50, 0.01, "center", "bottom"),
        ("A", 0.50, 0.99, "center", "top"),
    ):
        axes.text(
            x, y, label, transform=axes.transAxes, ha=horizontal, va=vertical,
            fontsize=9, fontweight="bold", color="white",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.65,
                  "boxstyle": "round,pad=0.14"},
        )
    colorbar_axes = figure.add_axes((0.84, 0.17, 0.035, 0.66))
    colorbar = figure.colorbar(heatmap, cax=colorbar_axes)
    if maximum_count <= 8:
        colorbar.set_ticks(range(1, maximum_count + 1))
    else:
        from matplotlib.ticker import MaxNLocator
        colorbar.locator = MaxNLocator(nbins=6, integer=True)
        colorbar.update_ticks()
    colorbar.ax.set_title("n", fontsize=7, pad=2)
    colorbar.ax.tick_params(labelsize=6, length=2)
    canvas.draw()
    image = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
    return image, incidence, {
        "representation": "anatomical maximum-intensity projection",
        "projection_axis": "superior-inferior (K)",
        "values": "raw endpoint-voxel incidence counts",
        "normalization": "none",
        "colormap": "viridis",
        "anatomical_template": template_path,
        "anatomical_template_sha256": template_sha,
        "maximum_raw_count": int(projection.max()) if projection.size else 0,
    }


def _panel_label(image, text, letter):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = _font(52, bold=True)
    label = f"{letter}  {text}"
    box = draw.textbbox((0, 0), label, font=font)
    width = box[2] - box[0] + 32
    height = box[3] - box[1] + 24
    draw.rounded_rectangle((18, 18, 18 + width, 18 + height), radius=12,
                           fill=(255, 255, 255), outline=(35, 35, 35), width=2)
    draw.text((34, 26), label, font=font, fill=(20, 20, 20))
    return image


def _summary_label(summary):
    parts = [
        f"Parent bundle {summary['parent_bundle_id']}",
        str(summary["direction"]),
        f"n = {summary['edge_count']} edges",
    ]
    if summary.get("mass") is not None:
        parts.append(f"mass = {summary['mass']:.4g}")
    if summary.get("fwer_corrected_p") is not None:
        parts.append(f"FWER p = {summary['fwer_corrected_p']:.4g}")
    return "   |   ".join(parts)


def _assemble_row(panels, summary, titles):
    width = FIGURE_WIDTH_PX
    height = ROW_HEADER_PX + PANEL_SIZE_PX + 25
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    summary_text = _summary_label(summary)
    summary_font_size = 54
    summary_font = _font(summary_font_size, bold=True)
    while (
        draw.textbbox((0, 0), summary_text, font=summary_font)[2] > width - 70
        and summary_font_size > 30
    ):
        summary_font_size -= 2
        summary_font = _font(summary_font_size, bold=True)
    draw.text((35, 35), summary_text, font=summary_font, fill=(20, 20, 20))
    occupied_width = len(panels) * PANEL_SIZE_PX + max(len(panels) - 1, 0) * PANEL_GAP_PX
    x = max((width - occupied_width) // 2, 0)
    letters = "ABCD"
    for index, (panel, title) in enumerate(zip(panels, titles)):
        prepared = panel.resize((PANEL_SIZE_PX, PANEL_SIZE_PX), Image.Resampling.LANCZOS)
        prepared = _panel_label(prepared, title, letters[index])
        canvas.paste(prepared, (x, ROW_HEADER_PX))
        x += PANEL_SIZE_PX + PANEL_GAP_PX
    return canvas


def _stack_rows(rows):
    height = sum(row.height for row in rows) + ROW_GAP_PX * max(len(rows) - 1, 0)
    image = Image.new("RGB", (FIGURE_WIDTH_PX, height), "white")
    y = 0
    for row in rows:
        image.paste(row, (0, y))
        y += row.height + ROW_GAP_PX
    return image


def _caption(parent, pipeline):
    summary = parent.summary
    text = (
        f"Parent bundle {summary['parent_bundle_id']} ({summary['direction']}; "
        f"{summary['edge_count']} edges). Left-lateral, superior/dorsal, and "
        "right-lateral parallel-projection views use identical scale, lighting, "
        "surface opacity, and camera distance. Centroid line width represents "
        "constituent edge count, not significance. The endpoint panel is a "
        "superior-inferior anatomical maximum-intensity projection of raw "
        "endpoint-voxel incidence counts. A separate full-edge triptych is "
        "provided for transparency."
    )
    if summary.get("mass") is not None:
        text += f" Bundle mass was {summary['mass']:.4g}."
    if summary.get("fwer_corrected_p") is not None:
        text += f" The source FWER-corrected p-value was {summary['fwer_corrected_p']:.4g}."
    if pipeline == "v3":
        text += " " + SUBDIVISION_SAFEGUARD
    return text


class PublicationExporter:
    """Create the standardized static subset of the publication roadmap."""

    def export(self, output_parent, network_plotter, edges_net, pipeline=None,
               input_path=None, provenance=None, progress_callback=None,
               stop_flag=None, color_mode=COLOR_MODE_PUBLICATION):
        if not network_plotter._edge_actors or not network_plotter.last_selection:
            raise ValueError("Plot a selection before exporting a publication set")
        parents, inference_manifest = resolve_parent_bundles(
            edges_net,
            network_plotter.last_selection,
            pipeline,
            input_path=input_path,
            provenance=provenance,
            network_plotter=network_plotter,
            color_mode=color_mode,
        )
        if not parents:
            raise ValueError("The rendered selection contains no parent bundles")

        displayed_group_counts = []
        for parent in parents:
            for fcn, bundle in sorted(set(
                (int(row[NETWORK_COL]), int(row[BUNDLE_COL]))
                for row in parent.edges
            )):
                displayed_group_counts.append(int(np.sum(
                    (parent.edges[:, NETWORK_COL] == fcn)
                    & (parent.edges[:, BUNDLE_COL] == bundle)
                )))
        centroid_count_range = (
            min(displayed_group_counts), max(displayed_group_counts)
        )

        output_parent = Path(output_parent).expanduser().resolve()
        output_parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        final_directory = output_parent / f"publication_export_{stamp}"
        counter = 2
        while final_directory.exists():
            final_directory = output_parent / f"publication_export_{stamp}_{counter}"
            counter += 1

        with tempfile.TemporaryDirectory(
            prefix=".publication_export_", dir=output_parent
        ) as temporary_directory:
            root = Path(temporary_directory)
            row_images = []
            output_records = []
            parent_records = []
            captions = []
            total_steps = max(len(parents) * 2 + 3, 1)
            completed_steps = 0

            def advance():
                nonlocal completed_steps
                completed_steps += 1
                if progress_callback:
                    progress_callback(min(100, int(100 * completed_steps / total_steps)))
                if stop_flag and stop_flag():
                    raise ExportCancelled("Publication export cancelled")

            for parent in parents:
                primary_views, camera_states = _render_brain_views(
                    parent,
                    network_plotter,
                    centroids=True,
                    centroid_count_range=centroid_count_range,
                    stop_flag=stop_flag,
                )
                advance()
                endpoint_panel, incidence, density_metadata = _endpoint_density_image(
                    parent.edges, PANEL_SIZE_PX
                )
                endpoint_image, _, _ = _endpoint_density_image(
                    parent.edges, PANEL_SIZE_PX * 2
                )
                primary_row = _assemble_row(
                    [
                        primary_views["left_lateral"],
                        primary_views["superior_dorsal"],
                        primary_views["right_lateral"],
                        endpoint_panel,
                    ],
                    parent.summary,
                    ("Left lateral", "Superior / dorsal", "Right lateral", "Endpoint density"),
                )
                views_name = f"parent_bundle_{parent.slug}_views.png"
                _save_png(root / views_name, primary_row)
                row_images.append(primary_row)

                endpoints_name = f"parent_bundle_{parent.slug}_endpoints.png"
                _save_png(root / endpoints_name, endpoint_image)
                density_csv_name = f"parent_bundle_{parent.slug}_endpoint_density.csv"
                _write_csv(
                    root / density_csv_name,
                    [
                        {"i": int(row[0]), "j": int(row[1]), "k": int(row[2]),
                         "raw_incident_edge_count": int(row[3])}
                        for row in incidence
                    ],
                    ("i", "j", "k", "raw_incident_edge_count"),
                )

                full_views, full_camera_states = _render_brain_views(
                    parent,
                    network_plotter,
                    centroids=False,
                    centroid_count_range=centroid_count_range,
                    stop_flag=stop_flag,
                )
                full_row = _assemble_row(
                    [
                        full_views["left_lateral"],
                        full_views["superior_dorsal"],
                        full_views["right_lateral"],
                    ],
                    parent.summary,
                    ("Full edges — left", "Full edges — dorsal", "Full edges — right"),
                )
                full_name = f"parent_bundle_{parent.slug}_full_edges.png"
                _save_png(root / full_name, full_row)
                advance()

                names = (views_name, endpoints_name, density_csv_name, full_name)
                for name in names:
                    output_records.append({
                        "path": name,
                        "sha256": _sha256(root / name),
                    })
                captions.append(_caption(parent, pipeline))
                parent_records.append({
                    "summary": parent.summary,
                    "colors": {
                        f"network_{key[0]}_display_group_{key[1]}": value
                        for key, value in parent.colors.items()
                    },
                    "primary_rendering": "centroids/skeletons",
                    "supplementary_rendering": "full edges",
                    "camera_states": camera_states,
                    "full_edge_camera_states": full_camera_states,
                    "endpoint_density": density_metadata,
                    "outputs": list(names),
                })

            summary_name = "bundle_summary.csv"
            summary_fields = (
                "parent_bundle_id", "network_id", "display_bundle_id",
                "direction", "edge_count", "mass", "source_p_value",
                "fwer_corrected_p",
                "p_value_source", "correction", "alpha", "inferential_status",
                "display_subdivision_count",
            )
            _write_csv(root / summary_name, [parent.summary for parent in parents], summary_fields)

            combined_png_name = "figure_parent_bundles.png"
            _save_png(root / combined_png_name, _stack_rows(row_images))
            combined_pdf_name = "figure_parent_bundles.pdf"
            pdf_temporary = root / ".figure_parent_bundles.tmp.pdf"
            row_images[0].save(
                pdf_temporary,
                format="PDF",
                save_all=True,
                append_images=row_images[1:],
                resolution=PUBLICATION_DPI,
                title="COFFEE-DAC standardized parent-bundle views",
                author="COFFEE-DAC",
                subject=(
                    "Static anatomical views, endpoint density, and associated "
                    "bundle-level inference"
                ),
            )
            os.replace(pdf_temporary, root / combined_pdf_name)

            captions_name = "figure_captions.txt"
            (root / captions_name).write_text(
                "\n\n".join(
                    f"Figure {index + 1}. {caption}"
                    for index, caption in enumerate(captions)
                ) + "\n",
                encoding="utf-8",
            )
            advance()

            for name in (
                summary_name, combined_png_name, combined_pdf_name, captions_name
            ):
                output_records.append({"path": name, "sha256": _sha256(root / name)})

            input_record = None
            if input_path:
                source = Path(input_path).expanduser().resolve()
                input_record = {
                    "path": os.fspath(source),
                    "sha256": _sha256(source) if source.is_file() else None,
                }
            manifest = {
                "schema_version": 1,
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "export_type": "standardized_parent_bundle_static_set",
                "pipeline": pipeline,
                "input": input_record,
                "source_inference": (
                    inference_manifest.get("inference") if inference_manifest else None
                ),
                "parent_bundles": parent_records,
                "outputs": output_records,
                "raster": {
                    "dpi": PUBLICATION_DPI,
                    "assembled_width_pixels": FIGURE_WIDTH_PX,
                    "assembled_width_inches": FIGURE_WIDTH_PX / PUBLICATION_DPI,
                    "background": "white",
                },
                "palette": {
                    "mode": color_mode,
                    "source": (
                        "current GUI bundle/FCN/default color resolution"
                        if color_mode == COLOR_MODE_GUI else
                        "effect direction for v2; display-group order for v3"
                    ),
                    "qualitative": (
                        None if color_mode == COLOR_MODE_GUI else "Okabe-Ito"
                    ),
                    "fallback_for_more_than_eight_subdivisions": (
                        None if color_mode == COLOR_MODE_GUI else "viridis"
                    ),
                    "fixed_across_all_outputs": True,
                },
                "centroid_line_width_mapping": {
                    "quantity": "constituent edge count",
                    "normalization": "log1p across every displayed group in this export set",
                    "edge_count_range": list(centroid_count_range),
                    "width_multiplier_range": [
                        CENTROID_WIDTH_MIN_MULT, CENTROID_WIDTH_MAX_MULT
                    ],
                    "absolute_width_cap": CENTROID_WIDTH_MAX_ABS,
                    "opacity_range": [CENTROID_OPACITY_MIN, CENTROID_OPACITY_MAX],
                    "encodes_significance": False,
                },
                "standardized_edge_style": {
                    "base_line_width": 3.0,
                    "curvature": 1.0,
                    "opacity": 0.85,
                    "endpoint_size": 1.0,
                },
                "orientation": {
                    "coordinate_frame": "brain3mm voxel-index (LAS)",
                    "face_labels": ANATOMICAL_ORIENTATION_LABELS,
                },
                "brain_meshes": _brain_mesh_metadata(network_plotter),
                "statistical_safeguard": (
                    SUBDIVISION_SAFEGUARD if pipeline == "v3" else None
                ),
                "software": {
                    "python": platform.python_version(),
                    "numpy": _package_version("numpy"),
                    "pyvista": _package_version("pyvista"),
                    "vtk": _package_version("vtk"),
                    "matplotlib": _package_version("matplotlib"),
                    "pillow": _package_version("pillow"),
                    "nibabel": _package_version("nibabel"),
                    "git_commit": _git_commit(),
                },
                "roadmap_items_not_generated": [
                    "dendrogram PDF", "MP4 rotation", "GIF rotation"
                ],
            }
            manifest_name = "publication_export_manifest.json"
            (root / manifest_name).write_text(
                json.dumps(_jsonable(manifest), indent=2, sort_keys=True,
                           allow_nan=False) + "\n",
                encoding="utf-8",
            )
            advance()

            os.replace(root, final_directory)

        if progress_callback:
            progress_callback(100)
        return {
            "directory": os.fspath(final_directory),
            "manifest": os.fspath(final_directory / "publication_export_manifest.json"),
            "figure_png": os.fspath(final_directory / "figure_parent_bundles.png"),
            "figure_pdf": os.fspath(final_directory / "figure_parent_bundles.pdf"),
            "summary": os.fspath(final_directory / "bundle_summary.csv"),
        }
