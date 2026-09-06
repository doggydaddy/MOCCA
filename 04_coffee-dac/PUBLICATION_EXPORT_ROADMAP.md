# COFFEE-DAC publication export roadmap

**Status:** static figure set partially implemented. The GUI exports
standardized multi-view parent-bundle figures, endpoint-density MIPs, full-edge
supplements, bundle summaries, captions, and a checksum manifest. Dendrogram
PDF and MP4 export remain deferred until the manuscript's statistical
limitations are addressed.

## Implemented increments

The GUI's **Export Figure** button exports the live camera view rather than
reconstructing it from the current tree selection. Raster exports use a
7-inch-wide, 600-dpi render and trim the uniform canvas border; PNG and TIFF are
supported. PDF and SVG use VTK's graphics export path, with the complex 3D
actors rasterized inside the vector container for practical file sizes. A
camera-linked anatomical orientation cube is present in the GUI and every
export.

Each figure is accompanied by a caption text file and a JSON metadata sidecar.
The metadata records the camera, rendering mode and styles, edge counts,
parent-level v3 inference separately from descriptive subdivisions, brain
meshes, checksums, software versions, and Git commit. This is intentionally a
single-view building block, not yet the standardized four-panel primary figure
described below.

The **Publication Set** choice now creates a timestamped export directory. For
each selected inferential bundle it produces a four-panel 7-inch-wide figure
with left-lateral, superior/dorsal, right-lateral, and endpoint-density views.
All 3D panels use parallel projection and the same camera distance, scale,
lighting, mesh opacity, and canvas dimensions. Primary panels use centroid
connections with one fixed, export-wide log edge-count-to-width mapping; a
separate three-view full-edge figure is also written.

Endpoint density is exported as a superior-inferior maximum-intensity
projection over raw incident-edge counts per endpoint voxel, overlaid on the
brain3mm anatomical template with R/L/A/P markers and a count colorbar. The raw
voxel counts are written to CSV. At export time, the user can preserve the
currently resolved GUI bundle/FCN colors (also used by GIF export) or choose a
color-vision-deficiency-safe Okabe-Ito palette (viridis when more than eight
colors are needed). The selected colors are frozen across primary and
supplementary outputs and recorded as RGBA values in the manifest.

When the input has an upstream FWER visualization manifest, source bundle ID,
effect label, edge count, mass, corrected p-value, correction method, and alpha
are recovered into `bundle_summary.csv` and captions. A bare edge-array p-value
is never labeled FWER-corrected without that provenance. A single v3 load can
legitimately contain more than one independently FWER-significant parent
bundle at once (e.g. two bundles both survive FWER at alpha=0.05 in the same
run); each is subdivided by its own independent edge-linkage tree and never
mixed with another parent's edges (see `coffee_dac_pipeline_v3.py`'s module
docstring). For each such parent bundle, inference remains attached only to
that complete parent, and a partial interactive subdivision selection is
expanded back to that parent's complete edge set for the standardized figure.

## Goal

Add a reproducible publication-export mode alongside the interactive GUI and
GIF exporter. A paper must remain understandable from static figures; animated
media should be supplementary.

## Terminology and inferential safeguards

- Use **inferential bundle** or **parent bundle** for a bundle tested by the
  bundle-level FWER pipeline.
- Use **display subdivision** or **visual sub-bundle** for a post hoc v3 split.
- Never call a display subdivision a "significant sub-bundle."
- Never assign or display the parent's corrected p-value in a way that suggests
  that subdivisions were tested independently.
- Add this statement to applicable captions or metadata:

  > Colors identify post hoc visual subdivisions of the parent bundle.
  > Statistical testing was performed on the complete parent bundle; the
  > subdivisions were not tested independently.

## Static primary-figure export

Generate one consistently formatted row per inferential bundle with:

1. left-lateral view;
2. superior/dorsal view;
3. right-lateral view;
4. endpoint-density view.

Requirements:

- identical camera, scale, lighting, surface opacity, and cropping across rows;
- tightly cropped output with minimal unused canvas;
- colorblind-safe visual-subdivision palette, fixed across every output;
- persistent anatomical orientation markers;
- parent bundle identifier, direction, edge count, mass, and FWER-corrected
  p-value in the row label or caption metadata;
- centroid/skeleton connections in the primary figure, with line width encoding
  constituent edge count rather than significance;
- full-edge static render exported separately for transparency;
- 300--600 dpi raster output at final print dimensions, with vector output when
  supported by the rendering path.

## Endpoint-density representation

For each parent bundle, count the number of incident bundle edges at every
endpoint voxel. Export either a surface heat map or anatomical
maximum-intensity projection of this endpoint incidence. Record whether values
are raw counts, proportions of bundle edges, or normalized within bundle.

This panel should communicate endpoint concentration independently of the
occlusion and line-density problems of a 3D edge rendering.

## Dendrogram export

- Export PDF or SVG from Matplotlib.
- Match branch colors to visual-sub-bundle colors in the brain rendering.
- State the distance metric, linkage method, and selected cut in the figure
  metadata and caption template.
- Draw the selected cut explicitly.
- Include display-subdivision edge counts.
- Support a contracted/truncated representation when an edge-level v3 tree has
  too many leaves to print legibly.
- Treat the dendrogram as descriptive visualization, not additional inference.

## Supplementary animation

Add MP4/H.264 export in addition to GIF:

- one short video per parent bundle;
- fixed elevation and reproducible rotation path;
- optional brief pauses at standard anatomical views;
- persistent orientation markers, bundle direction, and color legend;
- no changing color scale or line scaling during playback;
- conventional 16:9 or 4:3 aspect ratio and journal-configurable resolution;
- short plain-text legend exported beside each media file.

## Suggested export set

```text
publication_export/
  bundle_summary.csv
  figure_parent_bundles.pdf        # assembled standardized views, if feasible
  figure_parent_bundles.png
  parent_bundle_<id>_views.png
  parent_bundle_<id>_endpoints.png
  parent_bundle_<id>_endpoint_density.csv
  parent_bundle_<id>_full_edges.png
  parent_bundle_<id>_dendrogram.pdf
  parent_bundle_<id>_rotation.mp4
  parent_bundle_<id>_rotation.gif  # optional
  figure_captions.txt
  publication_export_manifest.json
```

The manifest should record input/output checksums, source bundle-level p-value,
camera states, brain mesh, colors, density normalization, line-width mapping,
software versions, Git commit, and whether each output uses full edges,
centroids, or post hoc subdivisions.

## Manuscript placement envisioned

- Main figure: standardized multi-view parent-bundle composite plus endpoint
  density.
- Main table: bundle edge count, mass, FWER p-value, and permutation precision
  interval.
- Supplementary figure: full-edge views.
- Supplementary figure: dendrogram and display-subdivision sizes.
- Supplementary video: rotating parent bundles.
- Supplementary data: machine-readable bundle edges and display-subdivision
  assignments.
