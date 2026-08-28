# GUI for COFFEE-DAC

A Graphic User Interface for COFFEE-DAC, a part of the MOCCA pipeline for
analysis of resting-state fMRI data.

COFFEE-DAC-program provides brain connection data from more raw fMRI-dataset to
more intuitive 3D-graphs, where the connections are represented by bundles of
connections and these bundles are grouped into FCNs (Functional Connectivity
Networks).

# Dependencies

This GUI is dependent on  the COFFEE-DAC program, ijk-csv-data (see below)
processed from the MOCCA analysis pipeline and the *brain_template.stl* with the
specific brain mesh belonging to the COFFEE-DAC pipeline analysis program.

# Prequisites

## Required Packages

List of required packages in $requirements.txt$:

* pandas
* PyQt5
* pyvistaqt
* nibabel
* scipy
* scikit-learn
* tqdm

## (Optional) Virtual environment

May use the virtual environment for easy loading: Enable the virtual environement

        source .venv/bin/activate

# How to run

        python pyqt_launcher.py

# Details 

## Loading Data

The program will immediately alert you to load data. The data is expected to be
in CSV format, with each row representing a single edge/connection between two
voxels in the following format:

$$<i_1>, <j_1>, <k_1>, <i_2>, <j_2>, <k_2>, <value>$$

Where $i_1, j_1, k_1$ and $i_2, j_2, k_2$ are the coordinates of the endpoints
of the connection in $ijk$ coordinate system (from *AFNI*'s *3dmaskdump*), and
$value$ is the weight of the connection. Currently, the framework only support
*thresholded* connections, hence $value$ is not used and assumed to take the
value of $1$.

## Choosing v2 vs v3

The pipeline that produces the input CSVs here is now the bundle-level FWER
path in `02_cudaPerm/` (see its README) plus
`03_prepResultsForVisualization/prepare_bundle_single_fwer.py`. That changes
what a "bundle" typically looks like: FWER-significant bundles under the
percolation-calibrated threshold are often already network-scale (thousands
of edges) rather than the many small bundles v1/v2's `hc2` step was designed
to merge upward into networks. There is nothing left to merge for those —
they *are* the network.

- **v2 (agglomerative, `hc2`)** — use when the exported significant bundle(s)
  are still small/numerous enough that grouping several of them together
  into a handful of networks would actually help interpretation. This is the
  original COFFEE-DAC bundle → network direction.
- **v3 (divisive)** — use when a single significant bundle is already
  large/network-scale on its own. v3 treats that bundle as the "network" and
  divides its individual edges into sub-bundles instead, for legibility only.
  See below.

There's no fixed edge-count rule for which to pick — look at the exported
bundle sizes (`observed_bundles_fwer.csv`'s `edge_count` column) and decide
based on whether merging or dividing would make the result easier to read.

## Pipeline v3: divisive sub-bundling

Hierarchical clustering has no native divisive primitive, so v3 goes the
only direction agglomerative clustering can: it treats the individual
**edge** as the smallest available unit (the same leaves `hc1` already
builds its tree from via `h1_dist` — the identical edge-to-edge distance
metric v1/v2 use to form bundles from a raw edge pool), builds one linkage
tree directly over a significant bundle's edges, and cuts that tree into N
sub-bundles. The p-value column is never touched: every sub-bundle keeps the
single whole-bundle FWER p-value it started with, since this is a rendering
aid, not a new statistical claim — splitting a significant bundle for
legibility must never be read as separate, uncorrected significance for any
individual piece.

```bash
# First run: build the tree and cut into 6 sub-bundles
python run_pipeline_v3.py path/to/significant_bundle_export.csv --bundles 6

# Re-cut an existing v3 cache into a different count -- instant, no
# recomputation (same "cache the whole tree, cut wherever" trick as v2's
# --recut for networks)
python run_pipeline_v3.py path/to/significant_bundle_export.csv --recut 10
```

Refuses (rather than silently degrading to an approximate tree) above
`--max-exact` edges (default 50,000, matching v1's own `hc1` default) since
an exact edge-level linkage tree needs O(N²) distances and a full tree is
required for the instant-recut guarantee above to actually hold.

In the GUI, the same "cached results found" dialog now also detects a v3
cache and offers **Load existing v3 results — divisive (fast)**; the recut
spinbox relabels itself to "Cut this bundle into N sub-bundles" when a v3
cache is selected. Both v2 and v3 caches can coexist for the same input CSV
(different filename suffixes), so trying both is non-destructive.

## Processing provenance and v2 caches

Every successful v2 processing run writes a three-file result set beside the
original input CSV:

```text
<input>_v2_processed.csv
<input>_v2_linkage.npy
<input>_v2_params.json
```

The JSON sidecar is the authoritative processing record. It contains all
resolved pipeline parameters, including defaults; input and output SHA-256
checksums; input/result counts; UTC timestamps; whether the run came from the
GUI, CLI, or API; Python/package versions; and the Git commit when available.
Persisted network re-cuts are appended to its `recuts` history.

The command-line pipeline automatically reuses a cache only when its input
checksum, output checksums, and complete parameter set match the manifest.
Legacy cache pairs without a parameter sidecar remain loadable through the GUI,
where they are identified as having unavailable parameter metadata, but the CLI
will reprocess them rather than assume their settings.

The GUI displays the recorded v2 parameters before loading a cache. Processing
performed directly by the GUI uses the v2 defaults and records those resolved
values in the same sidecar format.

Files ending in `_v2_processed.csv` are rejected as raw pipeline inputs by
default. This prevents accidental names such as
`_v2_processed_v2_processed.csv`. The CLI provides
`--allow-processed-input` for deliberate exceptional use.

## FCN and Bundles filtering

The FCN:s and bundles are organized in a tree system. In order to get a plot you have to
first mark the bundles you want to plot (or the button "All" for whole FCNs) and
then you press "Plot Selection" in order to see your selections plotted. The
button "Show All" will immediately plot all FCNs. You can also see your selected
bundles for each network in the right column "Selected Bundles". "Clear All"
will clear the whole plot at any time.

## Centroids

If you go to any FCN you will see that for each bundle there is a button option
titled "Centroids", if you click this button a toggle will appear and the
centroid of the bundle will appear if you select the bundle. The centroid of the
bundle represents the path between the average coordinate of the endpoints of
each side of the bundle, which is a function meant to simplify the plot. For
each FCN there is also a button option titled "Toggle All Centroids", which
toggles all "Centroid"-buttons for the FCN, which will lead to only centroids
appearing for the FCN after you press "Plot selection". The button in the main
frame of the GUI titled "Toggle All Centroids" will toggle the
"Centroid"-buttons for all bundles in all FCNS, so whatever bundle or FCN you
select, you will only get the centroids of the bundles.

## Coloring

Each FCN has a default color so all its bundles are colored the same. You change
the color for a bundle by selecting an FCN, then you will see each bundle has a
button titled "Color". This button will be marked with your selected color and
if it is color blanc it means that the bundle has the FCN's default color.

## Fine Tuning

For each bundle there is a button titled "Fine Tune", which is a feature
acivating the graphing parameter buttons "Curve", "Thickness". These buttons
adjust the parameters they are titled after for each bundle and you can see
their settings  when you press "Fine Tune". You can Fine Tune as well for whole
FCNs and as well for all FCNS at once with the buttons "Fine Tune FCN" and "Fine
Tune All FCNs". The endpoint size depends on the voxel-size in your dataset, 
therefore it is best to not adjust the endpoint size in the GUI.

## Export to GIFs

In order to export the given plot, make sure to have pressed plot selection
before. You can adjust the elevation angle of the "camera" pointed at the plot
with the slider labeled as "Elevation", it ranges between -60 and 60 degrees.
There is some slight vertical oscilation (5 degrees amplitude) in the plot in
order to hinder distortion and lagging. If you want to see the animation of the
GIF before exporting it, press "Live preview". The mouse does not contribute to
the angle adjustments of the plot in the GIF-animation, adjustments are made by
the slider. "Export All GIFs" will export one GIF of each FCN all at once.

## Dendrogram

"Show Dendrogram" will show a dendrogram of the hierarchical clusterings of the bundles. The titles of these are colored after their FCN-colors, as marked in the upper corner.
