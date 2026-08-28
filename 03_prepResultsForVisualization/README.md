# Result preparation

The original raw-permutation-p conversion and thresholding pipeline was
archived on 2026-08-26 under:

`archives/uncorrected_p_pipeline_2026_08_26/`

## Main pipeline: single-threshold bundle FWER export

`prepare_bundle_single_fwer.py` is the export script for the main pipeline —
a `02_cudaPerm/run_bundle_fwer.py` run made with a single, pre-registered
`--cluster-forming-p` (chosen by null-only percolation calibration, see
`02_cudaPerm/README.md`), not a threshold grid. It reads
`observed_bundles_fwer.csv` / `observed_edges_bundled.csv` directly from the
result directory (no `thresholds/<slug>/` subdirectory, since there's only
one threshold), selects bundles at or below `--alpha`, and writes a COFFEE-DAC
v2-shaped cache (raw + `_v2_processed.csv` + `_v2_linkage.npy` placeholder +
`_v2_params.json`) ready to load in the GUI.

```bash
.venv/bin/python 03_prepResultsForVisualization/prepare_bundle_single_fwer.py \
  /path/to/bundle_fwer_result_dir /path/to/visualization_output \
  <dataset_label> \
  --positive-label "controls > patients" --negative-label "patients > controls" \
  --alpha 0.05
```

`--positive-label`/`--negative-label` are **required**, not guessed: they
depend on which physical group was encoded as "group A" when the permutation
file was generated (`generatePermutations.py`'s row 0 is `range(nA)` — the
first `nA` filelist entries). Get this backwards and every bundle's
direction is misreported; verify it from the filelist/permutations
construction before running, the same way `02_cudaPerm/README.md`'s
"Generating permutations" section describes.

For diagnostic inspection when nothing (or very little) survives `--alpha`,
`--top-bundles N` additionally exports the N smallest-p bundles regardless of
significance, into a separately-labeled, explicitly non-significant export —
useful for checking, e.g., whether two adjacent significant bundles found at
a stricter threshold turn out to be a strict subset of one larger bundle
found at a more liberal (but still calibrated sub-critical) threshold.

In the GUI, select the exported CSV that does **not** end in
`_v2_processed.csv` and choose **Load existing v2 results (fast)** — or, if
the surviving bundle(s) are large enough to need dividing for legible
visualization, process with pipeline v3 instead (see
`04_coffee-dac/README.md`'s "Choosing v2 vs v3" section). The edge `pvalue`
column is the bundle's single-threshold FWER p-value, repeated on its member
edges.

## Optional: multi-threshold grid export

`prepare_bundle_grid_fwer.py` is the export script for the secondary,
optional `--cluster-forming-p-grid` path in `run_bundle_fwer.py` (an explicit
sensitivity analysis across several already-calibrated sub-critical
thresholds — see `02_cudaPerm/README.md`). It selects only bundles passing
grid-adjusted FWER, streams their member edges into separate files for each
cluster-forming threshold, and creates matching COFFEE-DAC v2 caches. It does
not rerun bundle formation or hierarchical clustering and does not modify the
inferential result files.

```bash
.venv/bin/python 03_prepResultsForVisualization/prepare_bundle_grid_fwer.py \
  /path/to/bundle_grid_results /path/to/visualization_output --alpha 0.05
```

Same GUI loading convention as above. If no bundles survive the chosen
alpha, `--top-bundles-per-threshold N` gives the same kind of explicitly
exploratory, non-significance-filtered diagnostic export.

**Never run this over a grid result that spans uncalibrated (potentially
super-critical) thresholds** — see `02_cudaPerm/archives/edgewise_fwer_and_supercritical_grid_2026_08_28/README.md`
for why that produces statistically valid but anatomically uninterpretable
giant "bundles".

AFNI support utilities remain in `afni/` because they are not specific to
either the archived uncorrected-p workflow or the bundle-FWER exports.
