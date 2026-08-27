# Result preparation

The original raw-permutation-p conversion and thresholding pipeline was
archived on 2026-08-26 under:

`archives/uncorrected_p_pipeline_2026_08_26/`

The active bundle-grid export is `prepare_bundle_grid_fwer.py`. It selects only
bundles passing grid-adjusted FWER, streams their member edges into separate
files for each cluster-forming threshold, and creates matching COFFEE-DAC v2
caches for immediate manual visualization. It does not rerun bundle formation
or hierarchical clustering and does not modify the inferential result files.

Example:

```bash
.venv/bin/python 03_prepResultsForVisualization/prepare_bundle_grid_fwer.py \
  /path/to/bundle_grid_results /path/to/visualization_output --alpha 0.05
```

In the GUI, select an exported CSV that does not end in `_v2_processed.csv`
and choose **Load existing v2 results (fast)**. The edge `pvalue` column is the
bundle's grid-adjusted FWER p-value, repeated on its member edges.

If no bundles survive the chosen alpha, a separate diagnostic export can be
made with `--top-bundles-per-threshold N`. Such output is explicitly marked as
exploratory in its filenames, README, and manifests. It retains the actual
grid-adjusted p-values and must not be described as statistically significant.

AFNI support utilities remain in `afni/` because they are not specific to the
archived uncorrected-p workflow.
