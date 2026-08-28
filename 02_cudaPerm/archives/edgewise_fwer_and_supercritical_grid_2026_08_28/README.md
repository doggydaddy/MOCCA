# Superseded: edgewise max-statistic FWER, and the pre-calibration super-critical grid

Archived on 2026-08-28. These are two distinct, now-abandoned approaches from
the path that led to the current bundle-level, percolation-calibrated FWER
pipeline (see `02_cudaPerm/README.md`). Both are statistically valid for what
they compute; neither is part of the recommended workflow any more. Full
narrative in `conversation_archives_2026-08-26.md` at the repo root.

## 1. Edgewise max-statistic FWER

- `permutationTest_cuda_fwer.cu` — CUDA backend computing the classic
  max-statistic FWER correction independently for every one of the
  atlas-free graph's ~1.78 billion edges.
- `perm_kernels.cu` / `perm_kernels.cuh` — the Welch t-statistic device
  kernel it used. Nothing else in the active pipeline depends on these;
  `permutationTest_cuda_bundle.cu` reimplements its own equivalent kernel
  inline and only mentions `perm_kernels.cu` in a comment.
- `results_io.c` / `results_io.h` — output-file I/O used only by
  `permutationTest_cuda_fwer`.
- `run_controls_vs_patients_subject_fwer_100k.sh`,
  `run_ltle_vs_rtle_subject_fwer_100k.sh` — launchers for the two production
  100k-permutation edgewise FWER runs.
- `run_permutationTest.sh` — general-purpose tmux launcher wrapper for this
  binary, originally at `pipeline/run_permutationTest.sh`; moved here since
  it has no purpose without `permutationTest_cuda_fwer`. Its `BINARY` path
  has been updated to point at this archive's own build output
  (`archives/edgewise_fwer_and_supercritical_grid_2026_08_28/build/`), so
  build here first (see below) before using it.

**Why abandoned:** correcting for the maximum statistic across ~1.78 billion
simultaneous tests is extremely conservative — far more severe than a
conventional voxelwise analysis with tens or hundreds of thousands of tests.
Both production runs found at most one edgewise-significant connection. This
motivated moving to bundle-level (cluster-mass) inference, which corrects
for far fewer, spatially coherent units instead of every individual edge.

`ccmat_io.c`/`ccmat_io.h` are **not** duplicated here — they remain the
single active copy in the parent `02_cudaPerm/` directory, shared with
`permutationTest_cuda_bundle.cu`. This archive's `CMakeLists.txt` builds
`permutationTest_cuda_fwer` standalone by referencing that parent copy
directly.

```bash
cmake -S . -B build
cmake --build build -j2
```

## 2. Pre-calibration super-critical grid

- `run_ltle_vs_rtle_bundle_grid_10k.sh` — launcher for the original
  seven-point cluster-forming grid (`0.001` down to `0.00001`) run through
  `run_bundle_fwer.py --cluster-forming-p-grid`.

**Why abandoned:** the historical transitive ("strict") bundler percolates
into a single brain-spanning giant component once the suprathreshold edge
graph gets dense enough. Several of this grid's more liberal points
(`0.001` through `0.0001`) sat on the super-critical side of that
transition, so bundles found there were valid permutation-FWER results for
the *statistic actually computed*, but not localized anatomical bundles —
one run's "significant bundle" covered 8.2 million edges and 98.7% of mask
voxels. The current pipeline instead calibrates a sub-critical
cluster-forming threshold from null permutations *before* inference (see
`percolation_calibration.py` and the main `02_cudaPerm/README.md`), then
runs a single pre-registered threshold rather than searching a grid that mixes
safe and unsafe regimes.

The general `--cluster-forming-p-grid` capability in `run_bundle_fwer.py`
itself was **not** removed — it is still valid for an explicit, declared
multi-threshold sensitivity analysis over a set of thresholds that have
*all* already been confirmed sub-critical by calibration. What was archived
here is only the specific launcher that used the old, uncalibrated,
partly super-critical grid.
