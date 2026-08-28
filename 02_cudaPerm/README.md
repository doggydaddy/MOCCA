# CUDA-accelerated bundle-level permutation FWER

> **Archive notes:**
> - 2026-08-26: the original edgewise, uncorrected-p CUDA/OpenMP executables
>   (`permutationTest_cuda`, `permutationTest_omp`, `createPerm.py`-style
>   usage) moved to `archives/uncorrected_p_pipeline_2026_08_26/`.
> - 2026-08-28: the edgewise **max-statistic FWER** backend
>   (`permutationTest_cuda_fwer.cu` and its `perm_kernels`/`results_io`
>   support files) and the pre-calibration seven-point **super-critical
>   grid** launcher moved to
>   `archives/edgewise_fwer_and_supercritical_grid_2026_08_28/`. Both were
>   valid for what they computed but are no longer the recommended path —
>   see that archive's README for why.
>
> Full project narrative, including the reasoning behind every design
> decision below, is in `conversation_archives_2026-08-26.md` at the repo
> root.

This module performs connection-wise (edgewise) Welch permutation testing
between two subject groups over an atlas-free voxel-to-voxel connectivity
graph, and produces **bundle-level, family-wise-error-corrected** results:
spatially coherent groups of surviving edges, each with a single corrected
p-value.

## Why bundling happens here, not downstream

COFFEE-DAC (module `04_coffee-dac/`) originally formed bundles *after* the
statistics were computed, from an already-thresholded edge CSV, purely to
make results legible to a human. Once FWER correction needed to happen at
the bundle level — not the edge level, correcting the ~1.78 billion
per-edge tests over an atlas-free graph is hopelessly conservative — bundle
formation had to move into the permutation loop itself: every one of the
10,000+ null permutations needs its own bundles built and its own maximum
bundle statistic recorded, and that has to run at GPU/C++ speed, not be
re-derived per-permutation in Python. So this module now owns both stages:

1. `permutationTest_cuda_bundle.cu` — CUDA backend. Computes the Welch
   t-statistic and Welch-Satterthwaite degrees of freedom for every edge and
   permutation, and writes out only the sparse set of edges exceeding a
   df-aware, two-sided cluster-forming p-value threshold (`--cluster-forming-p`).
2. `bundle_fwer_omp.cpp` — C++/OpenMP backend. Reads that sparse edge set,
   forms bundles (spatial isolation filter, strict/transitive bundling,
   intra-bundle pruning, endpoint-cluster pruning — the same deterministic
   stages COFFEE-DAC v1/v2 used), computes each bundle's mass statistic, and
   records the maximum across both signs for every permutation.
3. `run_bundle_fwer.py` — orchestrates both stages across the observed row
   and every null permutation, and converts the null distribution of maximum
   bundle statistics into per-bundle corrected p-values:
   `p_FWER = (1 + #{null max ≥ observed}) / (B + 1)`.

## The catch-22, and how it's resolved

Bundling needs a cluster-forming edge threshold (`p_CF`) chosen *before* any
statistics are corrected — but the historical transitive ("strict") bundler
is a union-find over voxel adjacency, and like any adjacency graph it has a
**percolation phase transition**: past some suprathreshold edge density, the
union-find chains across most of the brain and "the largest bundle" becomes
a single giant, anatomically meaningless component rather than a localized
structure. Pick `p_CF` too liberally and you get a statistically valid but
uninterpretable result (one real run's "significant bundle" covered 8.2
million edges, 98.7% of all mask voxels); pick it by eye to make the result
look like a sensible bundle and the whole analysis becomes circular.

The fix is **`percolation_calibration.py`**: a cheap, independent calibration
run (a few hundred null permutations, never the observed grouping) that
measures where that transition actually sits for *this* dataset's own null
adjacency geometry, before any real inference is run.

- **Order parameter:** the fraction of the mask's voxels touched by the
  single largest strict bundle (summed across both signs, matching how
  real inference takes one joint two-sided maximum). This is the
  giant-voxel-fraction metric devised for this purpose — not the more
  obvious "largest bundle edges / all retained edges" fraction, which was
  tried first and rejected: at very strict thresholds so few edges survive
  that a handful of them land in one component by chance, and edge-fraction
  spuriously climbs back toward 1.0 even though nothing is actually
  percolating. Voxel-fraction has a fixed denominator (total mask voxels)
  regardless of threshold, so it doesn't have this artifact — it falls
  monotonically all the way down the threshold grid, and it's also the
  metric that matches the original giant-component symptom directly ("98.7%
  of voxels touched").
- **Rule:** sweep a threshold grid purely on the null permutations; find the
  most liberal `p_CF` where a chosen percentile (default: 95th) of the null
  giant-voxel-fraction distribution stays at or below a small epsilon
  (default: 5%). That is the estimated transition. Recommend one grid step
  stricter as a safety margin against calibration sampling noise near the
  transition.
- Because this never looks at the observed bundle statistics, the resulting
  `p_CF` is a legitimate, pre-registered choice — not something picked
  because a result looked good.

```bash
.venv/bin/python 02_cudaPerm/percolation_calibration.py \
  FILELIST PERMUTATIONS OUTPUT_DIR \
  --calibration-permutations 200 \
  --neighbor-dist 1.0 --min-size 10 --min-cluster-voxels 6 \
  --bundle-threads 16
```

Reuses the same CUDA v3 sparse format as real inference: one CUDA pass at
the most liberal grid threshold, then the C++ bundler re-thresholds the
cached sparse edges at every stricter grid point for free. Writes
`percolation_calibration_curve.csv` (per-threshold, per-permutation giant
fractions), `percolation_calibration_summary.csv` (percentiles per
threshold), and `percolation_calibration_results.json` (the recommended
`p_CF`). Each dataset (different subject counts/grouping) needs its own
calibration run — but empirically the transition has landed at the same
grid point (`p_CF=1e-5`, recommended operating point `5e-6`) for both
datasets tried so far, since it's governed more by the fixed spatial mask
geometry (`neighbor_dist`, voxel count) than by sample size.

**Note on batch size:** each CUDA invocation reloads the *entire* subject
connectivity dataset (tens to hundreds of GB for real datasets) regardless
of how many permutation rows it's given, so `percolation_calibration.py`
defaults to one CUDA batch covering every calibration permutation to avoid
paying that reload cost more than once. `run_bundle_fwer.py`'s default
`--batch-size 8` is tuned for resumability on very long runs instead —
raise it explicitly (e.g. `--batch-size 1000`–`2500`) for large production
runs where disk is not warm/cached, or the reload overhead can dominate
wall time.

## Running the real inference

Once calibration has produced a `p_CF`, run the full permutation count with
that single, fixed threshold — **not** a multi-threshold grid. A grid's
`symmetric_permutation_min_p` correction pays a real, measurable power cost
for searching over candidate thresholds (observed directly: `p_grid_fwer`
ran ~0.03–0.05 higher than the single-threshold `p_threshold_fwer` at
matching rows in one real run); once calibration has already chosen a
threshold via an independent, null-only rule, that search is unneeded and
only costs power. The `--cluster-forming-p-grid` option described further
below still exists for genuinely exploratory multi-threshold sensitivity
analyses, but is not the default recommendation.

```bash
.venv/bin/python 02_cudaPerm/run_bundle_fwer.py \
  FILELIST PERMUTATIONS OUTPUT_DIR \
  --cluster-forming-p 5e-6 \
  --null-permutations 10000 \
  --statistic mass --neighbor-dist 1.0 --min-size 10 --min-cluster-voxels 6 \
  --bundle-method strict --bundle-engine cpp --bundle-threads 16 \
  --capacity 20000000 --batch-size 1000
```

(`--capacity` bounds sparse edges per CUDA part *per permutation row* — the
default of 10M was seen to overflow on a dense real permutation; 20M has
been sufficient for both datasets run so far.)

Produces `observed_bundles_fwer.csv` (every surviving bundle's `edge_count`,
`mass`, and corrected `p_fwer`) and `observed_edges_bundled.csv` (every
member edge, for visualization). Hand these to
`03_prepResultsForVisualization/prepare_bundle_single_fwer.py`.

### Long-running jobs

Launch any multi-hour job (calibration batches over a few hundred
permutations are usually minutes; full 10k-permutation production runs can
be hours) in a named `tmux` session with output logged to a file inside the
output directory, so it can be attached to and monitored independently:

```bash
tmux new-session -d -s <descriptive_name> \
  "<command> 2>&1 | tee OUTPUT_DIR/run.log"
tmux attach -t <descriptive_name>   # detach: Ctrl-b d
```

## Monte Carlo precision of the reported p-values

Every `p_fwer` / `p_threshold_fwer` / `p_grid_fwer` this pipeline reports is
a permutation-count ratio r/m (m = `null_permutations` + 1 trials including
the observed row; r = trials at least as extreme as observed) — a binomial
proportion with its own sampling uncertainty from having run only m
permutations. `bundle_fwer_precision.py` attaches an exact Clopper-Pearson
confidence interval to every reported p-value and flags any bundle whose CI
straddles `--alpha`, i.e. where more permutations could plausibly flip the
significance call. It recomputes each p-value from the underlying null
distribution (`null_max_bundle_statistics.npy` / `permutation_bundle_maxima_grid.csv`)
and raises if that doesn't match the value on disk, rather than trusting the
CSV's rounded float.

```bash
.venv/bin/python 02_cudaPerm/bundle_fwer_precision.py \
  /path/to/bundle_fwer_result_dir --alpha 0.05
```

Works on both single-threshold and grid output (auto-detected). On the
controls-vs-patients production run, bundles 94 and 95 (p_fwer = 0.042,
0.045) both have 95% CIs clear of 0.05 (0.038–0.046 and 0.041–0.049) — 10k
permutations resolves the call at that alpha. This is a precision check on
the null distribution sample size, not a power analysis: it says nothing
about the probability of having missed a real but smaller effect (e.g. the
LTLE/RTLE null result).

## Optional: multi-threshold grid FWER

`--cluster-forming-p-grid` remains available for an explicit, declared
sensitivity analysis over several thresholds that have *all* already been
confirmed sub-critical by calibration (e.g. `1e-5 5e-6 2e-6 1e-6`). CUDA
runs once at the most liberal grid threshold and stores each surviving
edge's t-statistic and Welch degrees of freedom (`--store-df`, sparse format
v3); the C++ engine reuses that payload at every stricter threshold,
recomputing bundle mass from `|t| - t_critical(df, p)` each time. The
maximum bundle mass per permutation, across all grid points and both signs,
becomes a symmetric permutation-tail rank; the minimum rank across
thresholds is the per-permutation search statistic, so reported
`p_grid_fwer` corrects simultaneously for bundle selection, both signs, and
the threshold search itself.

```bash
.venv/bin/python 02_cudaPerm/run_bundle_fwer.py FILELIST PERMUTATIONS OUTPUT \
  --cluster-forming-p-grid 1e-5 5e-6 2e-6 1e-6 \
  --null-permutations 10000 \
  --statistic mass --neighbor-dist 1.0 --min-size 10 --min-cluster-voxels 6 \
  --bundle-method strict --bundle-engine cpp --bundle-threads 16
```

Grid output includes `permutation_bundle_maxima_grid.csv`, a combined
`observed_bundles_grid_fwer.csv`, and threshold-specific edge/bundle files
under `thresholds/p_*/`. Hand these to
`03_prepResultsForVisualization/prepare_bundle_grid_fwer.py` instead of the
single-threshold export script. **Never run this over a grid spanning
uncalibrated (potentially super-critical) thresholds** — see the archived
seven-point grid for why that produces statistically valid but
uninterpretable giant components.

## Rejected experiment: bounded, non-chaining bundles

> **Status (2026-08-27): not adopted.** Preserved for provenance only; the
> active default remains the historical `strict` bundler, now used safely
> below its percolation transition via calibration instead.

`--bundle-method bounded` selects the separate `bundle_fwer_bounded_omp`
executable (same source as `bundle_fwer_omp.cpp`, built with
`DEFAULT_BOUNDED_BUNDLES`). It defines a bundle around its strongest
unassigned representative edge; a candidate joins only when its endpoints
can be oriented so each lies within `neighbor_dist` of the corresponding
representative endpoint, giving every bundle a hard Chebyshev diameter of
`2 * ceil(neighbor_dist)` voxels and preventing any transitive chain from
expanding it further. This does prevent percolation, but a full 10k
LTLE/RTLE run under this method found no grid-FWER-significant bundle and
produced 219,927 small, visually arbitrary-looking bundles — rejected in
favor of calibrating the historical bundler's threshold instead. Full
record in `archives/bounded_bundling_experiment_2026_08_27/`.

## Regression checks

```bash
.venv/bin/python -m unittest 02_cudaPerm/regression_bundle_fwer.py
.venv/bin/python -m unittest 02_cudaPerm/regression_bundle_fwer_cpp.py
.venv/bin/python -m unittest 02_cudaPerm/regression_cuda_bundle.py
```

The last uses a tiny synthetic CUDA fixture and is skipped when a GPU or the
newly built backend is unavailable. `bundle_fwer.py` is the readable Python
reference implementation these regress against; it is not used for
production runs (`bundle_fwer_omp` is, by default).

## Build

```bash
cmake -S 02_cudaPerm -B 02_cudaPerm/build
cmake --build 02_cudaPerm/build \
  --target permutationTest_cuda_bundle bundle_fwer_omp bundle_fwer_bounded_omp -j"$(nproc)"
```

## Generating permutations and file lists

```bash
python 02_cudaPerm/generatePermutations.py \
  -nPerm 10000 -nA <n_group_A> -nB <n_group_B> -o permutations.txt
```

Row 0 is always the true observed grouping (`range(nA)`: the first `nA`
filelist entries are group A), prepended automatically; rows 1.. are random
label permutations. `t = mean(group A) - mean(group B)`, so a positive
sign means group A greater and negative means group B greater — confirm
which physical group is "A" in your filelist before interpreting bundle
sign, since getting this backwards is a correctness bug, not a cosmetic one.

The file list is one subject connectivity-matrix path per line, in the same
order group A/group B were assigned when the permutation file was built:

```bash
find . -name 'groupA_subj*.ccmat' | sort > filelist.txt
find . -name 'groupB_subj*.ccmat' | sort >> filelist.txt
```

`average_ccmat_runs.py` builds subject-mean connectivity matrices from
repeated per-subject runs — the inferential unit is the subject, so repeated
runs should be averaged rather than treated as independent observations.

## Performance reference

On the 37-subject LTLE/RTLE subject-mean dataset (59,677 voxels, 1.7806
billion edges), the identical observed row plus 200 null rows at a fixed
`|t| >= 3.9` gave:

| Bundle engine | Total wall time | Speedup |
|---|---:|---:|
| Python oracle | 8:40:42 | 1x |
| C++/OpenMP (4 workers) | 0:11:48.59 | 44.09x |

All 201 threshold-edge counts, retained-edge counts, bundle counts, observed
edge assignments, observed bundle labels, and corrected p-values matched
between engines; maximum-statistic differences from floating-point
summation order were at most `3.97e-4` and changed no corrected p-value.
Peak process memory was ~36.7 GiB, dominated by the CUDA input chunk.

A full 10,000-permutation single-threshold production run (68 subjects,
`p_CF=5e-6`) completed in well under an hour of CUDA + bundling time once
data was read; the dominant real-world cost at that scale is reading each
subject's full connectivity matrix from disk once per CUDA invocation
(tens to hundreds of GB total) — see the batch-size note above.
