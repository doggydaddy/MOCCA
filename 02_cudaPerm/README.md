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
  --neighbor-dist 1.0 --min-size 10 --min-cluster-voxels 6 \
  --bundle-threads 16
```

Calibration now defaults to **rows 1–1000 only**, which are held out from
inference — see "Disjoint calibration and inference permutations" below. It
also reports how stable the selection is, by bootstrap-resampling and
subdividing those calibration rows (never an inference row):

```text
=== Selection stability (calibration rows only) ===
Bootstrap (1000 resamples): modal transition p_CF=1e-05 selected in 100.0% of replicates
  selection counts: {'1e-05': 1000}
Subdivision into 4 disjoint blocks of [250, 250, 250, 250] rows selected: [1e-05, 1e-05, 1e-05, 1e-05]
```

If the disjoint blocks disagree, the fix is a larger prospective calibration
set or a stricter predeclared rule — never a peek at the held-out inference
rows to break the tie.

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
  --statistic mass --neighbor-dist 1.0 --min-size 10 --min-cluster-voxels 6 \
  --bundle-method strict --bundle-engine cpp --bundle-threads 16 \
  --capacity 20000000 --batch-size 1000
```

Inference defaults to **row 0 plus rows 1001–11000** — the 10,000 held-out
nulls. `--null-permutations` has been removed in favour of
`--inference-permutations`, and passing the old flag raises an error pointing
at the replacement rather than silently running a different row set.

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

## Disjoint calibration and inference permutations

Calibration picks the cluster-forming threshold from the null distribution.
If inference then reuses those same null rows, the threshold was tuned on
part of the data its own p-values are computed against. The fix is a held-out
design, implemented in `permutation_rows.py` and shared by both stages:

```text
row 0             observed assignment
rows 1..1000      calibration set only   (1,000 null permutations)
rows 1001..11000  inference set only    (10,000 null permutations)
```

One master file with recorded, non-overlapping row ranges is easier to audit
than separate seeds or files, and guarantees no label assignment is reused
across the two stages. `generatePermutations.py` already draws every row
uniquely, so the two subsets are disjoint by construction as well as by range.

Because calibration maxima enter neither the numerator nor the denominator:

```text
p_FWER = (1 + #{inference maxima >= observed}) / 10001
```

The minimum attainable production p-value therefore stays `1/10001` even
though 11,000 null permutations were computed in total. `observed_bundles_fwer.csv`
now carries the raw `inference_exceedances` count alongside `p_fwer`, and
`bundle_fwer_results.json` records `p_fwer_denominator` and `p_fwer_formula`
explicitly.

The partition is an explicit, validated configuration rather than an implicit
convention. All four values are accepted by `generatePermutations.py`,
`percolation_calibration.py` and `run_bundle_fwer.py`, and each stage records
the **whole** partition in its manifest — not just the half it consumed:

```text
--calibration-permutations 1000   --calibration-start-row 1
--inference-permutations  10000   --inference-start-row  1001
```

Before any GPU work, both stages validate the master file and reject:
overlapping row ranges; duplicate rows; a row 0 that is not the observed
assignment; the observed assignment reappearing as a null; a file too short
for the declared ranges; and a row count that does not match the partition
exactly (`--allow-extra-permutation-rows` to use a longer file deliberately).
`run_bundle_fwer.py` additionally refuses to finish if the assembled null
distribution is not exactly the declared inference rows, so a calibration row
cannot leak into a resumed run.

### Interpretation of the existing run

The completed analysis used rows 1–200 for calibration and rows 1–10000 for
inference, so its calibration set was a 200-row *subset* of the production
null. That was not selection from the observed row, and a diagnostic
recalculation found a negligible numerical effect — but it is less clean than
a held-out design, and it remains reported accurately as the implementation
used for the current draft. The disjoint 1,000 + 10,000 design above applies
to the eventual combined Fisher-transform and covariate-adjusted confirmatory
rerun. Re-running the old design requires stating the old row ranges
explicitly (`--calibration-start-row 1 --calibration-permutations 200
--inference-start-row 1 ...` is now rejected as overlapping, which is the
point).

## Covariate-adjusted Freedman--Lane model

> **Status: implemented end to end and validated on GPU; not yet run in
> production.** See `manuscript/ANALYSIS_DECISIONS.md` (2026-09-02,
> "covariate-adjusted control--TLE analysis").

The completed control--TLE analysis permutes group labels with no
participant-level nuisance covariates. The confirmatory analysis estimates the
group term while adjusting for the demographics available for all 68
participants:

```text
r_ie = beta_0e + beta_Ge * group_i + beta_Ae * centered_age_i
                                   + beta_Se * sex_i + error_ie
```

### The statistic: HC2, because HC2 *is* Welch

The existing pipeline deliberately uses Welch's unequal-variance t, so the
adjusted statistic must not quietly become a pooled-variance one. The
resolution is not a compromise: for a two-group design with no covariates, the
**HC2**-studentized group coefficient equals Welch's t *exactly* (asserted to
12 decimal places in `regression_freedman_lane.py`). HC0 and HC3 do not.
Adding covariates therefore changes the model without changing the variance
assumptions, and the adjusted analysis is a strict generalization of the
published one rather than a different statistic.

### The permutation scheme

Covariate adjustment is part of permutation inference; it cannot be applied to
an observed result after the fact. `freedman_lane.py` implements
Freedman--Lane residual permutation (Winkler et al., *NeuroImage* 2014;
92:381--397): fit the reduced model `Z = [intercept, age, sex]`, permute its
residuals, add the nuisance fitted values back, then fit `X = [Z, group]` and
studentize. The same participant permutation is used for every edge, so
downstream thresholding and bundling are unchanged.

This needs **full-index permutations**: a complete reordering of all 68
residual vectors. The existing files store only the membership of group A and
cannot be reused. Generate the new representation with:

```bash
python 02_cudaPerm/generatePermutations.py \
  -nA 26 -nB 42 -o permutations_fullindex.txt \
  --representation full-index --seed <seed>
```

`--representation group-a` remains the default, so the unadjusted Welch
pipeline is untouched. Validation rejects a group-membership file supplied
where a full-index one is required, and vice versa.

### Why 1.78 billion edges x 10,001 permutations is affordable

Naively every (edge, permutation) pair needs its own regression. The algebra
collapses. Writing `M_Z`, `M_X` for the residual makers and
`a = M_Z g / (g'M_Z g)` for the Frisch--Waugh contrast, two orthogonality
facts (`a'Z = 0` and `M_X H_Z = 0`) mean a Freedman--Lane draw depends on the
data only through the **nuisance residuals** `u = M_Z y`, which are computed
*once per edge* and reused by every permutation:

```text
numerator       = a' P u
denominator^2   = u' (P' K P) u        K = M_X' diag(a_i^2/(1-h_ii)) M_X
```

`K` is one fixed matrix; `P'KP` only relabels it. Packing the upper triangle
of `u u'` turns the whole permutation set into two dense matrix products per
edge-chunk:

```text
numerators   = W  @ U       W:  (n_perm, 68)     U:  (68, n_edges)
denominators = KP @ UU      KP: (n_perm, 2346)   UU: (2346, n_edges)
```

`freedman_lane.py` emits `W` and `KP` — 92 MiB of float32 at n=68, B=10001,
small enough to stay resident on the GPU. float32 is safe here: `K` is
positive semidefinite and the packed form is well conditioned, so the absolute
error on `t` stays below 2.2e-6 (measured over 2000 edges x 51 permutations).
Relative error does grow near `t = 0`, where the numerator cancels, but those
edges are nowhere near any cluster-forming threshold.

Measured cost: 48.3 MFLOP per edge, **86 PFLOP** for the whole graph. That is
17 min at RTX 4090 peak fp32 and roughly 30--50 min at realistic GEMM
efficiency — the same order as the existing pipeline, whose real-world cost is
dominated by disk I/O anyway.

### The CUDA backend

`permutationTest_cuda_bundle --freedman-lane PLAN` switches the edge statistic
from unadjusted Welch to adjusted HC2 without touching the rest of the
pipeline: the same sparse format, the same C++ bundler, the same FWER
machinery. The Welch path is unchanged and remains the default.

Per part it runs `u = M_Z y` **once, in place** (only `p_z` temporaries are
needed, so no scratch buffer and no halving of the part size), then one
thresholding kernel per permutation reusing those residuals. Both kernels use
the projector identities above, so no `n x n` matrix is ever formed.

Two implementation details carry most of the performance:

- **Subject-major staging.** The Welch path stores a part edge-major
  (`[edge][subject]`). One thread per edge then strides `n_subjects * 4` bytes
  and every warp touches 32 separate cache lines. `readRowsSubjectMajor`
  loads the same rows as `[subject][edge]` instead — which is also a straight
  copy rather than a strided scatter, since each subject's chunk is already
  read contiguously — so consecutive threads read consecutive addresses. This
  alone was worth 2.3x. A shared-memory staging buffer fixes the coalescing
  too, but costs enough shared memory to cap occupancy near 16%, and measured
  slower.
- **Two passes, not one.** The squared denominator can be algebraically
  expanded to `A - 2 d.b + d'Cd`, computable in a single pass over the data
  (`C` turns out to be permutation-invariant). That halves DRAM traffic, but
  it is the classic unstable computational formula: measured on 4000 edges it
  raised the absolute error on `t` from 2.3e-6 to 1.2e-5. Since accuracy near
  the cluster-forming threshold decides bundle membership, the two-pass form
  is kept.

Measured on 68 subjects x 1.12M edges, taking the marginal cost between 201-
and 1201-permutation runs so fixture load and file writes cancel:

| Variant | ns per edge-permutation | Full run (1.78e9 edges x 10,001) |
|---|---:|---:|
| edge-major, no staging | 1.87 | 9.2 h |
| shared-memory staging | 1.30 | 6.4 h |
| **subject-major (current)** | **0.56** | **~2.8 h** |

At 0.56 ns the kernel moves ~971 GB/s, which is essentially the RTX 4090's
DRAM bandwidth — the statistic is memory-bound, not compute-bound, so further
gains would need L2 blocking (looping permutations inside an edge sub-chunk
that fits in L2) rather than cheaper arithmetic.

### Usage

```bash
# 1. design matrix (records its own coding, centering and contrast)
.venv/bin/python 02_cudaPerm/design_matrix.py \
  --file-list participants.txt --group-a-subjects 26 \
  --output-dir adjusted/design

# 2. Freedman-Lane tables for the held-out inference rows (row 0 + 1001..11000)
.venv/bin/python 02_cudaPerm/freedman_lane.py \
  --design adjusted/design/design.npz \
  --permutations permutations_fullindex.txt \
  --output-dir adjusted/tables

# 3. calibrate p_CF on the held-out calibration rows (adjusted null geometry)
.venv/bin/python 02_cudaPerm/percolation_calibration.py \
  participants.txt permutations_fullindex.txt adjusted/calibration \
  --neighbor-dist 1.0 --min-size 10 --min-cluster-voxels 6

# 4. adjusted inference on row 0 + the held-out inference rows
.venv/bin/python 02_cudaPerm/run_bundle_fwer.py \
  participants.txt permutations_fullindex.txt adjusted/results \
  --cluster-forming-p <calibrated> \
  --freedman-lane-plan adjusted/tables/freedman_lane_plan.flp \
  --statistic mass --neighbor-dist 1.0 --min-size 10 --min-cluster-voxels 6 \
  --bundle-method strict --bundle-engine cpp --bundle-threads 16 \
  --capacity 20000000 --batch-size 1000
```

`run_bundle_fwer.py` records `edge_statistic`, the plan's SHA-256 and the
permutation representation in `bundle_fwer_config.json`, and rejects a
group-membership permutation file before any GPU work. Step 3 is **not
optional**: `p_CF = 5e-6` was calibrated on the unadjusted Welch null and does
not carry over.

`design_matrix.py` encodes the covariate audit as behaviour, not commentary:

- **age** centered on the analysis-sample mean, **sex** as a 1 = female
  indicator — both recorded in `design_manifest.json` with the exact mean
  subtracted and reference level;
- **handedness** is *refused* as a primary covariate. All six left-handed
  participants are patients, so group and handedness are not separable;
  `--restrict-handedness R` runs the preferred 62-participant sensitivity
  analysis (26 controls, 36 patients) instead, and `--allow-confounded` is
  required to override deliberately;
- **run count** is available only on explicit request, as a
  measurement-precision covariate, never added by default;
- **motion** has no entry at all: no motion summary was delivered with this
  dataset, and none is inferred. The manifest records that absence explicitly;
- a rank-deficient design is rejected rather than silently pseudo-inverted.

### Correctness

`freedman_lane.py` stays the oracle: `regression_cuda_freedman_lane.py` runs
the real CUDA backend on a synthetic fixture with heteroscedastic noise, real
covariate signal and a planted effect, and requires that the GPU select
*exactly* the same suprathreshold edge set as the Python reference on every
permutation. Measured agreement is 7.3e-7 on the statistic. It also drives
`run_bundle_fwer.py` end to end and checks that the planted spatial effect
survives bundle-FWER, that the calibration row is never computed, and that the
denominator is the inference count plus one.

`freedman_lane.py` carries two independent implementations that are checked
against each other (`statistics` via the packed GEMM form, and
`statistics_projector` via the projector form the kernel uses); they agree to
1.8e-15.

### One documented approximation

The df-aware threshold uses a fixed residual df of `n - rank(X)` = 64. Unlike
Welch's edge-specific Satterthwaite df, an HC-studentized coefficient has no
exact small-sample null distribution, so that df is used *only* to convert a
cluster-forming p into a `|t|` threshold — never to report a p-value. FWER
control comes from the permutation distribution, which does not depend on it.

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
.venv/bin/python -m unittest 02_cudaPerm/regression_permutation_partition.py
.venv/bin/python -m unittest 02_cudaPerm/regression_freedman_lane.py
.venv/bin/python -m unittest 02_cudaPerm/regression_cuda_freedman_lane.py
```

`regression_permutation_partition.py` covers the row partition over a small
synthetic permutation file whose calibration rows, inference rows, exceedance
count and denominator are known exactly, plus the calibration stability
assessment and the full-index permutation representation.

`regression_freedman_lane.py` covers the covariate-adjusted model: the exact
HC2/Welch equivalence, the two-GEMM path against a literal per-draw
regression, float32 error bounds, design-matrix coding and refusals, and
exchangeability — that uncorrected p-values really are uniform under the null
when heteroscedastic noise and genuine age/sex signal are present, and that
covariate structure alone does not manufacture group effects.

`regression_cuda_freedman_lane.py` runs the real CUDA backend against that
oracle and drives the adjusted pipeline end to end; it is skipped without a
GPU or a built backend.

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
  -nA <n_group_A> -nB <n_group_B> -o permutations.txt --seed <seed>
```

This writes **11,001 rows** by default: row 0 observed, 1,000 calibration
nulls, 10,000 inference nulls. It validates what it wrote and emits
`permutations.txt.partition.json` recording the seed, row ranges, SHA-256, the
unique-row count, and the resulting FWER denominator.

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

`01p5_FisherCC/fisher_aggregate_ccmat.py` is its planned successor: it does
the same run-to-participant aggregation but on the Fisher `z` scale
(`atanh(r)`), which is better behaved for averaging and linear group
modelling than raw `r`. Its `raw-equal` mode reproduces `average_ccmat_runs.py`
bitwise, so the two scales can be compared without an implementation change
confounding the comparison. It also emits a per-file provenance sidecar and a
run manifest. It is implemented and validated but **not yet run in
production** — see `01p5_FisherCC/README.md` and the decision log in
`manuscript/ANALYSIS_DECISIONS.md`. Note that a switch to Fisher `z`
invalidates the existing `p_CF = 5e-6` calibration, which was measured on
raw-`r` participant matrices; `percolation_calibration.py` must be rerun
before any inference on Fisher `z` inputs.

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
