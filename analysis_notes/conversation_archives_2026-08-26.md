# MOCCA conversation archive — 2026-08-26

## Scope and fidelity

This file records the working conversation between the user and Codex through
2026-08-26 concerning permutation inference, bundle-level FWER, optimization,
and cleanup of the MOCCA repository.

The most recent exchanges are represented directly. Earlier parts of the
thread underwent automatic context compaction, so their assistant replies are
reconstructed from the retained technical record rather than claimed to be
word-for-word quotations. User questions, decisions, paths, configurations,
run timings, and implementation outcomes have been preserved as faithfully as
possible.

This archive intentionally excludes system/developer instructions and contains
only project-relevant discussion.

## Initial problem: asymmetric LTLE versus RTLE results

The conversation began with the observation that the uncorrected
`LTLEvsRTLE` results were strongly asymmetric: the positive direction
(left greater than right) yielded many connections, while the negative
direction yielded almost none, and none with default values.

The first statistical clarification was whether permutation testing removes
the need for FWER/FDR correction.

**User:**

> I thought FWER or FDR correction is not necessary with permutation testing?

The conclusion was that permutation testing supplies valid null distributions
and edgewise p-values, but edgewise permutation p-values remain uncorrected
when thousands or billions of hypotheses are tested. A correction is still
needed unless the permutation statistic itself incorporates the multiplicity,
for example through a maximum statistic.

The existing controls-versus-patients CSVs were inspected and found to be based
on uncorrected per-edge permutation p-values as well.

## Edgewise maximum-statistic FWER runs

The conversation then addressed whether corrected values could be recovered
from existing p-value files. For maximum-statistic FWER, the full permutation
statistics across all edges are needed; the stored uncorrected observed
p-values are insufficient. Therefore permutation testing had to be rerun.

The data became available under `/mnt/storage/`.

The number of permutations was discussed. A 100,000-permutation run was chosen
for precise tail resolution:

\[
p_{\min}=\frac{1}{B+1}.
\]

Thus 100,000 permutations give a minimum attainable corrected p-value near
`0.00001`, while 10,000 permutations give approximately `0.0001`.

The following subject-mean, two-tailed, edgewise maximum-statistic FWER jobs
were run in tmux:

- Controls versus patients: 26 versus 42 subjects, 100,000 permutations.
- LTLE versus RTLE: 24 versus 13 subjects, 100,000 permutations.

Relevant scripts retained in the active module are:

- `02_cudaPerm/run_controls_vs_patients_subject_fwer_100k.sh`
- `02_cudaPerm/run_ltle_vs_rtle_subject_fwer_100k.sh`
- `02_cudaPerm/average_ccmat_runs.py`

The subject-level analysis was explicitly preferred over treating repeated runs
as independent observations.

## Why edgewise FWER produced little or no significance

The lack of edgewise FWER discoveries was discussed. The main conclusion was
that this did not necessarily indicate a broken analysis. The atlas-free
connectivity matrix contains approximately:

\[
\frac{59677\times59676}{2}=1,780,642,326
\]

tested connections. This is substantially more severe than a conventional
voxelwise analysis with tens or hundreds of thousands of voxels. A global
maximum-edge correction over 1.78 billion tests is consequently extremely
conservative.

The user ruled out atlas reduction because atlas-free discovery is a central
goal, and did not want to replace the nonparametric design with a conventional
GLM model.

This led to the proposal to perform inference at the level of spatially
coherent edge bundles rather than individual edges.

## Bundle-level permutation FWER design

The agreed bundle-level procedure was:

1. Compute Welch t statistics for every edge in the observed grouping and in
   every label permutation.
2. Apply a two-sided cluster-forming threshold.
3. Split positive and negative effects.
4. Apply the existing COFFEE-DAC isolation, strict bundling, pruning,
   minimum-size, and endpoint-cluster stages.
5. Compute a statistic for every surviving bundle.
6. Store the maximum bundle statistic across both signs for each permutation.
7. Correct an observed bundle using:

   \[
   p_\mathrm{FWER}=
   \frac{1+\#\{M_b\geq S_\mathrm{observed}\}}{B+1}.
   \]

The default bundle statistic became cluster mass:

\[
S=\sum_{e\in C}(|t_e|-u_e),
\]

where `u_e` is the edge's cluster-forming critical t value. Extent remains an
available alternative.

The inferential claim is at the bundle level. It does not make every edge in a
significant bundle individually significant.

## Initial reference implementation

The user requested that established, working COFFEE-DAC routines not be
modified. New versions were therefore introduced alongside them.

The following experimental bundle-FWER files were implemented in
`02_cudaPerm/`:

- `bundle_fwer.py`
  - Readable Python reference/oracle.
  - Provides `compute_bundle_statistics()` and `max_bundle_statistic()`.
- `regression_bundle_fwer.py`
  - Regresses the reference implementation against four historical CSV/cache
    pairs.
- `permutationTest_cuda_bundle.cu`
  - Separate CUDA backend that scans all edges and writes only sparse
    suprathreshold records.
- `run_bundle_fwer.py`
  - Persistent controller for the observed row and null permutations.
- `bundle_fwer_omp.cpp`
  - Optimized C++/OpenMP implementation of deterministic bundle formation.
- `regression_bundle_fwer_cpp.py`
  - C++ versus Python oracle regression.
- `regression_cuda_bundle.py`
  - Tiny end-to-end GPU regression.

The established edgewise CUDA executables were not altered by this work.

## First performance benchmark and C++ optimization

The first 200-permutation LTLE/RTLE bundle test used the Python bundle
implementation with fixed `|t| >= 3.9`. It took approximately 8 hours,
40 minutes, and 42 seconds, with peak memory around 36 GiB. CUDA itself took
about 11 minutes; Python bundle construction accounted for nearly all the
remaining time.

The bottleneck motivated a C++/OpenMP replacement with:

- Compact voxel-index edge storage.
- Spatial hashing/voxel-neighbour lookup instead of quadratic Python pair
  scanning.
- Union-find for strict bundles.
- Exact synchronous fixed-point intra-bundle pruning.
- Endpoint spatial-cluster pruning.
- OpenMP parallelism across independent permutation files.

The optimized 201-row observed-plus-null benchmark completed in 11 minutes,
48.59 seconds, a total speedup of 44.09 times.

The equivalence audit found:

- All 201 threshold-edge, retained-edge, and bundle counts identical.
- All 4,138,514 observed endpoint coordinates and bundle/network labels
  identical.
- All 2,940 observed bundle IDs, signs, edge counts, and corrected p-values
  identical.
- Only small floating-point summation-order differences in mass, with no
  corrected p-value changes.

## Fixed-threshold 10K LTLE/RTLE benchmark

A 10,000-permutation LTLE-versus-RTLE subject-mean benchmark was run with:

- Fixed `|t| = 3.9`.
- Two-sided positive/negative maximum distribution.
- Bundle mass.
- Strict parameters `neighbor_dist=1`, `min_size=10`,
  `min_cluster_voxels=6`.
- Four C++ bundle threads.

Output directory:

`/mnt/storage/MOCCA_UCLA/bundle_fwer_LTLEvsRTLE_10k_t390_cpp`

Timing:

- CUDA: 6,455.272 seconds (`1h47m35s`).
- C++ bundle stage: 1,510.053 seconds (`25m10s`).
- Total: `2h12m45s`.
- Peak RSS: approximately 36.7 GiB.

The C++ engine was not fixed to four threads; four was simply the conservative
controller setting used for that run. The machine has 32 CPU cores and 256 GB
RAM. Thirty threads were considered feasible for fixed-threshold jobs, although
memory bandwidth and dense outlier permutations limit scaling.

## Cluster-forming threshold discussion

The arbitrary fixed `|t|=3.9` threshold was recognized as unsuitable for final
inference. The intended production rule became a two-sided, uncorrected,
df-aware cluster-forming threshold of `p_CF = 0.001`.

Welch degrees of freedom vary by edge and permutation, so the critical t value
must also vary. Approximate ranges discussed were:

- Controls/patients: typical critical t around 3.48, approximately 3.44–3.73
  over the possible df range.
- LTLE/RTLE: typical critical t around 3.73, approximately 3.59–4.32 over the
  possible df range.

It was emphasized that controls versus patients is the primary analysis. The
LTLE/RTLE comparison is a smaller and less balanced subgroup analysis.

## Mistaken controls/patients launch and correction

When asked to run controls versus patients for 10,000 permutations, Codex
initially interpreted “as well” as requesting the same fixed `|t|=3.9`
benchmark configuration. The job was launched and then immediately challenged
by the user:

> Hold on a minute, didn't we decide on equivalent t for p<0.001 for the t
> value threshold?

The fixed-threshold job was stopped. Its tmux child initially became orphaned
through the logging pipeline and was explicitly terminated. It produced no
usable scientific results, only empty sparse headers. The incomplete directory
was left as a record rather than silently deleted:

`/mnt/storage/MOCCA_UCLA/bundle_fwer_controlsVSpatients_10k_t390_cpp`

This was acknowledged as an assistant error.

## Df-aware CUDA extension

At the user's request, the bundle-FWER path was extended to support exact
per-edge/permutation Welch df-aware thresholding.

For every edge and permutation, CUDA now calculates:

\[
t=\frac{\bar{x}_A-\bar{x}_B}
{\sqrt{s_A^2/n_A+s_B^2/n_B}}
\]

and:

\[
\nu=\frac{(s_A^2/n_A+s_B^2/n_B)^2}
{(s_A^2/n_A)^2/(n_A-1)+(s_B^2/n_B)^2/(n_B-1)}.
\]

The two-sided critical t is obtained from a high-resolution lookup table over
Welch df, and an edge survives when `|t| >= tcrit(df)`.

The sparse format was versioned:

- v1: fixed-threshold records (`edge_index`, `t`).
- v2: df-aware records (`edge_index`, `t`, threshold excess).

Fixed-threshold compatibility was retained.

Validation included:

- Six CPU-side regressions.
- Four historical CSV/cache equivalence cases.
- A v2 sparse-format mass test.
- Two end-to-end GPU tests, one fixed and one df-aware.
- Independent comparison with SciPy's Welch statistic, fractional df, and
  critical t.

All tests passed.

The production controller option became:

```bash
--cluster-forming-p 0.001
```

## Controls versus patients: df-aware 10K bundle FWER

The corrected controls-versus-patients analysis was launched with:

- 26 controls and 42 patients.
- Subject means.
- 10,000 null permutations plus the observed row.
- Df-aware, two-sided `p_CF=0.001`.
- Bundle mass.
- Strict parameters `1/10/6`.
- 16 C++ bundle threads.
- Sparse capacity 20,000,000 edges per CUDA part.

Output directory:

`/mnt/storage/MOCCA_UCLA/bundle_fwer_controlsVSpatients_10k_p001_dfaware_cpp`

Timing:

- CUDA: 11,759.082 seconds (`3h15m59s`).
- C++ bundle stage: 1,449.522 seconds (`24m10s`).
- Total wall time: `3h40m09s`.
- Exit status: 0.
- Peak RSS: approximately 30.1 GB.

The result manifest recorded:

- 10,000 null permutations.
- 2,435 observed bundles.
- Minimum attainable FWER p approximately `0.00009999`.
- One observed bundle with `p_FWER <= 0.05`.

The significant bundle was negative (patients greater than controls under the
established group ordering):

- Bundle ID: 880.
- Edges: 6,283,411.
- Mass: approximately 2,406,904.
- Corrected p-value: `0.0326967`.
- Null exceedances: 326 of 10,000.

## Giant-component/percolation problem

The user immediately noted that a six-million-edge component is not a useful
anatomical “bundle.” Quantitative inspection confirmed that it was a giant
percolating component:

- 6,283,411 edges.
- 75.5% of all retained observed edges.
- 58,909 of 59,677 mask voxels touched (98.7%).
- Full mask bounding box.
- Negative t range approximately `-8.11` to `-3.44`.

The largest positive component also contained 1,581,948 edges and touched
40,404 voxels.

At `p_CF=.001`, 1.78 billion edge tests imply approximately 1.78 million
suprathreshold edges under a simple null expectation before accounting for
dependence. The transitive bundle relation therefore enters a percolation
regime. Local admissible merges chain across the brain, even under the “strict”
rule.

The permutation FWER result remains internally valid for the statistic that was
defined. Its interpretation is:

> A brain-wide negative suprathreshold connectivity component has greater mass
> than expected under label permutation.

It is not evidence for a localized anatomical bundle.

## Historical uncorrected thresholds

The user recalled that historical `runAll` analyses appeared anatomically more
useful with:

- Controls/patients: edgewise permutation `p < .0001`.
- LTLE/RTLE: edgewise permutation `p < .0005`.

Inspection showed:

- Controls CSVs contained only the discrete `p=.0001` bin.
- LTLE/RTLE CSVs contained bins `.0001`, `.0002`, `.0003`, and `.0004`.

At 10,000 permutations, `.0001` is effectively the permutation resolution
floor. These historical empirical p-value thresholds are therefore not the
same object as a continuous parametric df-aware cluster-forming p-value.

The historical `runAll` inputs also treated repeated runs as observations,
whereas current inference uses subject means. The old and new results are not
directly comparable.

It was clarified that sample size alone does not justify changing `p_CF`.
Df-aware p-values already account for sample size. `p_CF` primarily controls
suprathreshold graph density and topology.

## Threshold-selection options

Three approaches were discussed.

### Null-percolation calibration

Use calibration permutations to locate the transition at which one bundle
begins to dominate edge fraction and voxel coverage, then choose a cutoff on
the conservative side. This is computationally inexpensive but requires an
explicit calibration rule and preferably independent inference permutations.

### Multi-threshold/grid-corrected inference

Predefine a threshold grid:

```text
0.001, 0.0005, 0.0002, 0.0001,
0.00005, 0.00002, 0.00001
```

For every permutation and threshold:

1. Form bundles and calculate the maximum statistic across both signs.
2. Convert the maximum to a threshold-specific permutation rank, since raw
   mass scales are not directly comparable across thresholds.
3. Take the most extreme rank across thresholds for each permutation.
4. Use the permutation distribution of this minimum rank to correct the
   observed search over thresholds.

This is a permutation minP/maximum-statistic correction over:

- All bundles.
- Both signs.
- All candidate cluster-forming thresholds.

Each observed bundle can receive a threshold-specific FWER p-value and an
additional grid-adjusted FWER p-value.

The grid uses parametric df-aware cluster-forming p-values, so its values do not
depend on the number of permutations. The permutation count controls only the
resolution and stability of the final corrected bundle p-values.

The intended implementation is a new, separate path:

- CUDA stores `t` and Welch df once for all edges passing the most liberal
  `.001` threshold.
- The sparse record remains 16 bytes (`edge_index`, `t`, `df`).
- C++ reuses each sparse permutation at every stricter threshold.
- CUDA runs only once, not once per grid value.
- Estimated controls/patients or LTLE/RTLE 10K runtime: roughly 4.5–6 hours,
  depending on topology and I/O.

The user judged grid correction more generally defensible than selecting one
cutoff from null percolation.

### True TFCE/TFNBS

True threshold-free component enhancement would integrate over many levels,
often starting near t=0. For this 1.78-billion-edge atlas-free space, bundle
formation at liberal thresholds would be prohibitively expensive and dominated
by giant components. It was therefore rejected as impractical for the current
pipeline.

## Agreed next statistical implementation

The user authorized implementation of the seven-threshold corrected grid and a
10,000-permutation LTLE-versus-RTLE run.

That implementation was paused before source changes because the user requested
repository cleanup and archival first.

## Archiving the original uncorrected-p pipeline

On 2026-08-26, the user requested that original routines associated with the
uncorrected edgewise p-threshold workflow be moved into per-module archive
directories and kept safe from further routine editing.

### Module 02 archive

Location:

`02_cudaPerm/archives/uncorrected_p_pipeline_2026_08_26/`

Contents:

- `permutationTest_cuda.cu`
- `permutationTest_omp.c`
- `validate_pvalues.sh`
- `generateTestData/`
- `CHANGELOG.md`
- Independent archive `CMakeLists.txt`
- Pre-archive compiled binaries
- Archive README/manifest

Shared/current files intentionally retained in the active root include:

- `generatePermutations.py`
- `permutationTest_cuda_fwer.cu`
- `ccmat_io.*`, `perm_kernels.*`, and `results_io.*`
- Subject-averaging and bundle-FWER sources

The active CMake build no longer includes the two archived legacy targets.

### Module 03 archive

Location:

`03_prepResultsForVisualization/archives/uncorrected_p_pipeline_2026_08_26/`

Contents:

- `permout_to_csv.py`
- `find_pvalue_threshold.py`
- `quick_threshold_count.sh`
- `split_pos_neg_tstat.py`
- `apply_fdr.py`
- Original CUDA permutation conversion notebook
- Historical README

The AFNI utilities were retained because they are not specific to uncorrected
edgewise p-values.

### Module 04 archive

Location:

`04_coffee-dac/archives/uncorrected_p_pipeline_2026_08_26/`

Contents:

- `coffee_dac_pipeline.py`
- `coffee_dac_pipeline_v2.py`
- `run_pipeline.py`
- `run_pipeline_v2.py`
- `export_networks_to_gif.py`
- `visualizer_local.py`
- `my_colormap.py`
- Historical method documentation and notes

Small top-level compatibility modules re-export the frozen v1/v2 APIs so that
the current GUI and existing caches continue to work. Input CSVs, result caches,
figures, meshes, and `mocca_gui/` were not moved. The pre-existing
`04_coffee-dac/archives/results_archives/` directory was preserved untouched.

Archive verification completed successfully:

- Archived tracked sources byte-match their pre-move Git contents.
- Archived CUDA and OpenMP sources build independently.
- Active corrected CUDA/C++ targets build.
- COFFEE-DAC compatibility imports work.
- Six bundle regression tests pass.
- `git diff --check` passes.

## Current repository state and pending work

At the time of this conversation archive:

- The original uncorrected-p routines have been archived.
- The validated single-threshold df-aware bundle-FWER path remains active.
- The controls/patients `.001` result is complete and retained as an
  exploratory global-component result.
- The seven-threshold grid implementation has not yet been written.
- The authorized next analysis is LTLE versus RTLE, subject means, 10,000
  permutations, corrected over the full threshold grid.

The planned implementation sequence is:

1. Add a separate v3 CUDA sparse backend storing edge index, t, and Welch df.
2. Add a separate C++ multi-threshold bundle engine.
3. Add a controller implementing symmetric threshold-specific ranks and
   min-rank correction over the grid.
4. Regression-test every grid threshold against independent single-threshold
   runs.
5. Launch the LTLE/RTLE 10K grid-corrected run in tmux and verify startup.

## Important output locations

- Fixed-threshold LTLE/RTLE 10K benchmark:
  `/mnt/storage/MOCCA_UCLA/bundle_fwer_LTLEvsRTLE_10k_t390_cpp`
- Df-aware controls/patients 10K result:
  `/mnt/storage/MOCCA_UCLA/bundle_fwer_controlsVSpatients_10k_p001_dfaware_cpp`
- Incomplete mistaken fixed-threshold controls directory:
  `/mnt/storage/MOCCA_UCLA/bundle_fwer_controlsVSpatients_10k_t390_cpp`
- Controls subject-mean edgewise FWER inputs/results:
  `/mnt/storage/MOCCA_UCLA/permout_3mm_controlsVSpatients_subjectMean_100k_fwer`
- LTLE/RTLE subject-mean edgewise FWER inputs/results:
  `/mnt/storage/MOCCA_UCLA/permout_3mm_LTLEvsRTLE_subjectMean_100k_fwer`

## Methodological commitments recorded in the conversation

- Atlas-free discovery remains a core requirement.
- Subject is the inferential unit; repeated runs should be averaged rather than
  permuted as independent observations.
- Positive and negative effects are bundled separately but corrected using one
  joint two-sided maximum distribution.
- Welch t and edge-specific Welch df are used.
- The cluster-forming rule must be specified independently of whether the
  observed result is significant or visually attractive.
- Bundle-level inference must apply exactly the same deterministic operations
  to the observed grouping and every permutation.
- Any search over cluster-forming thresholds must itself be included in the
  permutation correction.
- Existing working or historical routines should be preserved; new inferential
  paths should be implemented as separate versions where practical.
- A statistically significant giant component may be a valid global result,
  but it must not be described as a localized anatomical bundle.

---

# Continuation — 2026-08-26 to 2026-08-27

This continuation records the work performed after the initial archive was
created. These exchanges were still present in the active conversation when
this section was written and are recorded as a detailed chronological account.

## Multi-threshold grid-FWER implementation

The user asked to proceed with the previously agreed threshold-grid method and
run it on LTLE versus RTLE. The active routines were extended; archived legacy
routines were not modified.

The implemented two-sided, df-aware cluster-forming grid was:

```text
0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001
```

The implementation consisted of:

- CUDA sparse format version 3, storing condensed edge index, Welch t, and
  edge-specific Welch-Satterthwaite df in 16 bytes per surviving edge.
- One CUDA pass at the most liberal grid threshold, p_CF = 0.001.
- Reuse of the liberal sparse edge set by the C++ bundler at every stricter
  threshold, with df-aware critical t and threshold excess recomputed for each
  edge and threshold.
- One maximum bundle mass per permutation, threshold, and both signs jointly.
- Symmetric permutation-tail ranking separately at every threshold.
- The minimum threshold-specific rank per permutation as the threshold-search
  statistic.
- `p_grid_fwer` values correcting simultaneously for bundle selection, both
  signs, and selection over the seven cluster-forming thresholds.

The main modified files were:

- `02_cudaPerm/permutationTest_cuda_bundle.cu`
- `02_cudaPerm/bundle_fwer_omp.cpp`
- `02_cudaPerm/run_bundle_fwer.py`
- `02_cudaPerm/regression_cuda_bundle.py`
- `02_cudaPerm/regression_bundle_fwer_cpp.py`
- `02_cudaPerm/README.md`

A reproducible production launcher was added as:

`02_cudaPerm/run_ltle_vs_rtle_bundle_grid_10k.sh`

The implementation passed:

- Existing C++ versus Python bundle-oracle regressions.
- Legacy fixed-threshold and df-aware single-threshold regressions.
- New stored-df rethresholding regression.
- CUDA end-to-end multi-threshold grid regression.
- A real LTLE/RTLE three-row smoke test.

The real smoke test produced 8,577,712 liberal-threshold observed edges and
completed CUDA for observed plus two null rows in 510.964 seconds. Reusing the
sparse files for all seven original bundle passes took 17.330 seconds.

## First LTLE/RTLE 10K grid run using the historical bundler

The full grid analysis was launched in tmux session:

```text
mocca_ltle_grid_10k
```

Output directory:

```text
/mnt/storage/MOCCA_UCLA/bundle_fwer_LTLEvsRTLE_10k_pgrid_dfaware_cpp
```

The saved configuration used:

- 37 subject means.
- 59,677 voxels and 1,780,642,326 possible edges.
- 10,000 null permutations plus observed row 0.
- Bundle mass.
- `neighbor_dist=1`.
- `min_size=10`.
- `min_cluster_voxels=6`.
- Historical transitive “strict” bundling.
- 16 C++ bundle threads.

The run completed successfully in 3:00:46 with exit status 0. It generated
14,034 observed bundles across thresholds and a minimum attainable corrected p
of 1/10001 = 0.000099990001.

The significant results at grid-FWER alpha 0.05 were four nested positive
components:

| p_CF | Edges in significant component | p_grid_fwer |
|---:|---:|---:|
| 0.001 | 8,206,314 | 0.038896 |
| 0.0005 | 4,169,920 | 0.039896 |
| 0.0002 | 1,534,919 | 0.043596 |
| 0.0001 | 594,380 | 0.049095 |

No negative, RTLE-greater-than-LTLE bundle survived grid-FWER 0.05. The
stricter grid points broke the component down further but did not remain
significant:

| p_CF | Largest component edges | p_grid_fwer |
|---:|---:|---:|
| 0.00005 | 142,777 | 0.065293 |
| 0.00002 | 13,374 | 0.126787 |
| 0.00001 | 1,570 | 0.293671 |

## First grid-result visualization export

The active visualization-preparation module was extended with:

`03_prepResultsForVisualization/prepare_bundle_grid_fwer.py`

It streams selected bundle edges and creates raw CSVs plus matching COFFEE-DAC
v2 processed caches, empty linkage placeholders, provenance manifests, and
summaries. The edge `pvalue` column is populated with the bundle-level
`p_grid_fwer` value. Each cluster-forming threshold remains a separate
representation and is never pooled across thresholds.

The four significant historical-bundler representations were exported under:

```text
/mnt/storage/MOCCA_UCLA/bundle_fwer_LTLEvsRTLE_10k_pgrid_dfaware_cpp/
  visualization_grid_fwer_alpha_0p05/
```

All generated caches passed checksum, row-count, label, and corrected-p-value
validation. Manual inspection nevertheless showed that even the most
restrictive significant representation, p_CF = 0.0001 with 594,380 edges,
still behaved as one brain-spanning component rather than a useful anatomical
bundle.

## Discovery of transitive chaining in the historical “strict” bundler

Inspection of both Python and C++ implementations established that the
historical strict bundler uses union-find. Pairwise-compatible edges are united,
and union-find then takes the transitive closure. Consequently, if A matches B
and B matches C, all three become one bundle even when A and C do not satisfy
the stated strict criterion directly.

This contradicted the old documentation, which claimed that an intermediate
edge could not bridge two otherwise incompatible edges. The statistical result
was still valid for the algorithm actually run because exactly the same
deterministic transformation was applied to observed and null permutations.
Its anatomical interpretation as a localized bundle, however, was not valid.

The user clarified that the original COFFEE-DAC bundler had been designed for
roughly 50,000–200,000 whole-brain edges. At that density, reasonable bundles
could form empirically. At roughly one million or more widespread edges, the
same adjacency relation crossed its percolation transition and formed a giant
component.

## Bounded non-chaining bundling experiment

A proposed replacement defined a bundle as connections between two compact
endpoint patches. It used the strongest unassigned edge as a representative.
A candidate edge could join only when its endpoints could be oriented so that
each endpoint was within `neighbor_dist` of the corresponding representative
endpoint. This prevented any transitive chain from expanding a bundle beyond
the representative’s fixed envelope.

The experiment was implemented as a separate executable and controller mode:

- `bundle_fwer_bounded_omp`
- `--bundle-method bounded`

The historical executable and default behavior were retained. With
`neighbor_dist=1`, the bounded construction imposed two 3×3×3-voxel endpoint
patches, providing a theoretical maximum of 27 × 27 = 729 unique edges per
bundle.

Regression tests demonstrated that:

- A synthetic chain merged into one component under the historical bundler but
  was split into three bounded bundles.
- Reversing sparse-record order produced identical bounded output.
- A complete 15,625-edge dense endpoint-pair field produced a largest bounded
  bundle of 64 edges and respected the 729-edge upper bound.
- All historical strict-bundling regressions continued to pass unchanged.
- CUDA/controller selection of the bounded grid mode passed end to end.

The real observed LTLE/RTLE smoke test gave:

| p_CF | Observed bundles | Median edges | Largest bundle |
|---:|---:|---:|---:|
| 0.001 | 109,997 | 42 | 667 |
| 0.0005 | 60,371 | 39 | 627 |
| 0.0002 | 26,170 | 36 | 561 |
| 0.0001 | 13,307 | 34 | 470 |
| 0.00005 | 6,615 | 32 | 378 |
| 0.00002 | 2,377 | 29 | 308 |
| 0.00001 | 1,090 | 29 | 241 |

Thus, the fixed envelope successfully removed giant components, but it also
produced very many small, spatially regular bundles.

## Full bounded LTLE/RTLE grid run

The bounded 10K run was launched in:

```text
tmux session: mocca_ltle_bounded_grid_10k
output: /mnt/storage/MOCCA_UCLA/bundle_fwer_LTLEvsRTLE_10k_pgrid_bounded_cpp
```

The first attempt stopped safely after 49:44 because permutation 1488 selected
10,000,528 sparse edges in CUDA part 7, slightly exceeding the configured
10,000,000-edge per-part capacity. No truncated result was accepted.

The run was resumed from the same output directory with capacity increased to
20,000,000 edges per CUDA part. The incomplete sparse batch was overwritten.
The resumed run completed successfully in 9:18:57 with exit status 0. Including
the failed attempt, approximately 10 hours 9 minutes of computation were used.

The completed result contained 219,927 observed bundles across all thresholds.
No bundle survived grid-FWER alpha 0.05. Best results by threshold were:

| p_CF | Best p_grid_fwer | Best bundle edges | Sign |
|---:|---:|---:|:---:|
| 0.001 | 0.173683 | 605 | positive |
| 0.0005 | 0.172883 | 560 | positive |
| 0.0002 | 0.171483 | 481 | positive |
| 0.0001 | 0.149685 | 427 | positive |
| 0.00005 | 0.154285 | 378 | positive |
| 0.00002 | 0.170183 | 258 | positive |
| 0.00001 | 0.166983 | 208 | positive |

## Bounded-result visualization and rejection

Because no bounded bundle passed alpha 0.05, the corrected visualization export
contained only an empty-results summary. For manual diagnostic inspection, a
separate and explicitly labeled exploratory export was created with the ten
smallest-p bundles at each threshold:

```text
/mnt/storage/MOCCA_UCLA/bundle_fwer_LTLEvsRTLE_10k_pgrid_bounded_cpp/
  visualization_exploratory_top10_per_threshold/
```

These 70 bundles retained their actual non-significant `p_grid_fwer` values in
the edge tables and manifests. All seven caches passed validation. They were
never represented as significant findings.

Manual inspection led the user to reject the bounded construction for two
reasons:

1. It produced far too many bundles and no corrected discoveries.
2. The fixed 3×3-voxel endpoint patches looked arbitrary and anatomically
   unconvincing.

The active workflow was therefore returned to the historical transitive
bundler. The production launcher now explicitly requests:

```text
--bundle-method strict
```

The bounded implementation and output were preserved for provenance but marked
as a rejected experiment. Its launcher and methodological record now live
under:

```text
02_cudaPerm/archives/bounded_bundling_experiment_2026_08_27/
```

No new permutation run was started after this decision.

## Current methodological position and next-session problem

The user’s present interpretation is that the historical bundling geometry is
acceptable at the edge densities for which COFFEE-DAC was originally designed,
but not after a liberal threshold leaves a million-scale, whole-brain edge set.
Operationally, a threshold is needed that keeps the suprathreshold edge graph
sufficiently sparse to remain below the historical bundler’s giant-component
or percolation transition.

The remaining methodological constraint is that this threshold cannot be
chosen merely because the observed bundles look good. The next session should
develop a defensible, predeclared rule—potentially based on null-permutation
topology, edge density, or a percolation criterion—and ensure that any threshold
search is included in the permutation correction.

The adopted state at the end of this continuation is therefore:

- Historical transitive strict bundling is active again.
- The seven-threshold grid machinery remains implemented and validated.
- Bounded 3×3 endpoint-patch bundling is rejected but reproducibly archived.
- Historical-grid and bounded-grid result directories remain intact.
- No bounded result is treated as statistically significant.
- Selecting a defensible sparsity/percolation rule is the next open task.
