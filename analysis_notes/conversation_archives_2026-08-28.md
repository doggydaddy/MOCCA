# MOCCA conversation archive — 2026-08-28

## Scope and fidelity

This file records the working conversation between the user and Claude
(Sonnet 5) on 2026-08-27/2026-08-28, continuing directly from
`conversation_archives_2026-08-26.md`. That file ended with the percolation
problem identified but unsolved: the historical strict bundler percolates
into brain-spanning giant components at liberal cluster-forming thresholds,
and the open task was "a defensible, predeclared rule... that keeps the
suprathreshold edge graph sufficiently sparse... without selecting a
threshold merely because the observed bundles look good."

This record is written from the live conversation transcript (no context
compaction occurred), so it is a faithful account rather than a
reconstruction. Technical details, exact commands, file paths, and reported
numbers have been preserved as precisely as possible. It intentionally
excludes system/developer instructions and contains only project-relevant
discussion.

## Devising a null-only percolation calibration method

The user opened by asking for guidance on the unsolved percolation problem.
Claude checked memory (empty at the time) and the prior archive file, then
proposed a formal, general, non-circular calibration method rather than
another manually-chosen threshold:

- **Order parameter:** initially proposed as the fraction of retained
  suprathreshold edges captured by the single largest strict bundle (both
  signs pooled, matching how the production pipeline already takes one
  joint two-sided maximum), with a voxel-coverage variant also proposed as
  a secondary diagnostic.
- **Calibration rule:** sweep a finer cluster-forming-p grid than the
  existing 7-point production grid, purely on an independent batch of null
  permutations (never the observed grouping); find the most liberal
  threshold where a high percentile (not the mean, since the FWER null
  tail is what a percolating outlier would distort) of the order parameter
  stays below a small epsilon (e.g. 1–5%); use one grid step stricter as a
  safety margin.
- This is decided entirely from null data, before the observed bundle
  statistics are ever inspected, so it cannot be "the threshold that made
  the result look good" — directly answering the prior archive's open
  constraint.

The user then asked what the added computational load would be. Claude
worked through the reasoning from the project's own prior benchmarks
(noting the smoke-test timing figure in the prior archive was dominated by
one-time GPU/data-loading overhead, not real per-row marginal cost) and
estimated roughly 5–15 minutes for a 200–500-permutation calibration batch,
recommending a small pilot first (~20–30 permutations) to measure the real
marginal cost empirically before trusting the extrapolation. The user said
"sure do so."

## Implementation: instrumentation + `percolation_calibration.py`

Claude implemented the calibration machinery as an **opt-in addition**,
leaving all existing validated production code paths and their regression
tests untouched:

- `02_cudaPerm/bundle_fwer_omp.cpp`: added a new `--giant-component-report
  FILE` CLI flag (default off). When set, for every permutation processed
  it additionally records `largest_bundle_edges` and `largest_bundle_voxels`
  (computed via `std::max_element` over bundle edge counts, then a
  sort/unique pass over that bundle's endpoint voxels) to a separate CSV,
  without touching the existing `write_maxima()` output schema at all.
  Rebuilt and reran the full existing regression suite
  (`regression_bundle_fwer_cpp.py`, `regression_bundle_fwer.py`'s
  independent pre-existing failure confirmed via `git stash` to predate
  this change, `regression_cuda_bundle.py` including a live GPU end-to-end
  run) — all passed unchanged.
- `02_cudaPerm/percolation_calibration.py` (new file): runs CUDA once at
  the most liberal candidate threshold with `--store-df` (reusing the v3
  sparse format), then re-thresholds the same cached sparse edges at a
  finer grid purely on null rows (`--first-null-row` default 1, always
  skipping row 0/the observed grouping). Default threshold grid:
  `1e-3, 7e-4, 5e-4, 3e-4, 2e-4, 1e-4, 7e-5, 5e-5, 3e-5, 2e-5, 1e-5, 5e-6,
  2e-6, 1e-6`. Outputs `percolation_calibration_curve.csv`,
  `_summary.csv`, and `_results.json` with the recommended threshold.
- Validated end-to-end with a synthetic smoke test (64-voxel toy dataset)
  before touching real data — correctly detected full percolation at every
  tested threshold in that undersized toy case and reported the expected
  "extend the grid" message rather than a false answer.

## First pilot run: a lesson about visibility, not timing

Claude launched a 30-permutation LTLE/RTLE pilot as a plain backgrounded
Bash command (not tmux). It exceeded the 2-minute default timeout and was
auto-backgrounded by the harness. When the user asked whether it was done
and noted they "don't see the tmux session," Claude clarified no tmux had
been used, then diagnosed the real state via `/proc/<pid>/io`: the CUDA
backend was reading through the full ~263 GB LTLE/RTLE connectivity dataset
(37 subjects × ~7.1 GB each, since atlas-free connectivity at 59,677 voxels
means ~1.78 billion edges per subject) at genuine disk speed — ZFS ARC was
capped at ~68 GB (`/proc/spl/kstat/zfs/arcstats`), far below the dataset
size, so it could not be served from cache. This was compounded by the
script's default `--batch-size 16`, which caused the 263 GB reload to
happen twice (rows 1–16, then 17–30) for a 30-permutation pilot.

The user clarified their actual concern: *"I wasn't really that concerned
about timing per say, rather I have difficulty knowing when a run is
complete... runs should be run through tmux so I can track them remotely
too."* Claude saved this as a standing feedback memory (launch long/GPU
jobs in named tmux sessions with logged output, not bare background shells)
and fixed `percolation_calibration.py`'s default batch size to cover all
requested calibration permutations in one CUDA invocation, avoiding
repeated full-dataset reloads.

## Pilot results: the edge-fraction metric is broken

The pilot completed (17:00 total). Its curve revealed a real flaw in the
original metric choice, not just noise:

| p_CF | mean retained edges | p95 giant-**edge**-fraction | p95 giant-**voxel**-fraction |
|---:|---:|---:|---:|
| 0.001 | 1.17M | 0.889 | 0.909 |
| 0.0001 | 8.8k | 0.472 | 0.262 |
| 3e-5 | 21.7k | 0.394 | 0.077 |
| 1e-5 | 5.8k | 0.360 (local min) | 0.020 |
| 5e-6 | 2.4k | 0.477 | 0.013 |
| 2e-6 | 718 | 0.639 | 0.004 |
| 1e-6 | 273 | **1.000** | 0.002 |

Edge-fraction falls as expected down to a minimum around `p=3e-5..1e-5`,
then **spuriously climbs back to 1.0** at the strictest grid point: with
only ~273 edges surviving on average, a handful land in one component by
chance, and edge-fraction's denominator (retained edges) shrinks with its
numerator, so it can't distinguish "genuinely one giant component" from
"almost nothing left to count." Voxel-fraction — fixed denominator (59,677
mask voxels regardless of threshold) — falls monotonically with no
reversal, and directly matches the original giant-component symptom
("98.7% of voxels touched"). Claude switched `percolation_calibration.py`
to use voxel-fraction as the primary/gating metric, keeping edge-fraction
as a diagnostic-only column, and re-verified the fix against the synthetic
smoke test.

## Confirmatory 200-permutation run (LTLE/RTLE)

Launched via tmux (session `mocca_percolation_calib_ltle_200`, per the new
convention) at 200 null permutations. Results were consistent with the
pilot and much more stable:

| p_CF | p95 giant-voxel-fraction (N=200) | (N=30 pilot) |
|---:|---:|---:|
| 2e-5 | 5.6% | 5.4% |
| 1e-5 | 3.3% | 2.0% |
| 5e-6 | 1.6% | 1.3% |
| 2e-6 | 0.7% | 0.4% |

**Transition: `p_CF=1e-5`. Recommended operating threshold (one step
stricter): `p_CF=5e-6`.**

## Grid vs. single fixed threshold: an avoidable power tax

The user asked whether FWER also corrects for the number of grid points,
and whether a grid scan is even necessary once calibration has picked a
threshold. Claude confirmed: `run_bundle_fwer.py`'s
`symmetric_permutation_min_p` grid correction takes the minimum rank across
all tested thresholds per null permutation, which is a real multiplicity
penalty. Once a specific threshold is pre-registered from an independent,
null-only calibration (decided before the observed data is inspected), that
search is a degree of freedom already spent — re-scanning and correcting
for it again only costs power. This was later directly confirmed
numerically once real inference was run (see below): `p_grid_fwer` ran
~0.03–0.05 higher than `p_threshold_fwer` at matching rows.

## Permutation count versus statistical power

The user separately asked whether increasing permutation count (beyond
10,000) would help reach significance. Claude explained that permutation
count controls the *precision* of a p-value estimate, not the true p-value:
with an observed statistic sitting at p≈0.2 (~2,000 of 10,000 nulls
exceeding it), the estimate is already fairly precise and nowhere near the
resolution floor (`1/10,001≈0.0001`); more permutations would tighten the
estimate but not move it toward significance. The actual lever for power is
sample size, which computation cannot manufacture.

## Controls/patients calibration

The user asked to run the controls/patients calibration before any longer
job ("before running a longer job, run the calibration run for controls
vs patients"). Launched via tmux (`mocca_percolation_calib_cvp_200`, 68
subjects — 26 controls + 42 patients, ~1.84× more subject data to read than
LTLE/RTLE). Result: the **same** transition and recommendation as
LTLE/RTLE — `p_CF=1e-5` transition, `p_CF=5e-6` recommended (1.9% p95
voxel-fraction) — despite the very different subject counts, consistent
with the transition being governed more by the fixed spatial mask geometry
(voxel count, `neighbor_dist`) than by sample size, matching the prior
archive's own note that df-aware thresholds already account for sample
size separately.

## Production run 1: LTLE/RTLE sub-critical grid

The user asked to run the actual production grid-FWER runs, LTLE/RTLE
first. Launched via tmux (`mocca_ltle_grid_subcritical_10k`) with the grid
restricted to the four confirmed sub-critical points
`{1e-5, 5e-6, 2e-6, 1e-6}`, 10,000 null permutations, historical strict
bundler, same bundling parameters as before (`neighbor_dist=1,
min_size=10, min_cluster_voxels=6`). Claude also raised `--batch-size` to
1000 (from the script's resumability-tuned default of 8) given the newly
discovered ~263 GB reload-per-invocation cost.

**Result:** 1,174 sane, localized observed bundles (max 1,570 edges, vs.
millions before calibration) — the percolation fix worked as intended.
**No bundle reached significance** (best `p_grid_fwer≈0.197`, positive
direction, nested/growing consistently across thresholds). A legitimate
null result for this smaller, unbalanced (24 vs. 13 subject) comparison,
not a pipeline artifact.

## Production run 2: controls/patients, single threshold p_CF=5e-6

The user asked to run controls/patients next, using the single fixed
calibrated threshold per the earlier grid-vs-single discussion. Launched
via tmux (`mocca_cvp_fwer_p5e-6_10k`), `--cluster-forming-p 5e-6`, 10,000
null permutations, `--batch-size 2500` (given the larger ~483 GB dataset).

**Result:** 458 sane bundles (max 5,723 edges). **Two bundles significant
at α=0.05:**

| bundle | sign | edges | mass | p_fwer |
|---:|---:|---:|---:|---:|
| 94 | negative (patients > controls) | 5,723 | 1,865.9 | 0.042 |
| 95 | negative (patients > controls) | 5,026 | 1,736.4 | 0.045 |

Same direction as the original invalid giant-component finding from the
prior archive, now as two anatomically plausible, localized bundles.

## New visualization export tool: `prepare_bundle_single_fwer.py`

The existing `prepare_bundle_grid_fwer.py` was hardcoded for grid-mode
output (different CSV schema, `thresholds/<slug>/` layout, hardcoded
LTLE/RTLE labels). Claude wrote a companion script for single-threshold
results, `03_prepResultsForVisualization/prepare_bundle_single_fwer.py`,
with `--positive-label`/`--negative-label` **required** rather than
guessed. Before running it, Claude verified the sign convention from
`generatePermutations.py` (`original_grouping = tuple(range(nA))`; the
26-entry group A matches the 26-control count) — confirming **positive =
controls > patients, negative = patients > controls**, consistent with the
prior archive's independent record. Exported both a significant-only set
(the two real bundles) and a separately-labeled, explicitly non-significant
top-10 exploratory set for visual context.

## Production run 3: controls/patients, single threshold p_CF=1e-5

The user asked to also run `p_CF=1e-5` ("worth checking out") for more
power, since it was also confirmed sub-critical. Launched via tmux
(`mocca_cvp_fwer_p1e-5_10k`).

**Result:** 680 bundles (max 40,971 edges). **One bundle significant:**
bundle 129, 40,971 edges, mass 13,538.6, **p_fwer=0.016** — stronger than
either `5e-6` bundle, but far larger.

## The "suspecting something" moment: bundle 129 contains 94 and 95

Asked to prep this result for visualization too, the user said "I'm
suspecting something..." Claude checked bundle 129's voxel footprint first
(6,866 of 59,677 mask voxels, 11.5% of the brain — above the null p95 of
4.6% at this threshold, i.e. consistent with genuine signal, not a
percolation artifact) and flagged the open question of anatomical
coherence versus the historical bundler's known chaining tendency. Directly
testing the containment hypothesis: **bundle 129 is a strict superset of
bundles 94 and 95** — 100% of both smaller bundles' edges (10,749/10,749)
are contained in bundle 129, which adds 30,222 further edges and 4,384
further voxels on top. The user confirmed this matched their suspicion.

## Design discussion: COFFEE-DAC's bundle/network scheme breaks down

The user then raised a structural problem: COFFEE-DAC's original
bundle→network stratification existed purely to aid human visualization,
not as a statistical claim — but now, post-FWER-correction and
percolation-calibration, significant "bundles" are themselves already
network-scale (bundle 129's endpoints "make anatomical sense... more so
than any set of results before," so believed real), leaving nothing above
them to merge into. Hierarchical clustering has no native divisive
primitive; the user floated using the smallest available unit (individual
edges) and hierarchical clustering as the way to go the other direction,
and asked Claude's opinion.

Claude investigated the existing COFFEE-DAC clustering machinery (via a
dedicated Explore agent) before proposing anything, finding:

- `hc2`/`bundle_dist` (bundle→network): closest-endpoint distance between
  bundles, average linkage, `fcluster(..., nr_networks, 'maxclust')`.
- `hc1`/`h1_dist` (edge→bundle, the step that forms bundles in the first
  place): pairwise endpoint distance **between individual edges**,
  complete/average linkage, same `fcluster` cut mechanism, with an
  explicit `max_exact=50_000` parameter — already validated at the exact
  scale needed (the old LTLE/RTLE reference dataset had 28,767 raw edges).
- `recut_networks` + the GUI's spinbox + `dendrogram_plotter.show_dendrogram`
  already implement "build the tree once, cut it wherever, instantly" —
  generically, not tied to any specific meaning of "leaf."

Claude's recommendation: reuse `h1_dist` directly — exactly the same
edge-to-edge distance metric already used to form bundles — as the
divisive mechanism, treating one significant bundle's edges as leaves and
cutting that tree into sub-bundles, with the invariant that the displayed
p-value must remain the bundle's single whole-bundle FWER value throughout
(sub-division is a rendering aid only, never a new significance claim).

## Implementation: pipeline v3 (divisive sub-bundling)

The user asked for this to be built properly as a distinct "v3" pipeline,
reusing the GUI's existing recut spinbox (reinterpreted as "cut into N
bundles" instead of "cut into N networks") and the existing dendrogram
plotter. Claude implemented:

- **`04_coffee-dac/coffee_dac_pipeline_v3.py`**: `build_edge_linkage()`
  (h1_dist + `scipy.cluster.hierarchy.linkage` over individual edges),
  `recut_subbundles()` (fcluster + BUNDLE_COL rewrite, NETWORK_COL/pvalue/
  tstat never touched), `process_edge_data_v3()`, and
  save/load/cache-validation functions mirroring `coffee_dac_pipeline_v2.py`'s
  exact on-disk shape (`_v3_processed.csv`, `_v3_linkage.npy`,
  `_v3_params.json`) so existing loading code needs minimal changes.
- **`04_coffee-dac/run_pipeline_v3.py`**: CLI mirroring `run_pipeline_v2.py`'s
  conventions (`--bundles N`, `--recut N`, `--h1-flag`, `--method`,
  `--max-exact`).
- **GUI wiring** (`mocca_gui/data_loader.py`, `main_window.py`,
  `dendrogram_plotter.py`): v3 cache detection alongside v1/v2 in the load
  dialog; new combo entries ("Load existing v3 results — divisive",
  "Re-process with pipeline v3"); the recut spinbox now relabels itself to
  "Cut this bundle into N sub-bundles (v3 divisive)" when a v3 cache is
  active, same underlying instant-recut mechanism.

While implementing this, Claude found and fixed **two genuine bugs** in
existing code that reusing the dendrogram plotter as-is would have hit:

1. `prepare_dendrogram_plot_data()` assumed one dendrogram leaf per
   *bundle* (true for v2's bundle-level tree) but v3's tree has one leaf
   per *edge* — tens of thousands for a large significant bundle. Fixed by
   explicitly tracking which pipeline produced the loaded data
   (`self.dendrogram_leaves`, set from the loader's result rather than
   inferred from array shapes, which could be ambiguous), building
   per-edge labels for v3, and using scipy's `truncate_mode='lastp'` so the
   display collapses deep subtrees into a legible handful of nodes.
2. Under truncation, `dendro["leaves"]` can return **internal
   (collapsed-subtree) node ids**, not just original leaf indices — the
   pre-existing tick-label coloring code assumed otherwise and would have
   raised `IndexError` the first time anyone truncated any dendrogram (v2
   or v3). Fixed by reusing the same `link_color_func` logic already used
   for coloring links, for any node id beyond the original leaf range.

Validated end-to-end on real data: bundle 129 (40,971 edges) — full
edge-level linkage build ~23s, instant recut to a different sub-bundle
count ~0.6s, `pvalue`/`network` columns confirmed constant across all
sub-bundles, sub-bundle sizes and spatial bounding boxes looked
distinct/plausible. Confirmed both the new v3 (truncated) and legacy v2
(untruncated) dendrogram rendering paths work with no regression, headless
via `QT_QPA_PLATFORM=offscreen`/`MPLBACKEND=Agg`.

## LTLE/RTLE: is the null result just small sample size?

The user asked whether re-running LTLE/RTLE at `p_CF=1e-5` was worth doing,
suspecting group size (RTLE, n=13) was the limiting factor. Claude found
the answer was **already present** in the existing sub-critical grid
output (`p_threshold_fwer` at `cluster_forming_p=1e-5` is exactly what a
standalone single-threshold run would produce) — no new run needed. Best
result at `1e-5`: `p_threshold_fwer=0.207`, essentially unchanged from
`5e-6`'s `0.201`. This threshold-insensitivity, together with the much
smaller effect magnitude versus controls/patients (mass 721.6 at best vs.
1,865.9) despite LTLE/RTLE's smaller/more unbalanced sample (24 vs. 13,
total n=37), supports the small-sample-size/underpowered-comparison
hypothesis — while Claude noted this can't fully rule out a genuinely
smaller or absent true effect independent of power; the two are
indistinguishable without more RTLE subjects.

## Repository cleanup and pipeline consolidation (2026-08-28)

With the calibrated single-threshold pipeline now validated end-to-end on
both datasets, the user asked to consolidate: make this the documented
"main" pipeline, archive deprecated subroutines, and update READMEs
(including documenting the calibration method and its voxel-fraction
metric). Claude executed:

**Archived** to
`02_cudaPerm/archives/edgewise_fwer_and_supercritical_grid_2026_08_28/`
(own README + standalone `CMakeLists.txt`, verified to build independently
by referencing the still-active, shared `ccmat_io.c/.h`):

- `permutationTest_cuda_fwer.cu` (edgewise max-statistic FWER — superseded
  by bundle-level FWER, since correcting for ~1.78 billion simultaneous
  edge tests was hopelessly conservative) plus its now-orphaned
  dependencies `perm_kernels.cu/.cuh` and `results_io.c/.h` (confirmed by
  grep that nothing else references them functionally).
- `run_controls_vs_patients_subject_fwer_100k.sh`,
  `run_ltle_vs_rtle_subject_fwer_100k.sh` (edgewise launchers).
- `run_ltle_vs_rtle_bundle_grid_10k.sh` (the old 7-point grid mixing
  super-critical thresholds — the run that produced the original giant
  8.2-million-edge component).
- `pipeline/run_permutationTest.sh` — found via a repo-wide grep for
  dangling references to the archived binary; this top-level launcher
  hardcoded the now-removed build path and would have silently broken.
  Moved into the same archive and repointed at its own build output.

Removed the now-orphaned `permutationTest_cuda_fwer` target from the active
`02_cudaPerm/CMakeLists.txt`; rebuilt the active project clean and reran
`regression_bundle_fwer_cpp` and `regression_cuda_bundle` — all passed.

**READMEs rewritten:**

- `02_cudaPerm/README.md` — full main-pipeline documentation: why bundling
  now happens in this module rather than downstream in COFFEE-DAC, the
  percolation catch-22 stated plainly, the calibration method and its
  voxel-fraction order parameter documented including *why* the
  edge-fraction alternative was tried and rejected, single-fixed-threshold
  established as the documented main path with the multi-threshold grid
  explicitly demoted to an optional sensitivity tool (citing the measured
  ~0.03–0.05 power cost), the batch-size/disk-reload operational note, and
  the tmux launch convention.
- `03_prepResultsForVisualization/README.md` — `prepare_bundle_single_fwer.py`
  established as the main export path, `prepare_bundle_grid_fwer.py`
  demoted to secondary/optional, sign-convention correctness warning kept
  prominent.
- `04_coffee-dac/README.md` — added a "Choosing v2 vs v3" section (few/small
  bundles → v2 merges upward into networks; one big significant bundle →
  v3 divides it into sub-bundles) and full v3 usage documentation.

## Methodological commitments recorded in this continuation

- A cluster-forming threshold must be calibrated from an independent batch
  of null permutations, using an order parameter whose denominator does
  not degrade at small sample sizes (voxel-fraction, not edge-fraction),
  and decided entirely before the observed bundle statistics are inspected.
- Once a threshold is pre-registered this way, a multi-threshold grid
  search and its associated multiplicity correction becomes an avoidable
  power cost, not a validity requirement — single-fixed-threshold FWER is
  the default going forward; the grid remains available only for an
  explicitly declared secondary sensitivity analysis over already-vetted
  sub-critical thresholds.
- Sub-bundling a large, already-significant bundle for visualization
  (pipeline v3) must never alter or appear to alter its statistical
  claim — every sub-bundle keeps the single parent bundle's FWER p-value,
  unconditionally.
- Long-running or GPU-bound jobs are launched in named tmux sessions with
  logged output, not bare background shells, so they can be monitored
  independently of the assistant's own task tracking.
- Archiving means moving superseded-but-valid code into a documented,
  independently-buildable archive directory with a manifest explaining why
  it was superseded — never deletion.

## Current repository state and pending work

At the time of this archive:

- The active bundle-level FWER pipeline (`02_cudaPerm/`) uses
  percolation-calibrated single fixed cluster-forming thresholds by
  default; both datasets calibrated to `p_CF=5e-6` (recommended) /
  `p_CF=1e-5` (transition, more power, both confirmed sub-critical).
- LTLE/RTLE: no significant bundle at either threshold; a real null result
  attributed provisionally to the smaller, unbalanced (24 vs. 13) sample,
  not further investigated computationally.
- Controls/patients: two significant bundles at `p_CF=5e-6` (94, 95); one
  larger, stronger, strictly-containing significant bundle at `p_CF=1e-5`
  (129) — anatomically plausible per the user's own domain judgment, not
  yet resolved into a final reporting choice between the two thresholds'
  results, nor visually inspected for whether bundle 129's extra recruited
  territory (beyond 94∪95) looks like a coherent extension or a chaining
  artifact.
- Pipeline v3 (divisive sub-bundling) is implemented and validated on
  bundle 129 programmatically, but not yet visually inspected in the
  actual PyQt GUI (no display available in the working environment).
- The edgewise max-statistic FWER path and the original super-critical
  grid are archived but remain independently buildable/runnable for
  reproducibility.
- Repository READMEs (`02_cudaPerm/`, `03_prepResultsForVisualization/`,
  `04_coffee-dac/`) now document the current main pipeline end-to-end,
  including the calibration method.

Open items for a future session: decide how to present/reconcile the two
controls-vs-patients thresholds' results (5e-6's two bundles vs. 1e-5's one
larger bundle) in any final report; visually inspect bundle 129's v3
sub-division and its extra recruited territory in the actual GUI; decide
whether LTLE/RTLE's power limitation warrants a formal post-hoc sensitivity
estimate.

## Post-cleanup audit: hardcoded dataset assumptions in analysis code

The user asked, as a final check for the session, whether any routine was
still hard-coded for a specific dataset rather than general and
parameterized — the stated principle being that only wrapper/launch scripts
should contain literal data/mask/output paths, and analysis programs
themselves should take everything as CLI arguments.

Claude swept all active `.py`/`.cpp`/`.cu` files (excluding archives and
regression-test fixture lists, which legitimately reference specific known
fixtures by name) for hardcoded `LTLE`/`RTLE`/`controlsVSpatients`/`patients`
strings and hardcoded `/mnt/storage`/`/mnt/islay` paths. Found one genuine
issue predating this session's own new scripts:
**`03_prepResultsForVisualization/prepare_bundle_grid_fwer.py`** had
`"LTLEvsRTLE"` baked into its output filename prefix and, more seriously,
`"positive_effect": "LTLE > RTLE"` / `"negative_effect": "RTLE > LTLE"`
hardcoded directly into every exported result's provenance manifest —
meaning running it against any other dataset (e.g. controls/patients) would
silently write a false direction label into the manifest, exactly the kind
of mislabeling risk already designed around in
`prepare_bundle_single_fwer.py`.

Fixed by adding the same required `dataset_label` / `--positive-label` /
`--negative-label` parameters used in the single-threshold export script,
replacing all four hardcoded spots (output filename, manifest fields,
generated README heading and body text). Verified: missing labels now fail
fast with a clear argparse error; re-ran the export against the real
LTLE/RTLE sub-critical grid result with explicit labels and confirmed the
manifest and filenames come out correctly parameterized. Updated
`03_prepResultsForVisualization/README.md`'s example command to match.

No other hardcoded dataset-specific logic was found in active code; the
one other absolute-path hit (`04_coffee-dac/generate_gm_stl.py`) is inside
a docstring usage example, not executable logic.
