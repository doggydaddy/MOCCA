# MOCCA conversation archive — 2026-09-03

## Scope and fidelity

This file records the working conversation between the user and Claude on
2026-09-02 through 2026-09-03 concerning implementation of the Fisher-`z`
aggregation stage, the disjoint calibration/inference permutation partition,
the covariate-adjusted Freedman--Lane confirmatory model, and the production
run and diagnosis of that model's null result on the controls-versus-TLE
dataset.

The most recent exchanges are represented directly. Earlier parts of the
thread underwent automatic context compaction, so some assistant replies are
reconstructed from the retained technical record rather than claimed to be
word-for-word quotations. User questions, decisions, paths, configurations,
run timings, and implementation outcomes have been preserved as faithfully as
possible.

This archive intentionally excludes system/developer instructions and
contains only project-relevant discussion. Full technical detail (equations,
tables, reproducible commands) lives in
`manuscript/APPENDIX_COVARIATE_ADJUSTED_ANALYSIS.md`, produced alongside this
archive; this file is the narrative record of how the session unfolded.

## Starting point: implementing three deferred manuscript decisions

The session began with `manuscript/ANALYSIS_DECISIONS.md` already recording
three accepted-but-unimplemented decisions from a prior session: an optional
Fisher-`z` participant-aggregation stage (`01p5_FisherCC`), a disjoint
1,000-calibration/10,000-inference permutation partition, and a
covariate-adjusted Freedman--Lane confirmatory model. The user asked to begin
implementing them, starting with `01p5_FisherCC`.

## 01p5_FisherCC: Fisher-transform participant aggregation

Claude read the existing `02_cudaPerm/average_ccmat_runs.py` (the raw-`r`
equivalent) and the binary CCMAT container format, then implemented
`01p5_FisherCC/fisher_aggregate_ccmat.py` with three modes: `fisher-equal`
(the planned primary mode: `atanh` transform, equal-weight mean),
`fisher-duration` (explicit weight table required; never derives weights from
data), and `raw-equal` (reproduces `average_ccmat_runs.py` bitwise, for
validation). Runs are canonically ordered within participant before
accumulation, giving bitwise invariance to file-list order and chunk size.
Clipping applies only to `|r| >= 1`, counted and logged, never silently
coerced elsewhere. Because the downstream C reader (`ccmat_io.c`) accepts only
`CCMAT_MAGIC`, the Fisher-`z` identity is declared in the output filename, a
per-file JSON sidecar, and the run manifest rather than in the binary header.

27 regression tests were written and passed, including a bitwise real-data
check against an independent float64 reference over 300,000 randomly sampled
edges from participant `s109`, and a disk-space pre-flight guard.

## permutation_rows.py: the disjoint calibration/inference partition

A shared module (`permutation_rows.py`) was built to make the partition an
explicit, validated configuration rather than an implicit convention, with
production defaults of 1,000 calibration permutations (rows 1--1000) and
10,000 inference permutations (rows 1001--11000), giving
`p_FWER = (1 + exceedances) / 10001`. All three programs
(`generatePermutations.py`, `percolation_calibration.py`,
`run_bundle_fwer.py`) were wired to accept and record the same four
partition arguments. `--null-permutations` was removed from
`run_bundle_fwer.py` in favor of `--inference-permutations`, erroring with
guidance rather than silently running a different row set under a familiar
flag. `percolation_calibration.py` gained a predeclared stability check
(bootstrap resampling and four-way subdivision of the calibration rows only)
that warns explicitly if disjoint calibration blocks disagree, rather than
letting an unstable threshold pass silently.

34 regression tests were written over a synthetic permutation file with known
calibration rows, inference rows, exceedance counts, and denominators.

## Covariate-adjusted Freedman--Lane model and CUDA backend

The user asked Claude to implement the full covariate-adjusted model,
including the CUDA backend, "so there is a complete analysis pipeline."

Claude derived and numerically verified (before writing any code) that the
HC2-studentized group coefficient equals Welch's unequal-variance `t` exactly
for a two-group design with no covariates -- making HC2 not a compromise
choice but a strict generalization of the existing statistic. It then derived
the Freedman--Lane algebraic reduction: because the contrast vector is
orthogonal to the nuisance design and the full-model residual maker
annihilates the nuisance-fitted values, a permuted draw's statistic depends on
the data only through the nuisance residuals computed once per edge, turning
what would be one regression per (edge, permutation) pair into two matrix
products per edge-chunk. Both reductions were checked against direct,
literal per-draw regression to machine precision before being trusted.

`design_matrix.py` was built to encode the covariate audit as behavior:
handedness is refused as a primary covariate (raises unless
`--allow-confounded` is passed, since all six left-handed participants are in
the patient group), run count is opt-in only, and motion has no code path at
all -- its absence is recorded as a data-provenance limitation rather than
inferred. `freedman_lane.py` implements the reference algebra and emits both
the packed-GEMM tables and a compact binary "plan" file
(`freedman_lane_plan.flp`) for the CUDA backend, which depends only on the
design matrix, not on any outcome data.

The CUDA backend gained a `--freedman-lane PLAN` path in
`permutationTest_cuda_bundle.cu`. Two performance decisions were made and
measured, not assumed: loading each data part **subject-major** rather than
edge-major (so a warp reads one coalesced cache line instead of 32 separate
ones) cut the per-edge-permutation cost from 1.87 ns to 0.56 ns; a
theoretically faster single-pass form for the squared denominator was tested
and rejected because it raised the absolute error on `t` from 2.3e-6 to
1.2e-5 -- unacceptable given that accuracy near the cluster-forming threshold
decides bundle membership. `regression_cuda_freedman_lane.py` requires the
real GPU backend to select exactly the same suprathreshold edge set as the
Python oracle on every permutation of a synthetic fixture (agreement 7.3e-7 on
the statistic) and drives the full `run_bundle_fwer.py` pipeline end to end.

104 tests passed across all regression suites. The full pipeline (18 files,
+5885/-142 lines) was committed as `1ffa09d`.

## Production run: Fisher stage and adjusted calibration

The user confirmed the existing run-level `.ccmat` files did not need
regenerating and asked to run the full pipeline on the controls-versus-TLE
dataset, starting with the Fisher transform and 1,000-permutation calibration,
to confirm whether the existing `p_CF=5e-6` operating threshold remained
valid.

`fisher-equal` was run on all 278 run matrices (26 controls, 42 TLE),
producing 68 participant Fisher-`z` matrices (452 GiB) with only 4 clipped
values out of roughly 4.9e11 correlations. The predicted participant ordering
(derived independently from the module's own grouping logic before the run)
matched the actual output exactly. Group A was verified against the
diagnosis tags to be exactly the 26 controls.

Calibration at 1,000 permutations against the **adjusted** statistic found the
transition had moved one grid step stricter than the historical unadjusted
value: transition `5e-6` (recommended operating point `2e-6`), versus the
historical `1e-5` transition (recommended `5e-6`). A predeclared stability
check flagged this as unstable across bootstrap resamples (70.8% for `5e-6`,
29.2% for `1e-5`) and disjoint calibration blocks. Claude reported this
plainly rather than picking a threshold, noting that `2e-6` was valid under
either bootstrap reading.

## Threshold decision and the first inference run

**User:**

> run the inference for 5e-6

Claude recorded this as the analyst's explicit, dated decision in the decision
log before running anything, together with an honest accounting of what the
choice was and was not: it passed the predeclared sub-criticality test on its
own terms (0.0283, comfortably under epsilon 0.05), but carried no
one-grid-step safety margin, and matched the historical unadjusted analysis's
threshold for direct comparability.

The run took roughly 3.7 hours (5 CUDA batches of 2,500 permutations each,
dominated by repeated 452 GiB reads). Partway through, at 2,500 of 10,000
nulls, Claude flagged a preliminary concern: the null maximum-bundle-mass
distribution looked heavily contaminated by percolating permutations (about
1% with wildly inflated mass), and speculated this heavy tail was likely
responsible for the emerging null result, proposing a `2e-6` rerun to
distinguish artifact from biology.

**Final result:** no bundle survived at alpha 0.05 (469 observed bundles;
strongest at `p_FWER=0.090`; Clopper--Pearson precision check confirmed no
bundle's 95% interval straddled 0.05, so 10,000 permutations resolved the
call). Diagnosing the null tail more carefully at completion, Claude corrected
its own partway-through speculation: excluding all percolating nulls only
moved the top p-value from 0.090 to 0.078, and the observed maximum sat at the
91st percentile of even the clean part of the null distribution. The
contamination was real (and worth fixing in the calibration rule, since it
tests the 95th percentile while FWER inference depends on roughly the top
1%), but it was not what was driving this particular null result.

## The user's physiological objection, and the raw-equal detour

**User:**

> indeed this is concerning. The differences observed prior appears
> physiological (since differences were located in the temporal lobes
> primarily), and should not be effects of age and sex... I find it hard to
> believe that this dataset wouldn't show any patient vs controls difference

The user asked for a `raw-equal` aggregation run to isolate the Fisher-`z`
transform from covariate adjustment as the explanation. Claude launched it
(reusing the existing design matrix, Freedman--Lane plan, and permutation
file, since none depend on outcome data), but the user stopped it partway
through:

**User:**

> force stop the job for now. I sense it would be a waste of time and
> computation.

Claude stopped the tmux session and the background wait task, and on
confirmation deleted the 214 GiB of partial output.

## The threshold angle, and a genuine correction from the user

The user then proposed a different diagnosis: that the calibration threshold
choice itself might be too conservative, arbitrarily discarding real edge
candidates, and asked whether the grid could be refined to keep more edges
while still avoiding a giant percolating bundle.

Claude explained the mechanism it expected to dominate -- that loosening
`p_CF` inflates the *null* tail at least as fast as it inflates the observed
signal, because a null permutation has no real spatial structure to organize
suprathreshold edges around -- and, while investigating, found and presented
the existing (pre-session) unadjusted `p1e-5` result as apparent supporting
evidence: its top bundle covered 11.5% of the brain mask versus 2.5% at
`p5e-6`, which Claude characterized as likely percolation contamination.

**User:**

> nice catch, and I like how you are reasoning. my mistake, rather I was
> referring to the results in
> bundle_fwer_controlsVSpatients_10k_<p_cf threshold here>_dfaware_cpp, where
> p1e-5 one gave more result, and I preferred that one[;] the basic shape and
> location of the bundle(s) is the same, and coffee-dac visualization allowed
> sub-bundle... to be more clear. you have to trust me when I say 1e-6 was
> done, but gave nothing so was not kept.

Claude explicitly accepted this correction: voxel footprint alone is a weak
proxy for "diffuse noise" versus "a genuinely larger real network," especially
when shape, location, and sub-bundle decomposition are reported as consistent
across thresholds. This was logged as a correction, not walked back
defensively.

Claude then ran two cheap, real-data diagnostics rather than continuing to
debate in the abstract: a single-CUDA-pass sweep of the real (adjusted,
observed-row) bundle structure across nine thresholds from `1e-6` to `1e-5`
(showing the adjusted bundle *does* grow compactly and coherently, unlike the
unadjusted `p1e-5` case), and a cross-check of those observed masses against
the already-computed 1,000-permutation calibration null (an independent null
set from the 10,000 production nulls). The cross-check was decisive: observed
mass roughly doubled from `5e-6` to `1e-5`, but the null's mass scaled up by
almost exactly the same factor, leaving the rough p-value unchanged at 0.083
for both. This is the opposite of the pattern in the unadjusted analysis
(`p_FWER` improved from 0.042 to 0.016 across the same two thresholds), and
was the first clean evidence that the null result was not primarily a
threshold artifact, within the range tested.

An orphaned second CUDA process (left over from a throwaway diagnostic row
that was no longer needed once the first pass's per-threshold files were
found to already contain everything required) was identified and killed
directly after `tmux kill-session` failed to stop it, freeing the GPU.

## The isolating experiment

**User:**

> this cross-check was really insightful... and while before I was leaning
> towards the covariate correction remove significance, with this evidence
> I'm leaning heavier on that path now... run it then.

Claude ran the cheapest remaining experiment to separate the two remaining
confounded variables: the already-computed Fisher-`z` participant matrices,
run through the **unadjusted** Welch statistic (no covariates), independently
calibrated (finding the transition at `1e-5`, essentially identical to the
historical raw-`r` calibration, confirming the Fisher-`z` transform alone does
not shift the null geometry), then run to full 10,000-permutation inference at
`p_CF=5e-6`.

**Result:** two bundles significant at alpha 0.05 (`p_FWER=0.041` and
`0.041`), closely matching the historical raw-`r` result (`0.042` and `0.045`)
in magnitude, footprint, and direction. This isolated the cause: the
Fisher-`z` transform reproduces the original finding; only the addition of
covariate adjustment removes it.

Along the way, Claude noticed and corrected an efficiency issue in the
adjusted inference run's batching (5 separate 452 GiB reads where 2 would
have sufficed) and applied the fix (`--batch-size 10000`) to this run, though
the uncoalesced legacy Welch kernel's slower per-permutation cost largely
offset the I/O savings in practice (total CUDA time was comparable to the
adjusted run despite reading the dataset three fewer times).

## Where the analysis stands

The user summarized their own reading of the accumulated evidence:

> I'm pretty convince[d] already that the changes we made made the analysis
> pipeline better and more robust, unfortunately for the dataset at hand,
> robustness means no significance. I'm in the thinking path of I have to
> present fisher-z adjusted, but not covariate-corrected results (just to be
> able to show SOMETHING), and explain why covariate-correction removes said
> results... I'm mentally preparing myself for that outcome...

All decisions, thresholds, calibration results, and both isolating diagnoses
are recorded contemporaneously in `manuscript/ANALYSIS_DECISIONS.md`. A
self-contained methods-and-results appendix,
`manuscript/APPENDIX_COVARIATE_ADJUSTED_ANALYSIS.md`, was produced at the
user's request to summarize the full chain of evidence with reproducible
detail, separate from this narrative archive.
