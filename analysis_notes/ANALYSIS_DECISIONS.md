# MOCCA analysis decision log

This file records methodological decisions that should survive beyond the
current working session. A decision marked **planned** must not be described in
the manuscript as completed until the corresponding analysis and validation
have been run.

## 2026-09-04: which covariate removes the effect -- post-hoc decomposition

**Status:** complete. The descriptive decomposition attributed the loss of
significance to age; the pre-declared confirmatory run then **refuted that
attribution**. Both are recorded below, in that order, because the refutation
carries a methodological lesson that the decomposition alone would have hidden.
**Bottom line: age alone does not remove the effect; sex is required.**

### A correction to the record

The appendix's 2x2 design is sometimes recalled as having tested raw-`r` with
covariate adjustment. It did not: **only three of the four cells were ever
run** (raw `r` unadjusted; Fisher `z` unadjusted; Fisher `z` adjusted). The
conclusion that adjustment -- not outcome scale -- removes significance rests
on *elimination*: rows 1 and 2 are near-identical, so the scale axis is inert,
and the loss appears only on the statistic axis. That reasoning is sound but it
is an inference from three cells, not a direct measurement of the fourth, and
the report should say so. The missing cell would only matter if scale and
adjustment interacted, which rows 1-2 make implausible.

### Question

The adjusted model carries age *and* sex together, so the appendix could not
say which is responsible, nor whether the responsible covariate *interacts*
with group at the locations of the previously significant bundles.

### Method

`04p5_covariate_analysis/covariate_decomposition.py` refits nested models on a fixed edge
set and compares the HC2 group statistic across them:

- **Target set:** the 9,674 edges of bundles 66 and 67 from the unadjusted
  Fisher-`z` run (`p_FWER = 0.041` both).
- **Background set:** 10,000 suprathreshold edges sampled from the *other*
  bundles of the same run -- comparable signal-to-noise, not selected for an
  extreme group effect.
- **Models:** `group`, `group + age`, `group + sex`, `group + age + sex`, built
  by the validated `design_matrix.build_design` so coding and centering match
  the production design exactly.
- **Extraction:** ccmat is a flat float32 array behind a 24-byte header, so the
  ~20k wanted edges are seeked to directly rather than scanning 68 x 7 GB.
  (First run incurred ~287 GB of device reads from kernel readahead on 4-byte
  seeks; `posix_fadvise(POSIX_FADV_RANDOM)` + `pread` has since been added.)
- **Validation:** the vectorized HC2 was checked against
  `freedman_lane.statistic_direct` over 30 random designs and reproduces
  Welch's `t` exactly in the no-covariate case.

### Descriptive result: age, not sex (subsequently refuted -- see below)

Bundle mass at `p_CF = 5e-6`, edge set held fixed, design varied:

| model | target mass | drop | background mass | drop |
|---|---:|---:|---:|---:|
| group only | 3494.5 | -- | 3185.2 | -- |
| **group + age** | 1436.5 | **-58.9%** | 2747.6 | -13.7% |
| group + sex | 3083.8 | -11.8% | 2269.5 | -28.7% |
| group + age + sex | 821.9 | -76.5% | 1666.6 | -47.7% |

Age alone removes 59% of the target bundles' mass; sex alone removes 12%, and
adds comparatively little on top of age (58.9% -> 76.5%).

**The background set rules out the obvious artifact.** These bundles were
selected for an extreme group contrast, so they have the most to lose under
*any* model perturbation -- regression to the mean predicts the target shrinks
more than background for **both** covariates. It does not. Age hits the target
**4.29x** harder than background; sex hits it **0.41x**, i.e. *less* than
background, in the opposite direction. Selection shrinkage cannot produce that
asymmetry, so the age effect is specific to these bundles rather than an
artifact of having selected them.

### Result: no interaction

| interaction | edge set | mean abs t | frac. p<0.05 | frac. p<0.001 |
|---|---|---:|---:|---:|
| group x age | target | 0.856 | 4.35% | 0.14% |
| group x age | background | 0.855 | 5.69% | 0.19% |
| group x sex | target | 0.806 | 5.54% | 0.23% |
| group x sex | background | 0.945 | 7.97% | 0.13% |

The target bundles' interaction statistics are indistinguishable from
background and sit **at or below the 5% expected by chance**; the target is if
anything quieter than background. There is no evidence that age moderates the
group effect where the effect was found.

### Interpretation of the decomposition (superseded)

Age **absorbs** the group effect without **moderating** it. That is the
signature of a confounder rather than an effect modifier, and it matches the
sample: controls 32.2 +/- 8.9 years versus TLE 37.1 +/- 10.8. The absence of an
interaction actively supports the confounding reading over a biological one.

This does **not** establish that age has a biological effect on these edges.
Age is the covariate more strongly correlated with group in this sample, so
adjusting for it removes group-associated variance whether or not age acts on
this connectivity. At n=68 those two readings cannot be separated.

### Limits

- **Descriptive, not inferential.** The bundles were selected on the group
  effect, so every quantity computed inside them is conditioned on that
  selection. The masses above are not p-values, and the interaction statistics
  are tested in a region chosen for its main effect -- circular by
  construction. Only the *relative* comparison (fixed edge set, varying design;
  target versus background) is being relied on.
- Sex's *larger* effect in the background set is unexplained and not pursued;
  it is reported for completeness rather than interpreted.

### Pre-declared confirmatory run (NOT YET RUN)

To convert "age is the culprit" from description into inference, a whole-brain
adjusted run with **age only** (`design_matrix.py --no-sex`) will be run at the
production specification: Fisher `z`, HC2 Freedman-Lane, `p_CF = 5e-6`, mass
statistic, max-statistic FWER, the same 1,000/10,000 disjoint partition. Its
threshold is **not** recalibrated; `p_CF = 5e-6` is carried over from the
existing adjusted calibration for direct comparability, and this is disclosed
as a deliberate continuity choice exactly as it was for the primary adjusted
result.

Declared in advance, so the reading is not chosen after the fact:

- If the age-only model yields **no significant bundle** (comparable to the
  age+sex `p_FWER = 0.090`), age alone accounts for the loss and the
  decomposition is confirmed.
- If it yields **two significant bundles** (comparable to the unadjusted
  0.041), age alone does *not* account for the loss, the decomposition is
  contradicted, and the joint adjustment's effect is not attributable to age
  in isolation.
- An **intermediate** result (significance lost but `p_FWER` materially below
  0.090) means age contributes most but not all of the loss; it would be
  reported as such rather than rounded to either extreme.

A sex-only run is *not* pre-declared: the decomposition gives no reason to
expect it to be informative, and running it only if the age-only result
disappoints would be a search.

### Confirmatory result: outcome 2 -- the decomposition is contradicted

Run at the pre-declared specification (design intercept + centered age +
group, rank 3/3, condition 23.5, residual df 65; all else at production
settings). **521 observed bundles, two significant at `alpha = 0.05`:**

| bundle | sign | edges | mass | p_FWER | 95% CI | CI crosses alpha |
|---:|---:|---:|---:|---:|---|---|
| 52 | TLE > controls | 8,258 | 2919.7 | **0.0349** | 0.0314-0.0387 | no |
| 51 | TLE > controls | 8,470 | 2359.1 | **0.0419** | 0.0381-0.0460 | no |
| 53 | TLE > controls | 6,792 | 1929.3 | 0.0506 | 0.0464-0.0551 | yes |

Omnibus p-values 0.0160 / 0.0158 / 0.0141, consistent with every other run.

This is **outcome 2 of the three declared in advance**: age-only adjustment
leaves the effect essentially intact (0.035, 0.042), closely matching the
unadjusted Fisher-`z` result (0.041, 0.041). **Age alone does not account for
the loss of significance. Sex is required -- alone or in combination.**

Note bundle 53 at `p_FWER = 0.0506` with a CI straddling alpha: a third bundle
sits on the significance boundary and is unresolved at 10,000 permutations.
It should be reported as such rather than as a clean negative.

### Why the decomposition misled -- the methodological lesson

This is not a coding error and the decomposition's arithmetic stands. The
decomposition compared the **observed** statistic across models. Significance
compares the observed statistic against **that model's own null**, and each
model has a different one. A covariate that shrinks the observed bundle mass
and its permutation null by comparable factors leaves `p_FWER` unchanged --
which is what age does, despite removing 59% of the observed mass. Sex removes
only 12% of the observed mass yet evidently moves the observed statistic
relative to its null.

**An observed-only decomposition cannot rank covariates by their effect on
significance: it measures numerators when inference is decided by a ratio.**
Only running each candidate model's own permutation null answers the question.
Diagnostic III should be read strictly as a statement about where variance
sits, never about what drives the result.

This is also the clearest possible vindication of pre-declaring the
confirmatory run's three readings: the descriptive result was clean,
internally consistent, had a plausible mechanism (age confounded with group at
32.2 vs 37.1 years), survived a regression-to-the-mean control, and was
**wrong**. Without the pre-declared confirmatory run it would have gone into
the manuscript as a finding.

### Sex-only run: declared as the fourth member of a sensitivity series

"Sex alone" versus "sex in combination with age" is not distinguished by
anything above. A sex-only run separates them, and is **declared here before
being run** as the fourth and final member of a complete sensitivity series
reported in full regardless of outcome:

| model | status |
|---|---|
| unadjusted (Fisher `z`) | run: 2 significant, 0.041 / 0.041 |
| group + age | run: 2 significant, 0.035 / 0.042 |
| **group + sex** | **declared, pending** |
| group + age + sex | run: 0 significant, min 0.090 |

Specification identical to the age-only run (`design_matrix.py --no-age`;
Fisher `z`, HC2 Freedman--Lane, mass, `p_CF = 5e-6` carried over without
recalibration, same 1,000/10,000 disjoint partition, max-statistic FWER).

Readings fixed in advance:

- **No significant bundle** (comparable to 0.090): sex alone accounts for the
  loss, and the series resolves cleanly to a single covariate.
- **Two significant bundles** (comparable to 0.041): *neither* covariate alone
  accounts for the loss, which must then arise from their combination -- the
  most interesting outcome, and one that would need explaining rather than
  reporting. Candidate mechanisms to examine in that case, none assumed:
  joint collinearity with group beyond either margin, the extra loss of a
  residual degree of freedom (65 -> 64), or the two covariates jointly
  spanning a direction in participant space that group also occupies.
- **Intermediate**: reported as such.

The justification for running this is that the age-only result falsified the
premise on which it was previously declined (that the decomposition gave no
reason to expect it informative). It is not being run because a previous
result disappointed, and the series is reported whole either way.

### Sex-only result, and the completed series

**One bundle significant** (bundle 107, 7,705 edges, mass 2585.3,
`p_FWER = 0.0382`, 95% CI 0.0345-0.0421, clear of alpha); the second bundle
sits at 0.0546 (CI 0.0502-0.0592, also clear of alpha, on the other side).
This is the declared **intermediate** outcome. Omnibus 0.025 / 0.024 / 0.021.

The complete sensitivity series, all four members at the identical
specification (Fisher `z`, HC2 Freedman--Lane, mass, `p_CF = 5e-6`,
max-statistic FWER, same partition):

| model | sig. | top p_FWER | suprathresh. edges | retained | top bundle edges | top mass | null q95 | obs / null q95 | coherence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unadjusted | 2 | 0.0407 | 77,552 | 59,536 | 5,013 | 1583 | 1260 | 1.26 | 0.084 |
| group + age | 2 | 0.0349 | 109,509 | 88,365 | 8,258 | 2920 | 1949 | 1.50 | 0.093 |
| group + sex | 1 | 0.0382 | 81,037 | 63,573 | 7,705 | 2585 | 1942 | 1.33 | 0.121 |
| group + age + sex | 0 | 0.0901 | 81,169 | 59,911 | 4,743 | 1212 | 2484 | 0.49 | 0.079 |

(*coherence* = the top bundle's share of all retained suprathreshold edges.)

**Neither covariate alone removes the effect; only their combination does.**

### The pre-declared mechanism is refuted

Collinearity was the leading candidate and it does not survive contact with
the numbers. Regressing `group` on the covariates:

| covariates | R^2 of group | VIF for group |
|---|---:|---:|
| age | 0.0538 | 1.057 |
| sex | 0.0447 | 1.047 |
| age + sex | 0.1077 | 1.121 |

Super-additivity is `+0.0092` in R^2 -- negligible. The residual SD of the
group contrast falls only from 0.4860 to 0.4590, a ~6% inflation of its
standard error, which cannot turn `p = 0.038` into `p = 0.090`. The
degrees-of-freedom loss (65 -> 64) is smaller still. Both pre-declared
mechanisms are therefore **rejected**.

### What the data do show (mechanism not established)

Two things move, and only in the joint model do they move *against each
other*:

- **Each covariate alone increases the observed signal.** Top bundle mass goes
  1583 (unadjusted) to 2920 (age) and 2585 (sex). This is ordinary precision
  gain: removing nuisance outcome variance shrinks residuals, raises `t`, and
  more edges clear the threshold. The permutation null rises in step
  (1260 -> 1949, 1942), so `p_FWER` barely moves.
- **Jointly, the observed signal collapses while the null keeps rising.** Top
  mass falls to 1212 -- below even the unadjusted value -- while null q95 rises
  to 2484. The observed-to-null ratio goes 1.26 -> 1.50 / 1.33 -> **0.49**.

The joint model retains as many suprathreshold edges as the sex-only model
(81,169 vs 81,037) but its largest bundle is nearly half the size (4,743 vs
7,705), and it has the **lowest spatial coherence of all four models** (0.079).
Joint adjustment does not remove suprathreshold edges so much as remove their
spatial contiguity, so they no longer assemble into a large bundle.

This is a description, not an explanation. Why age and sex jointly absorb a
spatially structured component that neither absorbs alone is **not
established** here, and should be reported as an open question rather than
given a mechanism it has not earned.

### Reporting consequence

The full four-model series is the honest object for the manuscript, not any
single anointed specification. It shows the adjustment decision is
load-bearing and non-obvious: each covariate alone is harmless or mildly
*beneficial* to sensitivity, while the conventional "adjust for age and sex"
model is the only one of the four that finds nothing. Presenting the age+sex
model alone as *the* analysis would hide that entirely.

## 2026-09-04: TFCE and bundle-level FDR -- built, calibrated, and run (null)

**Status:** run and complete. Two alternative correction schemes were
implemented, validated, and taken to production on the covariate-adjusted
controls-vs-TLE analysis. **Neither localizes an effect.** The entry also
records one decision settled by measurement (BH over BY), one calibration
criterion found to be unusable, and one hypothesis of the analyst's that the
data refuted.

### Motivation

The completed covariate-adjusted analysis (2026-09-03 entries below) found no
bundle significant at `alpha = 0.05` under mass + max-statistic FWER
(`p_FWER = 0.090`). Two questions followed: whether that null was an artifact
of the *correction scheme* rather than of the effect, and whether the
cluster-forming threshold could be removed from the inference altogether.
Both were pursued as pre-declared methodological alternatives applied to the
same frozen data, not as a search for a specification that produces
significance.

An additional observation, recorded here because it motivated everything
below: the same permutation null already answers a *global* question, and
answers it affirmatively. Using the existing 10,000-row adjusted inference
null, the observed row's total suprathreshold edge count, retained edge count
and bundle count give omnibus permutation p-values of **0.029, 0.031 and
0.022**. The adjusted analysis is therefore not null at the global level; it
is null at the level of *localization*. These are secondary, post hoc, and
near-collinear (one test, not three), and must be declared as such.

### What was built

- `02_cudaPerm/false_discovery.py` -- pooled-null uncorrected p-values and
  Benjamini-Hochberg / Benjamini-Yekutieli step-up adjustment. Validated
  against `scipy.stats.false_discovery_control` over random and tied input.
- `02_cudaPerm/bundle_fwer_omp.cpp` -- two opt-in additions, both default-off
  with existing output byte-identical: `--bundle-statistics FILE` (every
  bundle of every permutation, since FDR needs the individual statistics and
  not just the per-permutation maximum), and `tfce` as a third bundle
  statistic.
- `02_cudaPerm/run_bundle_fwer.py` -- mutually exclusive `--fwer` (default)
  and `--fdr`; `--statistic tfce` with `--tfce-*` options.
- `02_cudaPerm/tfce.py` -- Python reference implementation (oracle).
- `02_cudaPerm/tfce_calibration.py` -- null-only (E, H) calibration.
- `02_cudaPerm/fdr_null_calibration.py` -- complete-null false-positive check.
- Tests: `regression_bundle_fdr.py` (34) and `regression_bundle_tfce.py` (16).
  `regression_bundle_fwer.py` (4) was additionally repaired -- it had been
  erroring on all four tests since the 2026-08-28 archive cleanup moved its
  fixtures, unnoticed because it was not in the counted suite.

### TFCE design decisions

Elements are **edges**, and the adjacency integrated over is the strict
bundler's own, so TFCE enhances the bundle geometry already in use rather
than introducing a second notion of a cluster.

- **Height is the z-equivalent of the edge's two-sided p, not `|t|`.** Welch's
  Satterthwaite df varies per edge, so raw `|t|` is not comparable across the
  map. Thresholding at height z is thresholding each edge at its own critical
  t, which reuses the existing df-aware machinery unchanged.
- **Extent is distinct voxels touched, not edge count.** A densely connected
  region of V voxels carries ~V^2/2 edges, so an edge-count extent acts like a
  doubled exponent and restores the giant-component domination TFCE exists to
  damp. This also matches the percolation calibration's finding that
  voxel-fraction behaves where edge-fraction does not.
- **No pruning during integration.** `min_size`, `min_cluster_voxels` and the
  isolated-edge filters were always legibility filters rather than statistical
  ones, so they are applied only to the bundles finally reported. The
  permutation maximum is consequently the *unpruned* one, which is
  conservative by construction and asserted directly in the regression suite.

### Decision: BH is the primary FDR adjustment, on measured evidence

The analyst asked whether Benjamini-Yekutieli should be preferred over
Benjamini-Hochberg, given that BH requires positive regression dependency
(PRDS) and MOCCA's bundles are a data-dependent partition. An initial argument
for BY -- that percolation makes "one giant bundle" and "many moderate
bundles" competing outcomes, hence negatively dependent -- **was wrong and is
retracted here.** That competition only holds conditional on the total
suprathreshold mass being fixed, and the total is not fixed: across the
adjusted null it ranges from a median of ~1,700 retained edges to a maximum of
989,728. That unconditional variation dominates, and it induces *positive*
dependence. Separately, BY does not address the harder half of the problem
either: its proof also assumes a fixed family of hypotheses, not a random
number of randomly-shaped ones.

The question was therefore settled by measurement rather than argument, using
an identity that holds exactly under the complete null: every rejection is
false, so FDP is 1 whenever anything is rejected and

```text
FDR = E[FDP] = P(at least one rejection).
```

A procedure controlling FDR at `q` must declare at least one bundle in at most
a fraction `q` of pure-null permutations. `fdr_null_calibration.py` measures
this on the run's own permutations, each null permutation ranked against a
pooled null built from every *other* permutation (exact leave-one-out, so a
permutation never contributes to the distribution judging it):

| specification | q | BH any-rejection rate (95% CI) | BY any-rejection rate (95% CI) |
|---|---:|---:|---:|
| TFCE E=0.25, H=1 | 0.05 | **0.0265** (0.0235-0.0298) | 0.0022 (0.0015-0.0033) |
| TFCE E=0.25, H=1 | 0.10 | **0.0663** (0.0616-0.0714) | 0.0057 (0.0044-0.0074) |
| TFCE E=0.5, H=2 | 0.05 | **0.0250** (0.0221-0.0283) | 0.0024 (0.0016-0.0036) |

BH controls at roughly half its nominal budget; BY runs about twenty times
tighter than the budget and forfeits power for no measured benefit. **BH is
the primary reported adjustment; BY is retained as a conservative sensitivity
column and is always written.** Two limits: this tests control under the
complete null only (partial-null FDR is not directly checkable this way), and
it inherits the bundling and threshold settings of the run producing its
input.

For scale, on the historical mass-based adjusted run the pooled null held
702,794 bundles against 469 observed, giving a BY penalty `c(m) = 6.73`; on
the TFCE run, 5,244,081 null bundles against 1,801 observed, `c(m) = 8.07`.

### The null-only (E, H) criterion is unusable, and was counterproductive

`tfce_calibration.py` was run on the 1,000 held-out calibration rows over
E in {0.25, 0.5, 0.75, 1.0} and H in {1, 2, 3}, gating on the scale-free tail
ratio `q99 / median` of the null max-TFCE distribution:

| E \ H | H=1 | H=2 | H=3 |
|---|---:|---:|---:|
| 0.25 | **2.71** | 2.77 | 2.91 |
| 0.50 | 6.30 | 6.30 | 6.27 |
| 0.75 | 15.48 | 15.63 | 15.67 |
| 1.00 | 36.65 | 36.92 | 37.60 |

Two structural findings:

- **H is almost irrelevant** to the criterion (rows are flat to a few
  percent): H reweights heights, but the same bundle geometry recurs at every
  height, so numerator and denominator rescale together.
- **The criterion is monotone decreasing in E with no interior optimum.**
  Extent is exactly what giant components have, so suppressing extent always
  reduces tail heaviness. Extrapolated to E=0, TFCE degenerates into an
  integral of height alone -- a purely edgewise statistic with no spatial
  enhancement, i.e. the archived edgewise pipeline. The recommendation of
  E=0.25 is therefore the *boundary of the offered grid*, not a calibrated
  optimum; a grid including 0.1 would have returned 0.1.

This is structural, not a bug: a null-only criterion cannot select a
statistic's shape, because nulls carry no information about sensitivity. The
percolation calibration worked because it was a *constraint* with a physical
meaning (voxel fraction <= epsilon), not an unbounded objective.

Worse, the choice was counterproductive in practice: the calibrated
E=0.25, H=1 gave a *worse* max-statistic FWER p-value (0.100) than the
Smith & Nichols default E=0.5, H=2 (0.070) on the same data. Note also that
the FDR and FWER columns disagree about which (E, H) is preferable, because
FDR ranks against the pooled distribution of individual bundle statistics
while FWER ranks against the maxima; "the best (E, H)" is not even well
defined until the error rate is fixed.

**Decision:** the tail-ratio criterion is recorded as tried and rejected. If a
TFCE result is reported, `E=0.5, H=2` (Smith & Nichols) is the pre-declared
specification, with the table above published as the honest statement that
this pipeline's TFCE tail is extent-driven.

### Results (covariate-adjusted, Fisher-z, HC2 Freedman-Lane, 10,000 held-out rows)

| specification | z_min | observed bundles | best `p_uncorrected` | best `p_fdr_bh` | BH sig. | max-stat FWER p |
|---|---:|---:|---:|---:|---:|---:|
| mass, p_CF=5e-6 (historical) | -- | 469 | -- | -- | -- | 0.090 |
| TFCE E=0.25, H=1 | 4.0 | 1,801 | 2.03e-4 | 0.180 | **0** | 0.100 |
| TFCE E=0.5, H=2 | 4.0 | 1,801 | 1.37e-4 | 0.247 | **0** | **0.070** |
| TFCE E=0.5, H=2 | 4.6 | 420 | 3.54e-3 | 0.663 | **0** | 0.108 |

**No bundle is significant at q = 0.05 under BH or BY at any specification
tried.** TFCE at the Smith & Nichols exponents reaches `p_FWER = 0.070`,
nominally better than mass's 0.090 -- but see the correction below: that
statistic belongs to a percolating giant bundle, not to a localized one, and
the apparent improvement is therefore not a localization result.

The omnibus statistics recomputed on the TFCE run reproduce the earlier
values closely: **0.0289** (suprathreshold edges), **0.0289** (retained),
**0.0231** (bundle count), against 0.029 / 0.031 / 0.022 from the mass run.
Two entirely different statistics and thresholds, the same global answer.

### A hypothesis that was refuted

The calibrated sub-critical threshold `p_CF = 5e-6` corresponds to `z = 4.565`.
The production TFCE run integrated from `z_min = 4.0` (`p = 6.33e-5`), an order
of magnitude more liberal, and at that floor the observed data percolates:
823,768 suprathreshold edges, of which the largest single bundle holds
**464,036 -- 56% of all retained edges** (null median 66,166 edges, 390
bundles). The analyst predicted that this super-critical floor was suppressing
the result, and that re-integrating from `z_min = 4.6` -- every height
sub-critical -- would be the informative specification.

**It was the worst of the three** (`p_FWER = 0.108`, best `p_fdr_bh = 0.663`).
Restricting to 420 bundles over 25 heights discarded more real signal than it
removed giant-component contamination; TFCE's `h^H` weighting was already
damping the giant component enough that the lower heights were a net gain.
Recorded because the prediction was explicit, is a natural one for a reader to
form, and is wrong.

A corollary for the write-up: **TFCE is not literally threshold-free at this
scale.** Integrating from z=2 or 3, as one would on a voxel map, is not
tractable at 1.78e9 edges (z=3.0 implies ~4.8M null edges per permutation).
The sharp cluster-forming threshold is replaced by a soft floor, which is a
milder dependence but still a choice.

### Interpretation

Across two bundle statistics (mass, TFCE), two error rates (FWER, FDR), four
thresholds and three exponent settings, **nothing localizes**, while the
global effect is stable at p ~ 0.025 across independent specifications. The
supportable conclusion is that the covariate-adjusted group effect in this
sample is real at the global level but not attributable to any individual
bundle at this sample size -- no longer a suspicion but a result, and one that
is honestly reportable as such.

### Scope, and what was deliberately not done

- The search was **stopped deliberately**. The trend across `z_min` suggests
  more liberal floors keep improving the p-value, and further tuning could
  plausibly reach 0.05. That would be selecting a specification on its
  p-value. Three specifications is already at the limit of what is defensible
  to report, and any further variant must be pre-declared for a reason other
  than its outcome.
- The omnibus statistics are post hoc and near-collinear; they must be
  declared as a single secondary test and, if reported, computed identically
  for the unadjusted arm.
- Nothing here revises any completed run below. The historical mass-based
  results stand as reported.
- The unadjusted Fisher-z arm was subsequently run under the identical
  specification; see the section below, which supersedes the tentative
  reading above.

### Symmetry check: the unadjusted arm, and two structural findings

The unadjusted Welch/Fisher-z arm was run under the **identical**
specification (E=0.5, H=2, z_min=4.0, storage p=1e-4, BH at q=0.05, same
partition), differing only in the edge statistic and the corresponding
group-a permutation file. This arm is the control: it has a known effect,
two bundles at `p_FWER = 0.041 / 0.045` under mass + FWER at the calibrated
sub-critical `p_CF = 5e-6`.

| | adjusted | unadjusted |
|---|---:|---:|
| observed bundles | 1,801 | 1,437 |
| pooled null bundles | 5,244,081 | 3,725,637 |
| best `p_uncorrected` | 1.37e-4 | 5.3e-5 |
| best `p_fdr_bh` | 0.247 | 0.076 |
| **BH significant at q=0.05** | **0** | **0** |
| TFCE max-statistic FWER p | 0.070 | 0.0198 |
| omnibus (edges / retained / bundles) | .029/.029/.023 | .017/.017/.025 |

**TFCE + FDR fails to recover a known effect.** The unadjusted arm's effect is
real and was detected by the existing pipeline, yet TFCE + FDR declares
nothing. This is the decisive result: it separates "the adjusted effect is too
small" from "the new correction path is insensitive", and the answer is the
latter.

**Finding 1: bundle-level FDR is structurally *less* powerful than
max-statistic FWER in this pipeline.** For the top bundle with `k`
exceedances, `p_FWER ~= k/(B+1)` while `p_BH ~= m * k/(N+1)`, so

```text
p_BH / p_FWER  ~=  m * (B+1)/(N+1)  =  m / (mean bundles per null permutation)
```

Measured: **3.86x** (unadjusted, m=1,437 vs mean 372.6) and **3.43x**
(adjusted, m=1,801 vs mean 524.4). FDR's larger denominator (millions of
pooled bundles rather than thousands of maxima) is more than cancelled by its
multiplicity penalty. Worse, the cancellation is *adversarial*: the observed
row has more bundles than a null permutation precisely when a global effect is
present, so the very signature of an effect inflates `m` and erodes the FDR
advantage. The expectation that FDR would be the more permissive option --
the premise on which this whole line of work was opened -- is **wrong for this
pipeline**, and quantifiably so.

**Finding 2: TFCE's apparent gain is the giant component, not a
localization.** The top-ranked TFCE bundle in each arm is the percolating
component:

| arm | top bundle | edges | distinct voxels | % of mask | p_FWER |
|---|---:|---:|---:|---:|---:|
| adjusted | 285 | 464,036 | 34,536 | **57.9%** | 0.070 |
| unadjusted | 301 | 487,138 | 36,775 | **61.6%** | 0.0198 |

The single `p_FWER <= 0.05` result anywhere in this work is therefore the
statement "62% of the brain differs" -- statistically valid, anatomically
uninformative, and exactly the outcome the percolation calibration exists to
prevent. The next-largest bundles are localized (5.4% and 1.2% of mask in the
unadjusted arm) but reach only `p_FWER = 0.14` and above.

The escape route does not exist: integrating from a sub-critical floor
(z_min = 4.6, above the calibrated `p_CF = 5e-6` equivalent of z = 4.565)
was tried and is *worse* (`p_FWER = 0.108`). TFCE at this scale is
super-critical where it has power and under-powered where it is sub-critical.

**Conclusion, superseding the tentative reading above: neither TFCE nor
bundle-level FDR improves on the existing pipeline.** Mass + a null-calibrated
sub-critical `p_CF` + max-statistic FWER remains the best of the approaches
tried -- it is the only one that returned *localized*, interpretable
significant bundles (2 bundles, ~2.5% of mask, in the unadjusted arm). This
retroactively supports the percolation-calibration work rather than
superseding it, and the adjusted arm's null stands as a genuine
effect-size/localization limitation rather than an artifact of the correction
scheme.

**BH's complete-null rate replicates on the unadjusted arm**: 0.0261
(95% CI 0.0232-0.0294) at q=0.05 and 0.0654 (0.0607-0.0704) at q=0.10, against
BY's 0.0019 and 0.0059. Three independent measurements now agree that BH
controls at roughly half its nominal budget here.

### Reproducibility

Unadjusted-arm artifacts under
`/mnt/storage/MOCCA_UCLA/tfce_fdr_unadjusted_controlsVSpatients/`
(`inference_10k/`, `fdr_null_calibration/`); the command is the adjusted one
below with `--freedman-lane-plan` omitted and the group-a permutation file
`welch_fisherz_controlsVSpatients/permutations_groupA.txt` substituted.

Adjusted-arm artifacts under
`/mnt/storage/MOCCA_UCLA/tfce_fdr_adjusted_controlsVSpatients/`:
`calibration_1k/` ((E, H) calibration), `inference_10k/` (production TFCE+FDR
at E=0.25, H=1), `reintegrated_E0.5_H2/` and `reintegrated_E0.5_H2_z4.6/`
(re-integrations), and `fdr_null_calibration/` under the first and third.

The 10,001 v3 sparse files were retained (`--keep-sparse`, 21 GB), which is
what made the alternative specifications cost about four CPU-minutes each
instead of a fresh ~4-hour GPU pass. Measured costs: CUDA ~665 s per
invocation (dominated by the one-time 452 GiB read), TFCE bundling ~28 s per
1,000 permutations at `z_min = 4.0`.

```bash
.venv/bin/python 02_cudaPerm/tfce_calibration.py FILELIST PERMS CAL_DIR \
  --cluster-forming-p 1e-4 --tfce-z-min 4.0 --tfce-z-max 7.0 \
  --extent-exponents 0.25 0.5 0.75 1.0 --height-exponents 1.0 2.0 3.0 \
  --freedman-lane-plan PLAN --calibration-permutations 1000

.venv/bin/python 02_cudaPerm/run_bundle_fwer.py FILELIST PERMS OUT \
  --statistic tfce --fdr --fdr-q 0.05 --fdr-method bh \
  --cluster-forming-p 1e-4 --tfce-z-min 4.0 --tfce-z-max 7.0 --tfce-z-step 0.1 \
  --tfce-extent-exponent 0.5 --tfce-height-exponent 2.0 \
  --freedman-lane-plan PLAN --inference-permutations 10000 \
  --bundle-engine cpp --bundle-threads 16 --batch-size 2500 --keep-sparse

.venv/bin/python 02_cudaPerm/fdr_null_calibration.py \
  OUT/permutation_bundle_statistics.csv CALIB_OUT --fdr-q 0.05 0.10
```

## 2026-09-03: percolation-calibration rule -- upper-tail gap and p-value floor

**Status:** gap identified and quantified. A reporting-only fix (option 1
below) is the preferred direction but is **not implemented**. No completed
analysis is invalidated by this; see "Scope" at the end.

### The gap

The calibration rule selects `p_CF` by requiring the **95th percentile** of
the null giant-voxel-fraction to stay at or below `epsilon = 0.05`. FWER
inference, however, is decided by the **upper tail** of the null
maximum-bundle-mass distribution. The rule therefore constrains one quantity
at one part of its distribution while inference depends on a different
quantity at a different part.

The order parameter itself is not the problem and should not be changed on
this account: across the 1,000 calibration nulls at `p_CF = 5e-6`,
giant-voxel-fraction ranks max bundle mass well (Spearman 0.951, Pearson
0.791). The gap is in the **tolerance level**, not the metric.

### Why percolating nulls impose a floor on the attainable p-value

Splitting those 1,000 calibration nulls by the `epsilon` criterion at
`p_CF = 5e-6`:

| | count | median max mass | max | exceeds observed (1211.6) |
|---|---:|---:|---:|---:|
| at/below epsilon | 977 (97.7%) | 94.5 | 9,605 | 59 (6.0%) |
| above epsilon | 23 (2.3%) | 13,113 | 285,190 | **23 (100%)** |

Every percolating null exceeded the observed bundle. This is structural, not
chance: a giant component's mass is orders of magnitude beyond what any
spatially focal effect produces, so each percolating null contributes one
effectively guaranteed exceedance. Hence

```text
p_floor  ~=  (1 + n_percolating) / (B + 1)
```

The criterion "95th percentile <= epsilon" is exactly the statement "at most
5% of nulls may percolate." At that permitted rate and B = 10,000, the floor
is `(1 + 500)/10001 = 0.050` -- precisely `alpha`. **A threshold can pass
calibration in full and still make significance at `alpha = 0.05`
arithmetically unattainable for a focal effect.** The rule's tolerance for
contamination is numerically equal to the error rate it exists to support
inference at, leaving no margin between "passes" and "the test cannot
reject." The two values of 0.05 are unrelated conventions -- one a spatial
extent judgement, one an error rate -- that coincide only by accident.

### What happened in the completed runs

The realised percolation rate was 2.3% in the calibration set and 1.29% in
the 10,000-row inference set, giving floors of roughly 0.023 and 0.013. The
adjusted result's `p_FWER = 0.090` came from 900 exceedances, of which only
129 were percolating nulls; the remaining 771 were ordinary nulls that
genuinely exceeded the observed bundle. Excluding all percolating nulls moves
the p-value only from 0.090 to 0.078.

The floor was therefore real but **not the binding constraint**, and the
contamination was **not** what produced the null result -- an earlier
suspicion in this log's threshold-sensitivity entry, corrected there and
restated here for the avoidance of doubt.

### Options considered

1. **Report the implied floor (preferred).** Have
   `percolation_calibration.py` print, per grid point, the percolating-null
   fraction and the resulting `p_floor`, alongside the existing curve and
   stability output. Pure reporting, no change to the selection rule, no
   methodological risk, and it would have surfaced this immediately. Recording
   it in the calibration results JSON would also make the floor auditable per
   run.
2. Decouple the tolerance from `alpha` by testing a higher percentile (99th
   or 99.5th) against the same `epsilon`, permitting at most 0.5%
   percolation and putting the floor an order of magnitude below `alpha`.
   A prospective tightening of an existing parameter; would move the
   recommended operating point stricter.
3. Calibrate directly on the inference quantity (bound the fraction of nulls
   whose maximum mass exceeds a reference "implausible for a focal effect"
   value). More direct, but requires a dataset-dependent mass scale and
   risks circularity in choosing it.
4. Change the bundle statistic so it is less giant-dominated
   (extent-normalised mass, TFCE-style integration, or a non-chaining
   bundler). Substantially larger change; the bounded bundler was already
   tried and rejected on 2026-08-27 for separate reasons.

### Scope

This is a rule-design observation for future calibrations. It does not
retroactively affect any completed run: both the adjusted `p_CF = 5e-6`
inference and the Fisher-`z` unadjusted inference had p-value floors well
below their observed p-values, so neither was floor-limited, and neither
conclusion depends on this issue.

## 2026-09-03: Fisher-z, unadjusted Welch -- isolating the cause (run)

**Status:** run and complete. This isolates the Fisher-`z` transform from the
covariate adjustment as the two confounded changes between the completed
(significant) analysis and the adjusted (null) one above.

### Configuration

Same Fisher-`z` participant matrices as the adjusted run, same partition
(1,000 calibration, 10,000 held-out inference), but the **unadjusted** Welch
statistic (no age/sex adjustment, no Freedman--Lane) -- i.e. only the outcome
scale changed relative to the originally completed analysis.

Calibrated independently (this exact combination had not been calibrated
before): transition `p_CF = 1e-5` (p95 giant-voxel-fraction 0.0463),
recommended operating point `p_CF = 5e-6` (p95 = 0.0223). This is
essentially identical to the original raw-`r` unadjusted calibration
(transition 1e-5 at p95 = 0.0463, recommended 5e-6) and confirms **the
Fisher-`z` transform does not materially change the null percolation
geometry** -- the transition only moved when covariate adjustment was added
(to 5e-6/2e-6 in the adjusted run above). Bootstrap stability: 80.3% for 1e-5
(vs 70.8% for the adjusted run's 5e-6), a more stable calibration than the
adjusted case.

`p_CF = 5e-6` therefore has a genuine calibration margin here (unlike the
adjusted run, where it was the transition point itself) and matches the
threshold used in the originally completed analysis, making this the cleanest
possible apples-to-apples comparison.

### Result

| | original (raw r, unadjusted) | this run (Fisher z, unadjusted) |
|---|---|---|
| top bundle | 5,723 edges, 1,518 voxels (2.5%) | 5,013 edges, 1,336 voxels (2.2%) |
| p_FWER | 0.042 | 0.041 |
| 2nd bundle | 5,026 edges, p=0.045 | 4,661 edges, p=0.041 |
| significant at 0.05 | 2 bundles | **2 bundles** |

Both bundles are negative sign (TLE greater than controls) in both analyses.
`bundle_fwer_precision.py` confirms no bundle's 95% CI straddles alpha, so
this is not a borderline Monte Carlo artifact. Magnitude, footprint, direction
and count of significant bundles are all closely matched to the original
result.

### Conclusion

**The Fisher-`z` transform is not responsible for the loss of significance.**
The covariate adjustment is. This is now isolated cleanly: the only
combination that produces a null result is Fisher `z` **with** age/sex
adjustment; Fisher `z` alone reproduces the original finding almost exactly.

Combined with the threshold-sensitivity diagnostic above (which found the
adjusted analysis's null does not respond to loosening `p_CF` the way the
unadjusted analysis apparently does, ruling out threshold choice as the
explanation within the range tested), the confounded explanation set has been
narrowed to one: **the covariate-adjusted model removes the effect; the
outcome scale and the cluster-forming threshold do not.**

This does not, on its own, establish *why* adjustment removes it (shared
variance genuinely explained by age/sex vs. some other mechanism specific to
the Freedman--Lane implementation) -- but it does mean any manuscript
reporting a Fisher-`z`, non-covariate-adjusted result as the primary
confirmatory analysis, with the covariate-adjusted result reported and
explained as a robustness check that removes significance, is reporting a
result attributable specifically to covariate adjustment and not to the other
methodology changes made this session.

## 2026-09-03: threshold sensitivity diagnostic (adjusted analysis)

**Status:** exploratory/diagnostic only. Not a corrected inference and not a
change to the frozen `p_CF = 5e-6` result above.

### Motivation and a correction to an earlier claim

Prompted by the question of whether a more liberal `p_CF` would recover the
null result, the analyst noted that the existing (pre-session, unadjusted)
`bundle_fwer_controlsVSpatients_10k_p1e-5_dfaware_cpp` run was preferred to
the `p5e-6` run: same bundle shape and location, and COFFEE-DAC's sub-bundle
decomposition rendered it more clearly, with `p1e-6` tried and discarded as
giving nothing. The analyst's initial framing of the `p1e-5` unadjusted
bundle's larger voxel footprint (11.5% of the mask vs 2.5% at `p5e-6`) as
likely percolation contamination was **corrected by the user** and should be
treated as weak evidence on its own -- footprint alone does not distinguish a
genuinely larger real network from diffuse noise, particularly when shape,
location and sub-bundle decomposition are reported as consistent.

### Diagnostic: real observed-row structure across p_CF, adjusted analysis

A single CUDA pass on row 0 only (df-stored, most liberal grid point 1e-5,
~655 s, matching the single-permutation I/O cost) was re-thresholded for free
down to 1e-6. Unlike the unadjusted case, the adjusted top bundle stays
compact across the whole range tested:

| p_CF | bundles | top edges | top mass | top voxels | % of mask |
|---|---:|---:|---:|---:|---:|
| 1e-5 | 714 | 10219 | 2782.8 | 2076 | 3.5% |
| 8e-6 | 647 | 7854 | 2099.3 | 1734 | 2.9% |
| 7e-6 | 599 | 6866 | 1813.1 | 1626 | 2.7% |
| 6e-6 | 530 | 5831 | 1514.7 | 1475 | 2.5% |
| 5e-6 | 469 | 4743 | 1211.6 | 1298 | 2.2% |
| 4e-6 | 407 | 1939 | 609.1 | 353 | 0.6% |
| 3e-6 | 322 | 1552 | 475.2 | 304 | 0.5% |
| 2e-6 | 255 | 1082 | 313.4 | 242 | 0.4% |
| 1e-6 | 149 | 624 | 160.0 | 182 | 0.3% |

Sign is consistently negative (TLE greater than controls) at every threshold.
Growth is smooth with a step between 4e-6 and 5e-6 (mass roughly doubles,
voxel count roughly quadruples). This does not show the ballooning seen in the
unadjusted `p1e-5` bundle and is consistent with the user's account of a
genuine, coherent network.

### Cross-check against real (if lower-resolution) null data

The 1,000-permutation calibration run already computed real null
maximum-bundle-mass distributions at every grid point, from calibration rows
1--1000 -- independent of the 10,000 inference rows used for the frozen
result. Comparing the row-0 observed masses above against those null draws:

| p_CF | observed mass | calibration-null exceedances (of 1000) | rough p |
|---|---:|---:|---:|
| 1e-5 | 2782.8 | 82 | 0.083 |
| 5e-6 | 1211.6 | 82 | 0.083 |

The 5e-6 estimate (0.083) agrees closely with the official 10,000-permutation
result (0.090), a useful independent cross-validation of that result from a
disjoint null set. More importantly: **the rough p-value is essentially
unchanged between 5e-6 and 1e-5** even though the observed mass roughly
doubled, because the calibration null's mass also scales up correspondingly
(q95 1904.9 to 4392.8, also roughly doubling). Observed and null are moving
together here, unlike whatever is happening in the unadjusted analysis between
its own `p5e-6` and `p1e-5` runs (p_FWER 0.042 to 0.016 -- improving with a
more liberal threshold, the opposite pattern).

### Interpretation

Within the range actually tested (1e-6 to 1e-5), refining or loosening the
cluster-forming threshold does not rescue significance for the adjusted
analysis; this does not look like a threshold artifact in this range. The
observed bundle is plausibly real and grows coherently, but the covariate
adjustment appears to suppress its statistical strength relative to the
unadjusted analysis by a fairly stable amount across this threshold range,
rather than by an amount that specifically depends on threshold choice.

This diagnostic cannot separate *why* the adjusted analysis behaves
differently -- covariate adjustment removing genuinely shared variance,
versus some other mechanism -- from the Fisher-`z` transform. The cheapest
remaining experiment to isolate that is `fisher-equal` (already computed, no
new aggregation needed) run through the **unadjusted** Welch statistic: if
that recovers something close to the original unadjusted raw-`r` result, the
z-transform is not the driver and covariate adjustment is implicated; if it
is also null, the z-transform itself would need to be examined. Not run in
this session; would need a fresh group-a-style permutation file (cheap to
generate) plus calibration (~30 min) plus full inference (~3.7 h) if the
threshold needs reconfirming for that combination.

## 2026-09-03: adjusted control--TLE inference at p_CF = 5e-6 (run)

**Status:** run and complete. **No bundle is significant at alpha = 0.05.**

### Validation status at the point this ran

104 regression tests pass across five suites, none regressed by the
Freedman-Lane/adjusted-model work: `01p5_FisherCC` (27, Fisher-z aggregation),
`permutation_rows`/calibration stability (34), the Freedman-Lane Python
reference implementation (28), the CUDA Freedman-Lane backend against that
oracle (8), and the pre-existing unadjusted CUDA/C++ backend regression
(2 GPU + 5 CPU).

### Configuration

HC2 Freedman--Lane on the Fisher-`z` participant matrices, adjusted for
centered age and sex, contrast = group (positive means controls greater).
Row 0 plus the held-out inference rows 1001--11000; calibration rows 1--1000
excluded from both numerator and denominator, so
`p_FWER = (1 + exceedances) / 10001` with a floor of 9.999e-5. Threshold
frozen at `p_CF = 5e-6` before the run. Wall time about 3.7 hours.

### Result

The observed row produced 81,169 suprathreshold edges, 59,911 retained after
pruning, forming 469 bundles -- 423 negative (TLE greater) and 46 positive.
The largest bundle was 4,743 edges. No giant component formed on the observed
data.

| bundle | sign | edges | mass | exceedances | p_FWER |
|---|---|---|---|---|---|
| 46 | -1 | 4743 | 1211.6 | 900 | 0.090 |
| 47 | -1 | 3045 | 906.8 | 1148 | 0.115 |
| 48 | -1 | 2700 | 741.6 | 1327 | 0.133 |
| 0 | +1 | 2210 | 724.2 | 1351 | 0.135 |

Nothing reaches 0.05; one bundle is below 0.10. `bundle_fwer_precision.py`
reports that **no bundle's 95% Clopper--Pearson interval straddles 0.05**, so
10,000 permutations resolves this call and more permutations would not change
it.

### The null tail, and what it does not explain

The null maximum-bundle-mass distribution is strongly heavy-tailed: median
103, q90 1,083, q95 2,484, q99 13,329, maximum 287,738. About **1.29% of null
permutations percolate** (max mass above 10,000), with a median retained-edge
count of 162,187 against an overall null median of 1,713.

This was initially suspected to be the main cause of the null result, on the
grounds that `p_CF = 5e-6` carries no safety margin. **That suspicion was
wrong, and the diagnostic should be recorded because it is the more useful
finding.** Excluding the 129 percolating nulls entirely moves the strongest
bundle's p-value only from 0.090 to 0.078; excluding only the 31 most extreme
moves it to 0.087. The observed maximum sits at the **91st percentile** of the
null distribution, and the non-percolating part of that distribution already
has a 90th percentile of 1,083 against an observed 1,211.6. The observed
effect is simply not extreme relative to ordinary null permutations.

A rerun at `p_CF = 2e-6` would therefore be a legitimate sensitivity analysis
but is **not** expected to change the conclusion. It should not be run in the
hope of a different answer.

### A real gap in the calibration rule

Independently of this result: the calibration criterion tests the **95th
percentile** of the null giant-voxel-fraction against epsilon, but FWER
inference is governed by the **top ~1%** of the maximum-statistic
distribution. A threshold can therefore pass calibration while its FWER null
is still contaminated by percolating permutations, as `p_CF = 5e-6` is here.
Consider adding an upper-tail criterion (for example a 99th-percentile or
max-based ceiling on the giant-voxel-fraction) before the next calibration.
This did not drive the present result but is a genuine weakness of the rule.

### Interpretation

The completed unadjusted analysis reported two bundles at `p_FWER` 0.042 and
0.045 -- already marginal, just under alpha. The confirmatory analysis
adjusted for age and sex, which differ between groups (controls 32.2 years and
8/26 female; TLE 37.1 years and 22/42 female), and additionally moved the
outcome to the Fisher-`z` scale. Under adjustment the effect does not survive.
Three changes are confounded here -- Fisher `z`, covariate adjustment, and an
unchanged threshold whose margin properties changed -- so the drop cannot be
attributed to covariate adjustment alone from this run. A `raw-equal` Fisher
stage run would isolate the scale change if that attribution matters for the
manuscript.

This is a null confirmatory result, and must be reported as such.

## 2026-09-02: production Fisher stage and adjusted calibration (run)

**Status:** run and recorded. **The cluster-forming threshold is not yet
frozen**; no inference has been run and no adjusted result exists.

### Fisher aggregation (complete)

`01p5_FisherCC` `fisher-equal` was run on the existing run-level `ccmat_3mm`
matrices, which did not need regenerating. 278 run matrices became 68
participant Fisher-`z` matrices (26 controls, 42 TLE; 1--6 runs each, mean
4.09), 452 GiB, in `/mnt/storage/MOCCA_UCLA/fisherz_3mm_controlsVSpatients`.

Only **4 values** out of roughly 4.9e11 correlations required clipping:
participant s43 had one edge at exactly `r = 1.0`, and s57 had three at
`r = 1.0000001` to `1.0000002` -- float32 rounding artefacts from the CUDA
correlation, i.e. genuinely outside the valid range rather than merely at the
boundary. Both are patients; these are almost certainly duplicate or
near-constant voxel time series. The largest participant-level Fisher `z` is
9.89 (`r` about 0.9999999), which is a real near-unit correlation surviving
into a participant mean, not a clipping artefact.

A completed participant was verified bitwise against a direct float64
`mean(atanh(r))` over 300,000 randomly sampled real edges.

### Group direction

Group A is exactly the 26 controls, verified against the diagnosis tags rather
than assumed. A positive `beta_group` therefore means **controls greater than
TLE**.

### Adjusted calibration (complete)

1,000 calibration permutations (rows 1--1000 only; inference rows 1001--11000
held out and never read), HC2 Freedman--Lane statistic on the Fisher-`z`
participant matrices, residual df 64, grid 3e-4 down to 1e-6. CUDA pass 1,711 s
including the 452 GiB read; bundling 160 s.

**The transition moved one grid step stricter.** 95th-percentile null
giant-voxel-fraction against the predeclared epsilon of 0.05:

```text
p_CF     previous (unadjusted Welch, raw r, 200 rows)   this run (adjusted, Fisher z, 1000 rows)
1e-5     0.0463  (passes, by 7%)                        0.0525  (fails, by 5%)
5e-6     0.0188                                         0.0283
2e-6     0.0092                                         0.0125
```

- previous: transition 1e-5, recommended operating point **5e-6**
- this run: transition **5e-6**, recommended operating point **2e-6**

So `p_CF = 5e-6` is **not** confirmed as a safe operating point for the
adjusted analysis. It is now the transition estimate itself, and using it
would mean operating at the transition with no safety margin -- exactly what
the one-grid-step-stricter rule exists to prevent.

The honest reading is that 1e-5 has been sitting on the epsilon boundary all
along: it passed at 0.0463 and fails at 0.0525, both within about 5--7% of
epsilon. Two things changed at once -- the analysis (Fisher `z` plus
covariate adjustment) and the calibration sample (200 to 1,000 rows, so
roughly 50 rather than 10 observations in the tail the 95th percentile
depends on). They cannot be separated from this run alone, and the earlier
"pass" should be treated as having been marginal and under-resolved rather
than as having been contradicted.

### Selection stability

The predeclared stability check flagged the choice, correctly:

- bootstrap over the 1,000 calibration rows: transition 5e-6 in 70.8% of
  replicates, 1e-5 in 29.2%;
- four disjoint 250-row blocks selected [5e-6, 1e-5, 5e-6, 5e-6].

The instability is entirely about whether 1e-5 clears epsilon; nothing below
it is in question. Note that the predeclared one-step-stricter rule already
resolves it conservatively: if the transition is 5e-6 the rule gives 2e-6, and
if it is 1e-5 the rule gives 5e-6, so **2e-6 is valid under either reading**.

### Threshold frozen: p_CF = 5e-6

**Decided by the analyst on 2026-09-03, before any inference was run and
before any observed bundle statistic was inspected.** Inference uses
`p_CF = 5e-6`.

What this choice is and is not:

- 5e-6 **passes** the predeclared sub-criticality test on its own terms: its
  95th-percentile null giant-voxel-fraction is 0.0283, comfortably under the
  epsilon of 0.05. It is not a super-critical threshold.
- It is the *most liberal* passing grid point, i.e. the transition estimate
  itself, so it carries **no one-grid-step safety margin** against calibration
  sampling noise. The analyst-recommended alternative was 2e-6.
- It corresponds to the reading in which the transition sits at 1e-5, which
  29.2% of the calibration bootstrap replicates supported and which the
  previous unadjusted calibration selected. Under that reading 5e-6 *is* the
  predeclared operating point.
- It preserves continuity with the published unadjusted analysis, which also
  used `p_CF = 5e-6`, making the adjusted and unadjusted results directly
  comparable at a matched threshold.

The absence of a safety margin must be reported. Any bundle whose
significance would change under 2e-6 should be treated as
threshold-sensitive; a sensitivity analysis at 2e-6 is the natural check, and
would have to reuse the same held-out inference rows and be declared as a
sensitivity analysis rather than a second primary result.

Not chosen, and recorded for completeness: adopting 2e-6 (the predeclared rule
on the point estimate, safe under both bootstrap readings); enlarging the
calibration set, which would need a new master permutation file because the
current partition allocates exactly 1,000 calibration rows and inference rows
must not be borrowed; or adopting a stricter predeclared rule.

## 2026-09-02: covariate-adjusted control--TLE analysis

**Status:** accepted; implemented end to end and validated on GPU on
2026-09-02 (`02_cudaPerm/design_matrix.py`, `02_cudaPerm/freedman_lane.py`,
full-index permutation support in `generatePermutations.py`, and the
`--freedman-lane` path in `permutationTest_cuda_bundle.cu` driven by
`run_bundle_fwer.py --freedman-lane-plan`). **Not yet run in production, and
no adjusted result has been produced.** Nothing from this analysis may be
described in the manuscript.

Implementation notes recorded against the requirements below:

- **Statistic.** The heteroscedasticity-robust choice is HC2, and the choice
  is not a compromise: for a two-group design with no covariates the
  HC2-studentized group coefficient equals Welch's t *exactly* (asserted to 12
  decimal places in `regression_freedman_lane.py`; HC0 and HC3 do not). The
  adjusted analysis is therefore a strict generalization of the published one
  rather than a different statistic.
- **Design.** `design_matrix.py` builds the primary model with age centered on
  the analysis-sample mean and sex as a 1 = female indicator, and writes the
  exact coding, centering constant, reference levels, contrast vector, rank
  and condition number to `design_manifest.json`. Verified against the real
  table: 26 controls / 42 patients, mean ages 32.2 and 37.1, 8/26 and 22/42
  female, all six left-handers in the patient group, 62 right-handers.
- **Handedness** is refused as a primary covariate and requires an explicit
  `--allow-confounded` override; `--restrict-handedness R` produces the
  preferred 62-participant sensitivity design (26 controls, 36 patients).
  **Run count** is available only on explicit request. **Motion** has no code
  path at all -- the manifest records its absence as a provenance limitation
  rather than inferring anything.
- **Permutation representation.** Freedman--Lane needs a complete reordering
  of all 68 residual vectors, so `--representation full-index` was added to
  `generatePermutations.py`. The default stays `group-a`, leaving the
  unadjusted Welch pipeline untouched. Validation rejects a group-membership
  file supplied where a full-index one is required, confirming that the
  existing subject-level permutation files cannot be reused as-is.
- **Tractability.** The naive form needs one regression per (edge,
  permutation) pair. Using `a'Z = 0` and `M_X H_Z = 0`, a Freedman--Lane draw
  depends on the data only through the nuisance residuals `u = M_Z y`, which
  are computed once per edge and reused by every permutation; the numerator is
  `a'Pu` and the squared denominator is the quadratic form `u'(P'KP)u` for one
  fixed `K`. Packing the upper triangle of `u u'` turns the whole permutation
  set into two dense matrix products per edge chunk. Measured cost: 48.3 MFLOP
  per edge, 86 PFLOP for the full 1.78-billion-edge graph -- about 17 minutes
  at RTX 4090 peak fp32 and roughly 30--50 minutes at realistic GEMM
  efficiency, with a 92 MiB float32 table resident on the GPU.
- **Precision.** float32 tables keep the absolute error on `t` below 2.2e-6
  (measured over 2000 edges x 51 permutations). Relative error grows near
  `t = 0`, where the numerator cancels, but those edges are far from any
  cluster-forming threshold.
- **Exchangeability** is tested, not assumed: uncorrected p-values are uniform
  under the null in the presence of heteroscedastic noise and genuine age/sex
  signal, covariate structure alone does not manufacture group effects, and a
  real group effect is recovered.
- **Degrees of freedom.** An HC-studentized coefficient has no exact
  small-sample null distribution, so the df-aware cluster-forming threshold
  uses the fixed residual df `n - rank(X)` = 64. This is a documented
  approximation used only to convert a cluster-forming p into a `|t|`
  threshold, never to report a p-value; FWER control comes from the
  permutation distribution and does not depend on it.
- **CUDA backend.** `permutationTest_cuda_bundle --freedman-lane PLAN`
  computes `u = M_Z y` once per part, in place, then one thresholding kernel
  per permutation. It uses the projector identities rather than the packed
  GEMM form, so no `n x n` matrix is ever formed and the cost is O(n*p) per
  (edge, permutation). The Welch path is untouched and remains the default.
  The sparse output format, C++ bundler and FWER machinery are unchanged.
- **Measured performance.** 0.56 ns per edge-permutation on the RTX 4090,
  taken as the marginal cost between 201- and 1201-permutation runs so that
  fixture load and file writes cancel. That is about 2.8 hours for the full
  1.78-billion-edge graph at 10,001 permutations. Two changes account for most
  of it: loading each part **subject-major** rather than edge-major, so a warp
  reads one coalesced line instead of 32 separate ones (2.3x, and it also
  turns the reader's strided scatter into a straight copy); and *not* fusing
  the two passes over the data. The fused single-pass form
  `A - 2 d.b + d'Cd` halves DRAM traffic, but it is the unstable computational
  formula and raised the absolute error on `t` from 2.3e-6 to 1.2e-5 in
  measurement, so it was rejected -- accuracy near the cluster-forming
  threshold decides bundle membership. At 0.56 ns the kernel moves ~971 GB/s,
  essentially the card's DRAM bandwidth, so the statistic is memory-bound.
- **GPU validation.** `regression_cuda_freedman_lane.py` requires the CUDA
  backend to select exactly the same suprathreshold edge set as the Python
  oracle on every permutation (agreement 7.3e-7 on the statistic), and drives
  `run_bundle_fwer.py` end to end: a planted spatial effect survives
  bundle-FWER, the calibration row is never computed, and the denominator is
  the inference count plus one. `freedman_lane.py` additionally carries two
  independent implementations (packed GEMM and projector) that agree to
  1.8e-15.
- **Still to do before any adjusted result:** recalibrate the cluster-forming
  threshold. `p_CF = 5e-6` was calibrated on the unadjusted Welch null and
  does not carry over to the adjusted statistic's null geometry. Run
  `percolation_calibration.py` on the held-out calibration rows first.

### Reason for the revision

The completed control--TLE bundle analysis uses unadjusted group-label
permutations. Its edge statistic is a two-sample Welch statistic and its null
assignments preserve the observed group sizes, but the model contains no
participant-level nuisance covariates. A confirmatory analysis should estimate
the group term while accounting for the demographic covariates that are
available for the full sample.

### Covariate audit

`data/share_with_KI/KI_shared_subjects_list.csv` contains complete age, sex,
and handedness fields for all 68 participants.

- **Age:** include as a centered continuous covariate in the primary adjusted
  model. Controls are 32.2 +/- 8.9 years old and TLE participants are
  37.1 +/- 10.8 years old (a mean difference of approximately 4.9 years).
- **Sex:** include as a categorical covariate in the primary adjusted model.
  The sample contains 8/26 female controls and 22/42 female TLE participants.
- **Handedness:** do not include automatically as an ordinary primary-model
  covariate. All six left-handed participants belong to the TLE group, so there
  is no left-handed control overlap with which to distinguish group and
  handedness effects reliably. Prefer a sensitivity analysis restricted to the
  62 right-handed participants (26 controls and 36 TLE participants).
- **Number of usable runs:** this is derivable from the input file list.
  Controls have 4--5 runs (mean 4.31); TLE participants have 1--6 runs (mean
  3.95). Treat run count as a measurement-precision/data-availability issue,
  not automatically as a biological confound. Decide prospectively whether to
  examine it in a sensitivity analysis or account for unequal precision by a
  justified weighting model.
- **TLE laterality and diagnosis subtype:** do not use these as nuisance
  covariates in the primary control--TLE contrast because they are nested
  within the patient group.

### Motion information

Participant- or run-level motion summaries were not delivered with this
dataset. The analyst received preprocessed functional images after preprocessing
had been performed by another group. Repository and supplied-data inspection
found no mean framewise displacement, DVARS, censoring fraction, usable-volume
count, or original motion-parameter files.

The archived preprocessing description reports that motion-derived regressors
were used during time-series nuisance regression. That first-level correction
is not equivalent to controlling between-participant residual motion in the
group model. Unless the original preprocessing outputs can be recovered, the
confirmatory model cannot include a motion-summary covariate. This absence must
be reported transparently as a data-provenance limitation; no motion values are
to be inferred or invented from the delivered images.

If original motion traces later become available, define the summary metric
and exclusion/censoring rules before examining adjusted bundle results. Mean FD
and the number or proportion of retained volumes are candidates, but their use
requires a separate, explicit decision.

### Planned statistical model

For participant `i` and edge `e`, the minimal primary model is

```text
r_ie = beta_0e + beta_Ge * group_i
                 + beta_Ae * centered_age_i
                 + beta_Se * sex_i + error_ie
```

The contrast of interest is the coefficient `beta_Ge`. The exact coding and
centering of every design column must be written to the run manifest. Do not
select covariates according to their sample p-values.

The present pipeline deliberately uses a Welch statistic. The adjusted
implementation should retain an appropriate heteroscedasticity-robust,
studentized statistic rather than silently replacing Welch's unequal-variance
model with a pooled-variance statistic.

### Permutation strategy

Covariate adjustment is part of permutation inference; it cannot be applied to
the observed result after comparing it with the existing unadjusted null.
Implement a nuisance-aware general linear-model permutation scheme, with
Freedman--Lane residual permutation as the leading choice:

1. Let `Z = [intercept, centered age, sex]` and fit the reduced nuisance model
   for each edge.
2. Retain its fitted values and residuals.
3. For every null draw, apply one full 68-participant permutation to the
   residuals and add the nuisance fitted values back.
4. Fit/evaluate the full model `[Z, group]` and calculate the studentized group
   statistic.
5. Use the same participant permutation for every edge, then run the complete
   MOCCA thresholding, bundling, pruning, and maximum-bundle calculation.

The nuisance projection and fixed design-matrix algebra may be precomputed for
efficiency. Nevertheless, the adjusted group statistic and bundle maximum must
be evaluated under every permutation.

The existing subject-level permutation files store only the membership of the
26-person group A. They do not specify a full permutation of all 68 residual
vectors and therefore cannot be used as-is for Freedman--Lane inference. New
full-index permutation sets should be generated reproducibly.

Methodological basis: Winkler AM, Ridgway GR, Webster MA, Smith SM, Nichols TE.
Permutation inference for the general linear model. *NeuroImage* 2014;
92:381--397. <https://doi.org/10.1016/j.neuroimage.2014.01.060>.

### Work that can be reused and work that must be rerun

The existing subject-mean connectivity matrices, common mask, spatial
neighbour definitions, and most bundling machinery can be reused if the
first-level connectivity definition remains unchanged.

The following must be rerun for the adjusted analysis:

- the observed edgewise group statistic;
- all nuisance-aware permuted edgewise statistics;
- null-only calibration of the cluster-forming threshold;
- bundle construction and the maximum-bundle null distribution;
- bundle-level FWER p-values;
- any supplementary edgewise result that is claimed to be covariate-adjusted.

If a later decision changes raw `r` aggregation to Fisher `z`, changes motion
censoring/nuisance processing, or otherwise changes first-level connectivity,
the participant matrices must also be regenerated.

### Execution order

Do not start the expensive confirmatory permutation run until the remaining
limitations and sensitivity analyses have been reviewed. Freeze the outcome
scale, run-aggregation rule, nuisance design, heteroscedastic statistic,
calibration/inference separation, permutation count, and random seeds first;
then perform one definitive rerun and preserve a machine-readable manifest.

## 2026-09-02: optional Fisher-transform and participant-aggregation stage

**Status:** accepted; module implemented and validated as
`01p5_FisherCC/fisher_aggregate_ccmat.py` on 2026-09-02. **The production
rerun remains deferred** until the nuisance-adjusted model and the disjoint
calibration/inference permutation design below are also frozen, so that one
definitive analysis evaluates all three together. No result from this stage
has been produced, and none may be described in the manuscript.

Implementation notes recorded against the requirements below:

- All three planned modes are implemented; `fisher-equal` is the default.
  `raw-equal` reproduces `02_cudaPerm/average_ccmat_runs.py` bitwise, so the
  scale comparison is not confounded by an implementation change.
- Validation requirements 1--5 are covered by
  `01p5_FisherCC/regression_fisher_aggregate.py` (27 tests). Requirements 6
  and 7 are production activities and remain outstanding.
- Order invariance is achieved by construction rather than by tolerance:
  runs are sorted into a canonical within-participant order before
  accumulation, so file-list order and chunk size both give bitwise
  identical output.
- The binary container could not carry a distinct magic number: the
  downstream reader `02_cudaPerm/ccmat_io.c:isBinaryCCmat` accepts only
  `CCMAT_MAGIC`. The Fisher `z` scale is declared in the output filename
  (`s<id>_fisherz.ccmat`), a per-file JSON sidecar, and the run manifest.
- No clipping was required on the real run matrices inspected; the largest
  observed `|r|` was 0.9986. The clipping path is exercised by synthetic
  fixtures only.
- Empirical size of the change, measured on participant `s109` over 200,000
  randomly sampled edges: `tanh(mean z)` and `mean r` differ by 0.0011 on
  average and by up to 0.076 at the strongest edges. This is consistent with
  the rationale below --- small in general, largest exactly where threshold
  crossing and bundle membership are decided.
- Cost at production scale: reads approximately 1.8 TiB and writes
  approximately 449 GiB for the 278-run, 68-participant dataset; roughly one
  hour, extrapolated from an 80-second four-run test.

### Decision

Add an optional pipeline stage between `01_cudaCC` and `02_cudaPerm`, with the
working name `01p5_FisherCC`. Its purpose is to transform run-level Pearson
correlation matrices and produce exactly one aggregate matrix per participant
for downstream permutation inference.

The planned default confirmatory path is:

```text
01_cudaCC/run-level Pearson r matrices
    -> 01p5_FisherCC/run-level atanh(r)
    -> equal-run mean Fisher z matrix per participant
    -> 02_cudaPerm/participant-level adjusted permutation inference
```

The downstream inferential unit remains the participant. The module must not
offer or imply a mode in which repeated run-level matrices are entered into the
group permutation test as independent observations. The earlier 112-versus-166
run-level analysis violated participant-level independence and is not a
candidate confirmatory analysis.

Literal temporal concatenation of the supplied four-dimensional images is not
the planned primary method. It would require verified run provenance,
run-specific demeaning/scaling, explicit handling of boundaries, and complete
recalculation of the correlation matrices. The intermediate module instead
operates reproducibly on the already generated run-level correlation matrices.

### Rationale

Pearson `r` is bounded and has a sampling distribution whose skew and variance
depend on the underlying correlation. The Fisher transform

```text
z = atanh(r) = 0.5 * log((1 + r) / (1 - r))
```

places correlations on an unbounded, more nearly variance-stabilized scale for
aggregation and linear group modeling. For modest correlations the numerical
difference from raw-`r` averaging may be small, but rare strong correlations
can be more affected and may alter threshold crossing or bundle membership.
This is therefore a modest but worthwhile robustness improvement rather than a
claim that the existing raw-`r` participant aggregation is intrinsically
invalid.

The participant-level aggregate should remain on the Fisher-`z` scale during
group modeling. Apply `tanh` only to estimates presented on the correlation
scale for interpretation; do not back-transform participant matrices before
the group test.

### Planned modes

The stage should make the aggregation rule explicit rather than hiding it in
file handling:

- `fisher-equal`: transform each run matrix and take an equal-weight mean
  within participant; planned primary mode;
- `fisher-duration`: transform each run matrix and use a prospectively defined
  duration/precision weight; planned sensitivity mode only unless run
  provenance and effective sample-size assumptions justify promotion;
- `raw-equal`: reproduce the currently reported arithmetic mean of raw
  correlations for validation and direct sensitivity comparison.

The supplied four-dimensional files range from 90 to 600 volumes. Nominal
Fisher weights such as `n_timepoints - 3` assume independent observations, but
filtered BOLD series are temporally autocorrelated. Consequently, raw frame
counts must not be presented as exact inverse-variance weights. Any duration
weighting rule and its interpretation must be fixed before viewing the new
bundle results.

### Proposed interface and provenance requirements

Inputs should include the run-level CCMAT file list, participant/group
metadata, aggregation mode, and any optional weight table. Processing should
be streaming/chunked because a complete dense matrix cannot be held in memory
conveniently.

For numerical safety, clamp only values that would otherwise make `atanh`
undefined at exactly `-1` or `1`, accumulate in float64, and record the number
and extrema of clamped values. Do not silently coerce other values.

Outputs should include:

- one binary aggregate matrix per participant;
- a participant-ordered file list for `02_cudaPerm`;
- a machine-readable manifest recording input paths and checksums, participant
  and run mapping, input/output scale, transform, weights, time-point counts,
  clipping policy and count, numeric precision, software version, and Git
  commit;
- validation summaries comparing `fisher-equal`, `fisher-duration`, and
  `raw-equal` without selecting the primary mode from the observed group
  result.

Use an output name or header/manifest field that unmistakably identifies the
values as Fisher `z`, even if the existing CCMAT binary container is reused.

### Validation requirements

Before production use:

1. verify the transform and aggregation against a small float64 reference;
2. verify invariance to processing chunk size and run-file order;
3. verify that `raw-equal` reproduces the current participant matrices within
   documented floating-point tolerance;
4. verify participant ordering and group boundaries independently;
5. test exact/near `-1` and `1`, non-finite inputs, one-run participants, and
   unequal run counts;
6. record distributional and numerical differences between aggregation modes;
7. rerun threshold calibration and all claimed permutation inference after the
   final mode is frozen.

### Implementation order

The manuscript limitations are being reviewed in conceptual order, not in
order of coding difficulty. This Fisher stage is likely easier to implement
than the nuisance-aware permutation model discussed above, while recovery of
motion summaries may be impossible without additional upstream data. Delay the
production rerun until all decisions are frozen so that these changes can be
evaluated together in one definitive analysis.

## 2026-09-02: disjoint calibration and inference permutations

**Status:** accepted; implemented on 2026-09-02 in
`02_cudaPerm/permutation_rows.py` and wired into `generatePermutations.py`,
`percolation_calibration.py` and `run_bundle_fwer.py`. **Not yet run in
production**; the definitive rerun waits on the nuisance-adjusted model.

Implementation notes recorded against the requirements below:

- 1,000 calibration and 10,000 inference permutations are now the defaults in
  every program, so an 11,001-row master file is what a plain invocation
  produces and consumes.
- The four suggested configuration concepts are implemented verbatim as
  `--calibration-permutations`, `--inference-permutations`,
  `--calibration-start-row` and `--inference-start-row`. All three programs
  accept all four and record the whole partition, not only the half they
  consume.
- `--null-permutations` was removed from `run_bundle_fwer.py`. It now raises
  an error naming `--inference-permutations`, rather than silently running a
  different row set under a familiar flag.
- Automated validation rejects overlapping ranges, duplicate rows, a
  non-observed row 0, the observed assignment reappearing as a null, a file
  too short for the declared ranges, and a total row count that does not match
  the partition (overridable only by an explicit
  `--allow-extra-permutation-rows`). `run_bundle_fwer.py` additionally refuses
  to finish unless the assembled null distribution is exactly the declared
  inference rows, so a calibration row cannot leak into a resumed run.
- The regression test over a small synthetic permutation file is
  `02_cudaPerm/regression_permutation_partition.py` (27 tests); its fixture
  has known calibration rows, inference rows, exceedance count and
  denominator. The CUDA end-to-end fixtures in `regression_cuda_bundle.py`
  were extended to a three-row file so they exercise the real partition on the
  GPU rather than bypassing it.
- Calibration stability is assessed by bootstrap resampling and by four-way
  subdivision of the calibration rows only, and is written to
  `percolation_calibration_results.json`. Disagreement between disjoint blocks
  prints an explicit warning to increase the calibration set prospectively or
  adopt a stricter predeclared rule.
- `observed_bundles_fwer.csv` now carries `inference_exceedances` beside
  `p_fwer`, and `bundle_fwer_results.json` records `p_fwer_denominator` and
  `p_fwer_formula`. The config key `null_permutations` was kept, holding the
  inference count, so `bundle_fwer_precision.py` and existing result
  directories keep working.
- Not yet done: the production 11,001-row master file has not been generated.
  Under the covariate-adjusted model each null row must encode a complete
  68-participant residual-permutation order rather than group-A membership, so
  generating it now would produce a file of the wrong representation.

### Decision

For a production analysis described as using 10,000 inference permutations,
generate 11,000 unique null permutations in addition to the observed
assignment. Partition one reproducible master permutation set as follows:

```text
row 0             observed assignment
rows 1--1000      calibration set only (1,000 null permutations)
rows 1001--11000  inference set only (10,000 null permutations)
```

The two null subsets must be disjoint. Separate random seeds or files are not
required; one uniquely generated master file with recorded, non-overlapping
row ranges is easier to audit and guarantees that a label assignment is not
deliberately reused across the two stages.

For the planned nuisance-adjusted analysis, each null row will need to encode a
complete 68-participant residual-permutation order suitable for the selected
Freedman--Lane implementation, rather than only the membership of group A.

### Calibration stage

Use only rows 1--1000 to evaluate the predeclared cluster-forming-threshold
grid. Exclude row 0 and all inference rows. Apply the predeclared percolation
criterion to select the transition, retain the existing one-grid-step stricter
safety margin, and then freeze the chosen cluster-forming threshold before
examining the observed bundles or inference-set results.

One thousand calibration permutations provide approximately 50 observations
in the upper 5% tail used by a 95th-percentile rule, compared with approximately
10 observations when only 200 permutations are used. Assess selection
stability by resampling or subdividing the 1,000 calibration rows only. Do not
use inference rows to resolve an unstable choice. If stability is inadequate,
increase the calibration set prospectively or adopt a predeclared stricter
rule before running inference.

### Inference stage

After the cluster-forming threshold is frozen, apply it to row 0 and rows
1001--11000. Bundle-level FWER p-values must use only the 10,000 held-out
inference maxima:

```text
p_FWER = (1 + number of inference maxima >= observed statistic) / 10001
```

Calibration maxima must not enter the numerator or denominator. Consequently,
the minimum attainable production p-value remains `1 / 10001`, even though
11,000 null permutations were computed in total.

### Pipeline changes and provenance

The existing calibration and production programs can retain their core
statistic and bundling implementations. Redesign their orchestration so that
the row partition is an explicit, validated configuration rather than an
implicit convention. Suggested command/configuration concepts are:

```text
--calibration-permutations 1000
--inference-permutations 10000
--calibration-start-row 1
--inference-start-row 1001
```

The calibration manifest, production manifest, and final result metadata must
record:

- master permutation-file path and checksum;
- generator seed and permutation representation;
- observed, calibration, and inference row ranges;
- number of unique rows in each range and across the complete file;
- selected threshold, full calibration grid, criterion, safety margin, and
  calibration stability summary;
- the fact that calibration rows were excluded from FWER calculation;
- inference exceedance counts and the exact p-value denominator.

Automated validation must reject overlapping row ranges, duplicate rows when
uniqueness is required, a non-observed row 0, incorrect total row count, and
off-by-one errors in either subset. A regression test should use a small
synthetic permutation file for which the expected calibration rows, inference
rows, exceedance count, and denominator are known exactly.

### Interpretation of the existing run

The completed analysis used rows 1--200 for calibration and rows 1--10000 for
inference, so the calibration set was a 200-row subset of the production null.
This was not selection from the observed row and had a negligible numerical
effect in a diagnostic recalculation, but it is less clean than a held-out
design. It should remain reported accurately as the implementation used for
the current draft. Adopt the disjoint 1,000-plus-10,000 design in the eventual
combined Fisher-transform and covariate-adjusted confirmatory rerun.
