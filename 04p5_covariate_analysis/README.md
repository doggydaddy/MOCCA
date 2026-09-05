# 04p5 — Post-hoc covariate analysis

Tools and run scripts for the covariate investigation of the controls-vs-TLE
bundle result. Everything here is **post-hoc analysis of completed runs**, not
part of the production pipeline in `02_cudaPerm/`. It is kept separate for
that reason: none of it should be mistaken for a step that produces a primary
result.

Findings are written up in `manuscript/ANALYSIS_DECISIONS.md` (2026-09-04,
"which covariate removes the effect") and
`manuscript/APPENDIX_COVARIATE_ADJUSTED_ANALYSIS.md` §§3.6–3.8.

## The finding, in one table

Four models, identical in every other respect (Fisher-*z*, HC2 Freedman–Lane,
mass statistic, `p_CF = 5e-6`, max-statistic FWER, same 1,000/10,000 disjoint
partition):

| model | significant | top `p_FWER` | top bundle edges | top mass | null `q95` | obs / null | coherence |
|---|---:|---:|---:|---:|---:|---:|---:|
| unadjusted | 2 | 0.0407 | 5,013 | 1583 | 1260 | 1.26 | 0.084 |
| group + age | 2 | 0.0349 | 8,258 | 2920 | 1949 | 1.50 | 0.093 |
| group + sex | 1 | 0.0382 | 7,705 | 2585 | 1942 | 1.33 | 0.121 |
| group + age + sex | 0 | 0.0901 | 4,743 | 1212 | 2484 | **0.49** | 0.079 |

**Neither covariate alone removes the effect; only their combination does.**
Each covariate on its own *increases* the observed signal (ordinary precision
gain — less nuisance variance, larger `t`, more edges above threshold), and the
permutation null rises in step, so `p_FWER` barely moves. Only jointly do the
two move against each other.

Collinearity does not explain it: the joint VIF for group is 1.121 (1.057 and
1.047 marginally), a ~6% inflation of the group coefficient's standard error.
The mechanism is **not established** — three candidate explanations were
proposed and each refuted. See the appendix.

## A methodological warning worth carrying forward

An earlier descriptive decomposition (`covariate_decomposition.py`) attributed
the loss to *age*, cleanly and with a plausible mechanism. A pre-declared
confirmatory run then refuted it. The reason is general:

> An observed-only decomposition cannot rank covariates by their effect on
> significance. It measures numerators, while inference is decided by a ratio
> against a null that changes with the model.

Age removes 59% of the observed bundle mass and changes `p_FWER` hardly at all,
because its permutation null shrinks by a comparable factor. Only running each
candidate model's own null answers the question. Use
`covariate_decomposition.py` to ask *where variance sits*, never *what drives
the result*.

## Contents

| file | purpose |
|---|---|
| `covariate_decomposition.py` | Refits nested models on a fixed edge set (previously significant bundles vs a background set) and compares the HC2 group statistic. **Descriptive only** — see warning above. |
| `sensitivity_series.py` | Assembles completed runs into the comparison table above. |
| `covariate_collinearity.py` | `R²` and VIF of the group contrast against each covariate set; the audit trail for rejecting the collinearity explanation. |
| `run_ageonly.sh`, `run_sexonly.sh` | The two confirmatory whole-brain runs, both pre-declared before execution. |
| `regression_covariate_analysis.py` | 8 tests: vectorized HC2 against `freedman_lane.statistic_direct` and against Welch; condensed-index round trip and orientation invariance; random-access edge reads; explained-variance identities. |

These import `design_matrix` and `freedman_lane` from `../02_cudaPerm`, which
they add to `sys.path` themselves; run them from anywhere.

## Reproducing

```bash
# The four-model comparison from completed run directories
.venv/bin/python 04p5_covariate_analysis/sensitivity_series.py \
  "unadjusted=$B/welch_fisherz_controlsVSpatients/inference_10k_p5e-6" \
  "group + age=$B/ageonly_controlsVSpatients/inference_10k_p5e-6" \
  "group + sex=$B/sexonly_controlsVSpatients/inference_10k_p5e-6" \
  "group + age + sex=$B/adjusted_controlsVSpatients/inference_10k_p5e-6"

# Why collinearity is not the explanation
.venv/bin/python 04p5_covariate_analysis/covariate_collinearity.py \
  $B/adjusted_controlsVSpatients/design/design.npz

# Where the variance sits (descriptive; read the warning first)
.venv/bin/python 04p5_covariate_analysis/covariate_decomposition.py \
  $B/fisherz_3mm_controlsVSpatients/participants.txt \
  $B/welch_fisherz_controlsVSpatients/inference_10k_p5e-6/observed_edges_bundled.csv \
  OUTPUT_DIR --group-a-subjects 26 --bundles 66 67

.venv/bin/python 04p5_covariate_analysis/regression_covariate_analysis.py
```

A single-covariate whole-brain run is `design_matrix.py --no-sex` (or
`--no-age`) followed by `freedman_lane.py` and `run_bundle_fwer.py`; both shell
scripts here show the full invocation.

## Note on cost

`covariate_decomposition.py` reads individual edges from the 68 participant
matrices (~7 GB each) by seeking directly into the flat float32 payload behind
the 24-byte ccmat header. It sets `posix_fadvise(POSIX_FADV_RANDOM)` first:
without it the kernel treats every 4-byte seek as the start of a sequential
scan, which cost ~287 GB of device reads for ~5 MB of wanted data on the first
run.
