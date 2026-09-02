# 01p5_FisherCC — Fisher transform and participant aggregation

> **Status: implemented, validated, not yet run in production.**
> `manuscript/ANALYSIS_DECISIONS.md` (2026-09-02, "optional Fisher-transform
> and participant-aggregation stage") accepts this stage in principle but
> defers the production rerun until the nuisance-adjusted model, the
> disjoint calibration/inference permutation design, and the remaining
> manuscript limitations are all frozen, so that one definitive analysis
> evaluates them together. Nothing here should be described in the
> manuscript as a completed analysis.

Optional stage between `01_cudaCC` and `02_cudaPerm`. It turns run-level
Pearson `r` matrices into **exactly one aggregate matrix per participant**:

```text
01_cudaCC run-level Pearson r matrices
    -> 01p5_FisherCC run-level atanh(r)
    -> equal-run mean Fisher z matrix per participant
    -> 02_cudaPerm participant-level permutation inference
```

## Why

Pearson `r` is bounded, and the skew and variance of its sampling
distribution depend on the underlying correlation. The Fisher transform

```text
z = atanh(r) = 0.5 * log((1 + r) / (1 - r))
```

puts correlations on an unbounded, more nearly variance-stabilized scale
before they are averaged within participant and entered into a linear group
model. This is a modest robustness improvement, not a claim that the
existing raw-`r` participant mean is invalid: on real data (`s109`, two
runs, 200k randomly sampled edges) `tanh(mean z)` and `mean r` differ by
0.0011 on average — but by up to 0.076 at the strongest edges, which is
exactly where threshold crossing and bundle membership are decided.

**The aggregate stays on the Fisher-`z` scale through the group test.**
Apply `tanh` only to estimates that are presented on the correlation scale
for interpretation. Participant matrices are never back-transformed here;
the manifest records `"back_transformed": false`.

## The inferential unit is the participant

This module offers no mode in which repeated run-level matrices enter the
group permutation test as independent observations. The earlier
112-versus-166 *run-level* analysis violated participant-level independence
and is not a candidate confirmatory analysis.

Literal temporal concatenation of the supplied 4D images is also not the
planned method — it would need verified run provenance, run-specific
demeaning/scaling, explicit boundary handling, and full recalculation of the
correlation matrices. This stage instead operates reproducibly on the
already-generated run-level correlation matrices.

## Modes

| Mode | Transform | Weighting | Output name | Role |
|---|---|---|---|---|
| `fisher-equal` | `atanh` | equal | `s<id>_fisherz.ccmat` | **planned primary** |
| `fisher-duration` | `atanh` | supplied weight table | `s<id>_fisherz_w.ccmat` | sensitivity only |
| `raw-equal` | none | equal | `s<id>_rawmean.ccmat` | validation / reproduces the current pipeline |

`raw-equal` is bitwise identical to `02_cudaPerm/average_ccmat_runs.py`
(asserted by the regression suite), so the two aggregation scales can be
compared without confounding the comparison with an implementation change.

### On duration weighting

`fisher-duration` requires an explicit `--weight-table`. This program never
derives weights from the data. The supplied 4D files range from 90 to 600
volumes, and the nominal Fisher weight `n_timepoints - 3` assumes
independent observations — filtered BOLD series are temporally
autocorrelated, so raw frame counts **must not** be presented as exact
inverse-variance weights. `--timepoint-column` carries frame counts into the
manifest as provenance only; it never becomes a weight. Any weighting rule
and its interpretation must be fixed before the resulting bundles are
viewed.

Weight table format (keyed on the run matrix filename):

```csv
run,weight,n_timepoints
s109_1.ccmat,1.0,180
s109_2.ccmat,0.8,152
```

## Usage

```bash
.venv/bin/python 01p5_FisherCC/fisher_aggregate_ccmat.py \
  --file-list data/share_with_KI/filelist_controlsVSpatients_runAll_ccmat3mm.txt \
  --group-a-runs 112 \
  --output-dir /mnt/storage/MOCCA_UCLA/fisherz_3mm_controlsVSpatients \
  --output-file-list /mnt/storage/MOCCA_UCLA/fisherz_3mm_controlsVSpatients/participants.txt \
  --mode fisher-equal
```

`--dry-run` validates the file list, participant grouping, group boundary,
weights and free space, then stops without writing. Use it first.

On the controls-versus-patients file list this resolves to 278 run matrices
→ 68 participants (26 group A, 42 group B), 59,677 voxels, 1.7806 billion
edges. The stage then prints the `-nA`/`-nB` values that
`02_cudaPerm/generatePermutations.py` needs, so the group sizes cannot drift
between the two steps.

**Scale.** Each matrix is 6.6 GiB. A full run reads ~1.8 TiB and writes
~449 GiB; a 4-run smoke test took 80 s, extrapolating to roughly an hour.
Launch it in a named `tmux` session so it can be monitored independently:

```bash
tmux new-session -d -s fisher_agg \
  ".venv/bin/python 01p5_FisherCC/fisher_aggregate_ccmat.py ... 2>&1 | tee OUTPUT_DIR/run.log"
tmux attach -t fisher_agg   # detach: Ctrl-b d
```

A pre-flight check refuses to start when the output volume cannot hold the
matrices that still need writing (`--allow-low-space` overrides), rather
than dying on ENOSPC an hour in. Outputs already present at full size are
not counted, so resuming is never blocked by space the run already claimed.

### Resuming

A participant is skipped only when its output container *and* its JSON
sidecar match the current mode, transform, run set and weights. Changing
mode or weights therefore recomputes rather than silently reusing a matrix
built under different rules. `--force` recomputes everything.

## Outputs

- `s<id>_<suffix>.ccmat` — one aggregate matrix per participant, in the
  unchanged binary CCMAT container.
- `s<id>_<suffix>.ccmat.json` — per-file sidecar: mode, transform,
  input/output scale, group, run list, weights, clipping statistics,
  precision, timestamp, and software/git provenance. Provenance travels with
  the file if it is moved.
- `participants.txt` — participant-ordered file list for `02_cudaPerm`,
  group A first.
- `fisher_aggregation_manifest_<mode>.json` — the run manifest: input paths
  and fingerprints, participant/run mapping, input and output scale,
  transform, weights, time-point counts, clipping policy and count, numeric
  precision, chunk size, free space at start, command line, software
  versions, and git commit/dirty state.

### Why the scale is not in the binary header

The downstream C reader `02_cudaPerm/ccmat_io.c:isBinaryCCmat` accepts only
`CCMAT_MAGIC`, and the 24-byte header has no spare field. A distinct magic
would break `02_cudaPerm` outright. The Fisher-`z` identity is therefore
declared unmistakably in the **output filename**, the **per-file sidecar**,
and the **manifest** instead — a file named `s109_fisherz.ccmat` cannot be
mistaken for a raw-`r` mean.

## Numerical policy

- **Accumulation** is float64 and strictly element-wise; storage is float32,
  matching the container and what `02_cudaPerm` reads.
- **Chunking** does not change any per-element arithmetic, so output is
  bitwise identical for any `--chunk-elements`.
- **Run order** does not matter: runs are sorted into a canonical
  within-participant order before accumulation, so a reordered input file
  list gives a bitwise identical result rather than a nearly identical one.
  Participant order still follows the input file list, preserving the group
  boundary.
- **Clipping** applies only to values that would make `atanh` undefined or
  NaN, i.e. `|r| >= 1`, which are clipped to `±nextafter(1, 0)` in float64
  (max `|z|` ≈ 18.37). No other value is modified. The count is split into
  exactly-unit and beyond-unit values, with the extrema of the clipped
  inputs, and recorded in both sidecar and manifest. No clipping was needed
  on the real run matrices inspected (max `|r|` = 0.9986).
- **Non-finite inputs** are rejected with the offending file and element
  index. They are never silently coerced.

## Regression checks

```bash
.venv/bin/python -m unittest 01p5_FisherCC.regression_fisher_aggregate
```

27 tests covering validation requirements 1–5 from the decision log:
transform and aggregation against a small float64 reference (all three
modes); bitwise invariance to chunk size and run-file order; `raw-equal`
reproducing `average_ccmat_runs.py`; independent verification of participant
ordering and group boundaries; and the edge cases — exact and near `±1`,
out-of-range values, non-finite inputs, one-run participants, and unequal
run counts. Also covered: mode/weight-table argument validation, weights
resolving before any I/O, resume safety, the downstream header contract, and
the free-space guard.

Requirements 6 (recording distributional differences between modes) and 7
(rerunning threshold calibration and all claimed inference after the mode is
frozen) are production activities, not unit tests.

## Before production use

Per the decision log, a Fisher-`z` first-level change invalidates everything
downstream of it. After freezing the mode, rerun in order:

1. `02_cudaPerm/generatePermutations.py` — new permutations for the frozen
   design (and, once the covariate-adjusted model lands, full 68-participant
   index permutations rather than group-A membership rows).
2. `02_cudaPerm/percolation_calibration.py` — the cluster-forming threshold
   must be recalibrated; the existing `p_CF = 5e-6` was calibrated on the
   raw-`r` participant matrices and does not carry over unexamined.
3. `02_cudaPerm/run_bundle_fwer.py` — observed and null bundle statistics.
4. `03_prepResultsForVisualization/` and `04_coffee-dac/` exports.
