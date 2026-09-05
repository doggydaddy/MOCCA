# analysis_notes

The tracked technical record of the MOCCA analysis: methodological decisions,
completed runs, diagnostics, and the reasoning behind them, in chronological
order. This directory exists so that record survives in git independent of
`manuscript/`, which is gitignored (the manuscript itself may be held back
pre-publication; the underlying analysis should not be).

## Contents

- **`ANALYSIS_DECISIONS.md`** — the decision log. Dated entries, newest first.
  A decision marked **planned** must not be described as completed until the
  corresponding run and validation exist. This is the complete technical
  record: every methodological choice, every run's configuration and result,
  every diagnostic, and — where one was wrong — the correction, left in place
  rather than silently edited out.
- **`conversation_archives_2026-08-26.md`**, **`...-08-28.md`**,
  **`...-09-03.md`** — session-by-session narrative accounts that predate this
  directory's creation (2026-09-05), continuing one another in sequence.
  Retained for the reasoning trail behind entries in the decision log that
  point back to them; not maintained going forward in favor of the log itself.

## Relationship to `manuscript/`

`manuscript/APPENDIX_COVARIATE_ADJUSTED_ANALYSIS.md` is the manuscript-styled
(LaTeX-convention) version of one strand of this log — the covariate-adjusted
confirmatory reanalysis and its diagnostics — written to be adapted directly
into the paper. It is a **subset and a restyling**, not an independent record:
everything in it (or its substance) is also in `ANALYSIS_DECISIONS.md`, so
losing access to `manuscript/` does not lose the underlying analysis. New
findings should be recorded here first; porting to the appendix's prose is a
separate, later step.

## Adding to this log

New dated entries go at the top of `ANALYSIS_DECISIONS.md`, newest first. When
a later result contradicts an earlier entry (as happened twice in the
2026-09-04 covariate work), correct the record in place rather than only
appending — leave the wrong reading visible with what refuted it, since the
refutation is often the more informative part.
