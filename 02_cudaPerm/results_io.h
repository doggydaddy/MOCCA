/*
 * results_io.h  —  Writing and resuming permutation test output files
 *
 * Covers both text (.permout text) and binary (.permout binary) output,
 * including full-result saves, partial appends, and resume-detection.
 */
#ifndef RESULTS_IO_H
#define RESULTS_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* Binary format constants are needed here for saveResToBinary etc. */
#include "ccmat_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Full-result saves (single-chunk path) ── */
void saveResToText  (float *outputData, size_t nrows, char *fileName);
void saveResToBinary(float *outputData, size_t nrows, char *fileName);

/* ── Resume detection (streaming path) ── */
size_t countExistingResults      (char *fileName, size_t nrows);
size_t countExistingResultsBinary(char *fileName, size_t nrows);

/* ── Partial appends (streaming path) ── */
void appendPartialResults      (float *outputData, size_t start_idx, size_t end_idx,
                                size_t nrows, char *fileName, int is_first_write);
void appendPartialResultsBinary(FILE *output, float *outputData, size_t n_vals);

#ifdef __cplusplus
}
#endif

#endif /* RESULTS_IO_H */
