/*
 * perm_kernels.cuh  —  CUDA kernel declarations for permutation testing
 *
 * Declares the device helper and the three kernels used by
 * permutationTest_cuda_fwer:
 *
 *   CUDA_perm              — standard per-connection permutation test
 *   CUDA_perm_fwer_pass1   — FWER pass 1: build per-permutation max-|t| distribution
 *   CUDA_perm_fwer_pass2   — FWER pass 2: compute FWER-corrected p-values
 */
#ifndef PERM_KERNELS_CUH
#define PERM_KERNELS_CUH

#include <cuda_runtime.h>
#include <stddef.h>

/* ── Device helper ── */

///
/// @brief Atomic max for non-negative floats (CUDA has no native float atomicMax).
///
/// Uses the __int_as_float / __float_as_int reinterpretation trick.
/// Valid because IEEE 754 positive floats have the same bit-order as unsigned
/// integers when sign bits are equal, and we only ever store |t| values (>= 0).
///
__device__ float atomicMaxFloat(float* addr, float value);

/* ── Kernels ── */

///
/// @brief Standard per-connection permutation test.
/// @param input        subject data buffer, layout: [conn * nr_sub + sub]
/// @param onehot       one-hot permutation matrix, layout: [perm * nr_sub + sub]
/// @param nr_vals      number of connections in this chunk
/// @param nr_sub       number of subjects
/// @param nr_perm      number of permutations (including observed, perm 0)
/// @param output_pval  output: uncorrected p-values (one per connection)
/// @param output_tstat output: observed Welch's t-statistics
/// @param two_tailed   1 = two-tailed (|t|), 0 = one-tailed (directional)
///
/// Each block processes one connection; threads within a block stripe across
/// permutations.  The p-value uses the standard +1 correction:
///   p = (count + 1) / (nr_perm + 1)
///
__global__
void CUDA_perm(float *input, int *onehot,
               size_t nr_vals, size_t nr_sub, size_t nr_perm,
               float *output_pval, float *output_tstat, int two_tailed);

///
/// @brief FWER pass 1: accumulate per-permutation global max |t|.
/// @param input        subject data buffer, layout: [conn * nr_sub + sub]
/// @param onehot       one-hot permutation matrix, layout: [perm * nr_sub + sub]
/// @param nr_vals      number of connections in this chunk
/// @param nr_sub       number of subjects
/// @param nr_perm      number of permutations
/// @param d_max_t      device array of size nr_perm; updated via atomicMaxFloat
/// @param output_tstat output: observed t-statistics (written by thread 0 of each block)
/// @param two_tailed   reserved for future directional FWER variants (currently unused)
///
/// Each block processes one connection across all permutations.
/// d_max_t persists across kernel launches so it accumulates the global max
/// over multiple streaming chunks.
///
__global__
void CUDA_perm_fwer_pass1(float *input, int *onehot,
                           size_t nr_vals, size_t nr_sub, size_t nr_perm,
                           float *d_max_t, float *output_tstat, int two_tailed);

///
/// @brief FWER pass 2: compute FWER-corrected p-values from the max-|t| distribution.
/// @param input        subject data buffer, layout: [conn * nr_sub + sub]
/// @param onehot       one-hot permutation matrix, layout: [perm * nr_sub + sub]
/// @param nr_vals      number of connections in this chunk
/// @param nr_sub       number of subjects
/// @param nr_perm      number of permutations
/// @param d_max_t      finalised max-|t| null distribution from pass 1
/// @param output_pval  output: FWER-corrected p-values
/// @param output_tstat output: observed t-statistics
/// @param two_tailed   reserved (currently unused)
///
/// p-value = (# permutations where max_t >= |t_obs|) / nr_perm.
/// No +1 correction: perm 0 (observed) is already included in d_max_t.
///
__global__
void CUDA_perm_fwer_pass2(float *input, int *onehot,
                           size_t nr_vals, size_t nr_sub, size_t nr_perm,
                           float *d_max_t,
                           float *output_pval, float *output_tstat, int two_tailed);

#endif /* PERM_KERNELS_CUH */
