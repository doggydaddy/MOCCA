/*
 * perm_kernels.cu  —  CUDA kernel implementations for permutation testing
 *
 * All three kernels share the same inner loop structure:
 *   1. Thread 0 computes the observed Welch's t-statistic (perm 0).
 *   2. All threads stripe across remaining permutations.
 *   3. Results are reduced in shared memory by thread 0.
 *
 * The __device__ helper welch_tstat() eliminates the repeated inline
 * t-statistic calculation that was previously copy-pasted five times.
 */
#include "perm_kernels.cuh"

#include <math.h>

/* ── Private device helper ──────────────────────────────────────────────────── */

///
/// @brief Computes Welch's two-sample t-statistic for one connection × one permutation.
/// @param input    subject data for this connection: input[j] = value for subject j
/// @param onehot   one-hot group labels for this permutation: onehot[j] ∈ {0,1}
/// @param nr_sub   number of subjects
/// @return t = (mean_A - mean_B) / sqrt(var_A/nA + var_B/nB), or 0 if SE < 1e-12
///
__device__
static float welch_tstat(const float* __restrict__ input,
                          const int*   __restrict__ onehot,
                          int nr_sub)
{
    float a_sum = 0.f, b_sum = 0.f;
    float a_sq  = 0.f, b_sq  = 0.f;
    float nA    = 0.f, nB    = 0.f;

    for (int j = 0; j < nr_sub; ++j)
    {
        float val = input[j];
        if (onehot[j] == 0) { b_sum += val; b_sq += val * val; nB++; }
        else                { a_sum += val; a_sq += val * val; nA++; }
    }

    float a_mean = (nA > 0.f) ? a_sum / nA : 0.f;
    float b_mean = (nB > 0.f) ? b_sum / nB : 0.f;
    float a_var  = (nA > 1.f) ? (a_sq - a_sum * a_sum / nA) / (nA - 1.f) : 0.f;
    float b_var  = (nB > 1.f) ? (b_sq - b_sum * b_sum / nB) / (nB - 1.f) : 0.f;
    float se     = sqrtf(a_var / fmaxf(nA, 1.f) + b_var / fmaxf(nB, 1.f));

    return (se > 1e-12f) ? (a_mean - b_mean) / se : 0.f;
}

/* ── atomicMaxFloat ─────────────────────────────────────────────────────────── */

__device__ float atomicMaxFloat(float* addr, float value)
{
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/* ── CUDA_perm ──────────────────────────────────────────────────────────────── */

__global__
void CUDA_perm(float *input, int *onehot,
               size_t nr_vals, size_t nr_sub, size_t nr_perm,
               float *output_pval, float *output_tstat, int two_tailed)
{
    int n          = blockIdx.x;
    int tid        = threadIdx.x;
    int block_size = blockDim.x;

    if (n >= (int)nr_vals) return;

    extern __shared__ float shared_pval[];

    /* Pointers to this connection's row in the input and observed-perm onehot. */
    const float* conn_data = input + (size_t)n * nr_sub;
    const int*   obs_label = onehot;   /* perm 0 */

    /* Thread 0: observed t-statistic. */
    float t_obs = 0.f;
    if (tid == 0)
        t_obs = welch_tstat(conn_data, obs_label, nr_sub);

    /* Broadcast t_obs. */
    if (tid == 0) shared_pval[0] = t_obs;
    __syncthreads();
    t_obs = shared_pval[0];

    /* Each thread counts extreme permutations in its stripe. */
    float local_pval = 0.f;
    for (int i = 1 + tid; i < (int)nr_perm; i += block_size)
    {
        float tstat = welch_tstat(conn_data, onehot + (size_t)i * nr_sub, nr_sub);

        if (two_tailed)
        {
            if (fabsf(tstat) >= fabsf(t_obs)) local_pval++;
        }
        else
        {
            /* One-tailed: test in the direction of the observed effect.
             *   t_obs >= 0  →  upper tail  (count tstat >= t_obs)
             *   t_obs <  0  →  lower tail  (count tstat <= t_obs) */
            if (t_obs >= 0.f) { if (tstat >= t_obs) local_pval++; }
            else              { if (tstat <= t_obs) local_pval++; }
        }
    }

    /* Reduce in shared memory; thread 0 writes the final p-value. */
    shared_pval[tid] = local_pval;
    __syncthreads();

    if (tid == 0)
    {
        float total = 0.f;
        for (int i = 0; i < block_size; i++) total += shared_pval[i];
        /* Standard +1 correction: observed statistic counts as one permutation. */
        output_pval[n]  = (total + 1.f) / (float)(nr_perm + 1);
        output_tstat[n] = t_obs;
    }
}

/* ── CUDA_perm_fwer_pass1 ───────────────────────────────────────────────────── */

__global__
void CUDA_perm_fwer_pass1(float *input, int *onehot,
                           size_t nr_vals, size_t nr_sub, size_t nr_perm,
                           float *d_max_t, float *output_tstat, int two_tailed)
{
    int n          = blockIdx.x;
    int tid        = threadIdx.x;
    int block_size = blockDim.x;

    if (n >= (int)nr_vals) return;

    extern __shared__ float smem[];

    const float* conn_data = input + (size_t)n * nr_sub;

    /* Thread 0: observed t-statistic (perm 0); update max for perm 0. */
    float t_obs = 0.f;
    if (tid == 0)
    {
        t_obs = welch_tstat(conn_data, onehot, nr_sub);
        output_tstat[n] = t_obs;
        atomicMaxFloat(&d_max_t[0], fabsf(t_obs));
    }

    /* Broadcast t_obs (available for directional future extensions). */
    if (tid == 0) smem[0] = t_obs;
    __syncthreads();
    t_obs = smem[0];
    (void)t_obs; /* suppress unused-variable warning when two_tailed path unused */

    /* Stripe across permutations 1 .. nr_perm-1; update global max per perm. */
    for (int i = 1 + tid; i < (int)nr_perm; i += block_size)
    {
        float tstat = welch_tstat(conn_data, onehot + (size_t)i * nr_sub, nr_sub);
        /* FWER max-statistic null distribution is always built from |t|. */
        atomicMaxFloat(&d_max_t[i], fabsf(tstat));
    }
}

/* ── CUDA_perm_fwer_pass2 ───────────────────────────────────────────────────── */

__global__
void CUDA_perm_fwer_pass2(float *input, int *onehot,
                           size_t nr_vals, size_t nr_sub, size_t nr_perm,
                           float *d_max_t,
                           float *output_pval, float *output_tstat, int two_tailed)
{
    int n          = blockIdx.x;
    int tid        = threadIdx.x;
    int block_size = blockDim.x;

    if (n >= (int)nr_vals) return;

    extern __shared__ float smem[];

    const float* conn_data = input + (size_t)n * nr_sub;

    /* Thread 0: re-compute observed t_obs and record it. */
    float t_obs = 0.f;
    if (tid == 0)
    {
        t_obs = welch_tstat(conn_data, onehot, nr_sub);
        output_tstat[n] = t_obs;
    }

    /* Broadcast |t_obs| to all threads. */
    if (tid == 0) smem[0] = t_obs;
    __syncthreads();
    float abs_t_obs = fabsf(smem[0]);

    /* Each thread counts how many null max_t values are >= |t_obs|. */
    float local_count = 0.f;
    for (int i = tid; i < (int)nr_perm; i += block_size)
        if (d_max_t[i] >= abs_t_obs) local_count += 1.f;

    /* Reduce in shared memory; thread 0 computes the final p-value. */
    smem[tid] = local_count;
    __syncthreads();

    if (tid == 0)
    {
        float total = 0.f;
        for (int i = 0; i < block_size; i++) total += smem[i];
        /* FWER p-value: no +1 correction — perm 0 is already in d_max_t. */
        output_pval[n] = total / (float)nr_perm;
    }
}
