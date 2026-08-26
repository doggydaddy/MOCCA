/*
 * permutationTest_cuda_bundle.cu
 *
 * Separate sparse backend for atlas-free bundle-level permutation inference.
 * Existing permutation executables are intentionally untouched.
 *
 * For a requested batch of label permutations, this program streams every
 * connection, computes the same Welch statistic as perm_kernels.cu, and writes
 * only edges satisfying the requested cluster-forming rule.  This can be a
 * fixed |t| threshold or a two-sided p threshold converted to t with each
 * edge's Welch-Satterthwaite degrees of freedom.  A persistent
 * Python controller consumes these sparse files with bundle_fwer.py and keeps
 * one maximum bundle statistic per permutation.
 */

#include <cuda_runtime.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdexcept>
#include <vector>

#include <boost/math/distributions/students_t.hpp>

#include "ccmat_io.h"


#define BUNDLE_SPARSE_MAGIC 0x4C444E42u  /* "BNDL" little-endian */
#define BUNDLE_SPARSE_VERSION_FIXED 1u
#define BUNDLE_SPARSE_VERSION_DF_AWARE 2u
#define BUNDLE_SPARSE_FLAG_DF_AWARE 1u
#define T_LOOKUP_STEPS_PER_DF 4096u


#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t permutation_index;
    uint64_t n_records;
    uint64_t n_voxels;
    uint64_t n_total_edges;
    float threshold;
    uint32_t reserved;
} BundleSparseHeader;

typedef struct {
    uint64_t edge_index;
    float tstat;
} BundleSparseRecordV1;

typedef struct {
    uint64_t edge_index;
    float tstat;
    float excess;
} BundleSparseRecordV2;
#pragma pack(pop)


static_assert(sizeof(BundleSparseHeader) == 48, "unexpected sparse header size");
static_assert(sizeof(BundleSparseRecordV1) == 12, "unexpected v1 sparse record size");
static_assert(sizeof(BundleSparseRecordV2) == 16, "unexpected v2 sparse record size");


static void cuda_check(cudaError_t error, const char *operation)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR during %s: %s\n",
                operation, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}


typedef struct {
    float tstat;
    float degrees_of_freedom;
} BundleWelchResult;


__device__ static BundleWelchResult bundle_welch_result(
    const float *input, const int *onehot, int nr_sub)
{
    float a_sum = 0.f, b_sum = 0.f;
    float a_sq = 0.f, b_sq = 0.f;
    float n_a = 0.f, n_b = 0.f;

    for (int subject = 0; subject < nr_sub; ++subject) {
        float value = input[subject];
        if (onehot[subject] == 0) {
            b_sum += value;
            b_sq += value * value;
            n_b += 1.f;
        } else {
            a_sum += value;
            a_sq += value * value;
            n_a += 1.f;
        }
    }

    float a_mean = (n_a > 0.f) ? a_sum / n_a : 0.f;
    float b_mean = (n_b > 0.f) ? b_sum / n_b : 0.f;
    float a_var = (n_a > 1.f)
        ? (a_sq - a_sum * a_sum / n_a) / (n_a - 1.f) : 0.f;
    float b_var = (n_b > 1.f)
        ? (b_sq - b_sum * b_sum / n_b) / (n_b - 1.f) : 0.f;
    float standard_error = sqrtf(
        a_var / fmaxf(n_a, 1.f) + b_var / fmaxf(n_b, 1.f));

    BundleWelchResult result = {0.f, 1.f};
    if (standard_error <= 1e-12f)
        return result;

    result.tstat = (a_mean - b_mean) / standard_error;
    float a_term = a_var / n_a;
    float b_term = b_var / n_b;
    float denominator = a_term * a_term / (n_a - 1.f)
        + b_term * b_term / (n_b - 1.f);
    if (denominator > 1e-20f) {
        float numerator = a_term + b_term;
        result.degrees_of_freedom = numerator * numerator / denominator;
    }
    result.degrees_of_freedom = fminf(
        fmaxf(result.degrees_of_freedom, 1.f),
        fmaxf((float)nr_sub - 2.f, 1.f));
    return result;
}


__device__ static float interpolated_critical_t(
    float degrees_of_freedom,
    const float *critical_t,
    uint32_t table_size)
{
    float position = (degrees_of_freedom - 1.f) * T_LOOKUP_STEPS_PER_DF;
    position = fminf(fmaxf(position, 0.f), (float)(table_size - 1));
    uint32_t lower = (uint32_t)floorf(position);
    uint32_t upper = lower + 1 < table_size ? lower + 1 : lower;
    float fraction = position - (float)lower;
    return critical_t[lower]
        + fraction * (critical_t[upper] - critical_t[lower]);
}


__global__ static void threshold_permutation(
    const float *input,
    const int *labels,
    size_t n_edges,
    int n_subjects,
    float threshold,
    const float *critical_t,
    uint32_t critical_t_count,
    int df_aware,
    uint64_t global_edge_start,
    uint64_t capacity,
    uint64_t *output_indices,
    float *output_tstats,
    float *output_excess,
    unsigned long long *output_count,
    int *overflow)
{
    uint64_t edge = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= n_edges)
        return;

    BundleWelchResult welch = bundle_welch_result(
        input + edge * (uint64_t)n_subjects, labels, n_subjects);
    float edge_threshold = df_aware
        ? interpolated_critical_t(
            welch.degrees_of_freedom, critical_t, critical_t_count)
        : threshold;
    float excess = fabsf(welch.tstat) - edge_threshold;
    if (!isfinite(welch.tstat) || !isfinite(excess) || excess < 0.f)
        return;

    unsigned long long position = atomicAdd(output_count, 1ULL);
    if (position < capacity) {
        output_indices[position] = global_edge_start + edge;
        output_tstats[position] = welch.tstat;
        output_excess[position] = excess;
    } else {
        atomicExch(overflow, 1);
    }
}


static void write_header(
    FILE *stream,
    uint64_t permutation_index,
    uint64_t n_records,
    uint64_t n_voxels,
    uint64_t n_total_edges,
    float cluster_forming_value,
    bool df_aware)
{
    BundleSparseHeader header = {
        BUNDLE_SPARSE_MAGIC,
        df_aware ? BUNDLE_SPARSE_VERSION_DF_AWARE
                 : BUNDLE_SPARSE_VERSION_FIXED,
        permutation_index,
        n_records,
        n_voxels,
        n_total_edges,
        cluster_forming_value,
        df_aware ? BUNDLE_SPARSE_FLAG_DF_AWARE : 0u,
    };
    if (fseek(stream, 0, SEEK_SET) != 0 ||
            fwrite(&header, sizeof(header), 1, stream) != 1) {
        perror("writing sparse header");
        exit(EXIT_FAILURE);
    }
}


static char *sparse_path(const char *prefix, size_t permutation_index)
{
    size_t length = strlen(prefix) + 40;
    char *path = (char *)malloc(length);
    if (!path) {
        fprintf(stderr, "Out of memory creating sparse output path.\n");
        exit(EXIT_FAILURE);
    }
    snprintf(path, length, "%s_perm%06zu.bsp", prefix, permutation_index);
    return path;
}


static size_t parse_size(const char *text, const char *option)
{
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0') {
        fprintf(stderr, "Invalid %s value: %s\n", option, text);
        exit(EXIT_FAILURE);
    }
    return (size_t)value;
}


static float parse_probability(const char *text, const char *option)
{
    errno = 0;
    char *end = NULL;
    float value = strtof(text, &end);
    if (errno || end == text || *end != '\0' || !(value > 0.f && value < 1.f)) {
        fprintf(stderr, "Invalid %s probability: %s\n", option, text);
        exit(EXIT_FAILURE);
    }
    return value;
}


static std::vector<float> critical_t_lookup(size_t n_subjects, double two_sided_p)
{
    if (n_subjects < 4)
        throw std::runtime_error("Welch testing requires at least four subjects.");
    uint32_t maximum_df = (uint32_t)n_subjects - 2u;
    size_t count = (size_t)(maximum_df - 1u) * T_LOOKUP_STEPS_PER_DF + 1u;
    std::vector<float> lookup(count);
    for (size_t index = 0; index < count; ++index) {
        double df = 1.0 + (double)index / T_LOOKUP_STEPS_PER_DF;
        boost::math::students_t_distribution<double> distribution(df);
        lookup[index] = (float)boost::math::quantile(
            boost::math::complement(distribution, two_sided_p / 2.0));
    }
    return lookup;
}


int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <filelist> <permutations> <output_prefix> <|t| threshold> "
            "[--cluster-forming-p P] [--start-perm N] [--count N] "
            "[--capacity N]\n"
            "Use threshold 0 with --cluster-forming-p for df-aware Welch t.\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const char *filelist = argv[1];
    const char *permutation_file = argv[2];
    const char *output_prefix = argv[3];
    float threshold = strtof(argv[4], NULL);
    float cluster_forming_p = 0.f;
    bool df_aware = false;
    size_t start_permutation = 0;
    size_t requested_count = 1;
    size_t capacity = 10000000;

    for (int arg = 5; arg < argc; ++arg) {
        if (strcmp(argv[arg], "--cluster-forming-p") == 0 && arg + 1 < argc) {
            cluster_forming_p = parse_probability(
                argv[++arg], "--cluster-forming-p");
            df_aware = true;
        } else if (strcmp(argv[arg], "--start-perm") == 0 && arg + 1 < argc) {
            start_permutation = parse_size(argv[++arg], "--start-perm");
        } else if (strcmp(argv[arg], "--count") == 0 && arg + 1 < argc) {
            requested_count = parse_size(argv[++arg], "--count");
        } else if (strcmp(argv[arg], "--capacity") == 0 && arg + 1 < argc) {
            capacity = parse_size(argv[++arg], "--capacity");
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[arg]);
            return EXIT_FAILURE;
        }
    }
    if (df_aware) {
        if (threshold != 0.f) {
            fprintf(stderr,
                "Use positional threshold 0 with --cluster-forming-p.\n");
            return EXIT_FAILURE;
        }
    } else if (!(threshold > 0.f) || !isfinite(threshold)) {
        fprintf(stderr, "The cluster-forming |t| threshold must be finite and > 0.\n");
        return EXIT_FAILURE;
    }
    if (requested_count == 0 || capacity == 0) {
        fprintf(stderr, "--count and --capacity must be positive.\n");
        return EXIT_FAILURE;
    }

    size_t n_voxels = peekFileList((char *)filelist);
    size_t n_total_edges = n_voxels * (n_voxels - 1) / 2;
    size_t n_subjects = getNumberLines((char *)filelist);
    size_t n_permutations = getNumberLines((char *)permutation_file);
    if (start_permutation >= n_permutations ||
            requested_count > n_permutations - start_permutation) {
        fprintf(stderr,
            "Requested permutations [%zu, %zu) exceed the %zu-row file.\n",
            start_permutation, start_permutation + requested_count,
            n_permutations);
        return EXIT_FAILURE;
    }

    printf("========================================\n");
    printf("Sparse bundle-permutation CUDA backend\n");
    printf("  subjects       : %zu\n", n_subjects);
    printf("  voxels         : %zu\n", n_voxels);
    printf("  edges          : %zu\n", n_total_edges);
    printf("  permutation rows: %zu\n", n_permutations);
    printf("  requested      : [%zu, %zu)\n",
           start_permutation, start_permutation + requested_count);
    if (df_aware)
        printf("  cluster p      : %.8g (two-sided, Welch df-aware)\n",
               cluster_forming_p);
    else
        printf("  |t| threshold  : %.6f\n", threshold);
    printf("  capacity/part  : %zu sparse edges\n", capacity);
    printf("========================================\n");
    fflush(stdout);

    size_t permutation_values = n_permutations * n_subjects;
    int *host_permutations = (int *)calloc(permutation_values, sizeof(int));
    if (!host_permutations) {
        fprintf(stderr, "Unable to allocate host permutation buffer.\n");
        return EXIT_FAILURE;
    }
    parsePermutations((char *)permutation_file, host_permutations, n_subjects);

    int *device_permutations = NULL;
    uint64_t *device_indices = NULL;
    float *device_tstats = NULL;
    float *device_excess = NULL;
    float *device_critical_t = NULL;
    unsigned long long *device_count = NULL;
    int *device_overflow = NULL;
    cuda_check(cudaMalloc((void **)&device_permutations,
                          permutation_values * sizeof(int)),
               "allocating permutation labels");
    cuda_check(cudaMemcpy(device_permutations, host_permutations,
                          permutation_values * sizeof(int),
                          cudaMemcpyHostToDevice),
               "uploading permutation labels");
    free(host_permutations);

    cuda_check(cudaMalloc((void **)&device_indices,
                          capacity * sizeof(uint64_t)),
               "allocating sparse indices");
    cuda_check(cudaMalloc((void **)&device_tstats,
                          capacity * sizeof(float)),
               "allocating sparse t-statistics");
    cuda_check(cudaMalloc((void **)&device_excess,
                          capacity * sizeof(float)),
               "allocating sparse threshold excess");

    std::vector<float> host_critical_t;
    if (df_aware) {
        host_critical_t = critical_t_lookup(n_subjects, cluster_forming_p);
        cuda_check(cudaMalloc((void **)&device_critical_t,
                              host_critical_t.size() * sizeof(float)),
                   "allocating critical-t lookup");
        cuda_check(cudaMemcpy(device_critical_t, host_critical_t.data(),
                              host_critical_t.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "uploading critical-t lookup");
    }
    cuda_check(cudaMalloc((void **)&device_count,
                          sizeof(unsigned long long)),
               "allocating sparse counter");
    cuda_check(cudaMalloc((void **)&device_overflow, sizeof(int)),
               "allocating overflow flag");

    size_t free_memory = 0, total_memory = 0;
    cuda_check(cudaMemGetInfo(&free_memory, &total_memory),
               "querying GPU memory");
    size_t safety_memory = free_memory / 10;
    size_t available_for_input = free_memory > safety_memory
        ? free_memory - safety_memory : free_memory;
    size_t max_edges_per_part = available_for_input /
        (n_subjects * sizeof(float));
    if (max_edges_per_part == 0) {
        fprintf(stderr, "Insufficient GPU memory for even one connection.\n");
        return EXIT_FAILURE;
    }
    if (max_edges_per_part > n_total_edges)
        max_edges_per_part = n_total_edges;
    size_t n_parts = (n_total_edges + max_edges_per_part - 1) /
        max_edges_per_part;
    printf("  GPU memory     : %.2f / %.2f GiB free\n",
           free_memory / 1073741824.0, total_memory / 1073741824.0);
    printf("  edges/part     : %zu (%zu parts)\n",
           max_edges_per_part, n_parts);
    fflush(stdout);

    std::vector<FILE *> output_streams(requested_count, NULL);
    std::vector<char *> output_paths(requested_count, NULL);
    std::vector<uint64_t> record_counts(requested_count, 0);
    for (size_t local = 0; local < requested_count; ++local) {
        size_t permutation_index = start_permutation + local;
        output_paths[local] = sparse_path(output_prefix, permutation_index);
        output_streams[local] = fopen(output_paths[local], "wb+");
        if (!output_streams[local]) {
            fprintf(stderr, "Cannot open %s: %s\n",
                    output_paths[local], strerror(errno));
            return EXIT_FAILURE;
        }
        write_header(output_streams[local], permutation_index, 0,
                     n_voxels, n_total_edges,
                     df_aware ? cluster_forming_p : threshold, df_aware);
        if (fseek(output_streams[local], 0, SEEK_END) != 0) {
            perror("seeking sparse output");
            return EXIT_FAILURE;
        }
    }

    FileHandleArray *open_files = openAllSubjectFiles(
        (char *)filelist, n_subjects);
    const int threads = 256;

    for (size_t part = 0; part < n_parts; ++part) {
        size_t part_start = part * max_edges_per_part;
        size_t part_count = max_edges_per_part;
        if (part_count > n_total_edges - part_start)
            part_count = n_total_edges - part_start;
        size_t part_end = part_start + part_count - 1;

        printf("[part %zu/%zu] loading edges [%zu, %zu]\n",
               part + 1, n_parts, part_start, part_end);
        fflush(stdout);
        float *host_input = (float *)malloc(
            part_count * n_subjects * sizeof(float));
        if (!host_input) {
            fprintf(stderr, "Unable to allocate host input part.\n");
            return EXIT_FAILURE;
        }
        readRowsFromOpenFiles(open_files, part_start, part_end,
                              (int)n_subjects, host_input);

        float *device_input = NULL;
        cuda_check(cudaMalloc((void **)&device_input,
                              part_count * n_subjects * sizeof(float)),
                   "allocating input part");
        cuda_check(cudaMemcpy(device_input, host_input,
                              part_count * n_subjects * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "uploading input part");
        free(host_input);

        for (size_t local = 0; local < requested_count; ++local) {
            size_t permutation_index = start_permutation + local;
            cuda_check(cudaMemset(device_count, 0,
                                  sizeof(unsigned long long)),
                       "resetting sparse counter");
            cuda_check(cudaMemset(device_overflow, 0, sizeof(int)),
                       "resetting overflow flag");

            size_t blocks = (part_count + threads - 1) / threads;
            threshold_permutation<<<blocks, threads>>>(
                device_input,
                device_permutations + permutation_index * n_subjects,
                part_count,
                (int)n_subjects,
                threshold,
                device_critical_t,
                (uint32_t)host_critical_t.size(),
                df_aware ? 1 : 0,
                part_start,
                capacity,
                device_indices,
                device_tstats,
                device_excess,
                device_count,
                device_overflow);
            cuda_check(cudaGetLastError(), "launching threshold kernel");
            cuda_check(cudaDeviceSynchronize(), "running threshold kernel");

            unsigned long long selected = 0;
            int overflow = 0;
            cuda_check(cudaMemcpy(&selected, device_count,
                                  sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost),
                       "downloading sparse count");
            cuda_check(cudaMemcpy(&overflow, device_overflow, sizeof(int),
                                  cudaMemcpyDeviceToHost),
                       "downloading overflow flag");
            if (overflow || selected > capacity) {
                fprintf(stderr,
                    "Sparse capacity exceeded for permutation %zu, part %zu: "
                    "%llu > %zu. Rerun with --capacity at least %llu.\n",
                    permutation_index, part + 1, selected, capacity, selected);
                return EXIT_FAILURE;
            }
            if (selected == 0)
                continue;

            std::vector<uint64_t> host_indices((size_t)selected);
            std::vector<float> host_tstats((size_t)selected);
            std::vector<float> host_excess((size_t)selected);
            cuda_check(cudaMemcpy(host_indices.data(), device_indices,
                                  selected * sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost),
                       "downloading sparse indices");
            cuda_check(cudaMemcpy(host_tstats.data(), device_tstats,
                                  selected * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "downloading sparse t-statistics");
            if (df_aware) {
                cuda_check(cudaMemcpy(host_excess.data(), device_excess,
                                      selected * sizeof(float),
                                      cudaMemcpyDeviceToHost),
                           "downloading sparse threshold excess");
            }

            if (df_aware) {
                std::vector<BundleSparseRecordV2> records((size_t)selected);
                for (size_t index = 0; index < (size_t)selected; ++index) {
                    records[index].edge_index = host_indices[index];
                    records[index].tstat = host_tstats[index];
                    records[index].excess = host_excess[index];
                }
                if (fwrite(records.data(), sizeof(BundleSparseRecordV2),
                           records.size(), output_streams[local]) != records.size()) {
                    perror("writing df-aware sparse records");
                    return EXIT_FAILURE;
                }
            } else {
                std::vector<BundleSparseRecordV1> records((size_t)selected);
                for (size_t index = 0; index < (size_t)selected; ++index) {
                    records[index].edge_index = host_indices[index];
                    records[index].tstat = host_tstats[index];
                }
                if (fwrite(records.data(), sizeof(BundleSparseRecordV1),
                           records.size(), output_streams[local]) != records.size()) {
                    perror("writing fixed-threshold sparse records");
                    return EXIT_FAILURE;
                }
            }
            record_counts[local] += (uint64_t)selected;
        }

        cudaFree(device_input);
    }

    closeAllSubjectFiles(open_files);
    for (size_t local = 0; local < requested_count; ++local) {
        size_t permutation_index = start_permutation + local;
        write_header(output_streams[local], permutation_index,
                     record_counts[local], n_voxels, n_total_edges,
                     df_aware ? cluster_forming_p : threshold, df_aware);
        fclose(output_streams[local]);
        printf("[permutation %zu] wrote %llu sparse edges to %s\n",
               permutation_index,
               (unsigned long long)record_counts[local],
               output_paths[local]);
        free(output_paths[local]);
    }

    cudaFree(device_permutations);
    cudaFree(device_indices);
    cudaFree(device_tstats);
    cudaFree(device_excess);
    cudaFree(device_critical_t);
    cudaFree(device_count);
    cudaFree(device_overflow);
    printf("Sparse bundle-permutation batch complete.\n");
    return EXIT_SUCCESS;
}
