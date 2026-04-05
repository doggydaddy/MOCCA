#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "ccmat_io.h"
#include "results_io.h"
#include "perm_kernels.cuh"

/* Write a permout binary header to both the p-value and t-stat output files. */
static void write_permout_headers(FILE *out_pval, FILE *out_tstat,
                                  size_t gV, size_t n_elem)
{
    uint32_t magic   = PERMOUT_MAGIC;
    uint32_t version = CCMAT_VERSION;
    uint64_t gV64    = (uint64_t)gV;
    uint64_t ne64    = (uint64_t)n_elem;
    FILE *fps[2] = { out_pval, out_tstat };
    for (int f = 0; f < 2; f++) {
        fwrite(&magic,   sizeof(uint32_t), 1, fps[f]);
        fwrite(&version, sizeof(uint32_t), 1, fps[f]);
        fwrite(&gV64,    sizeof(uint64_t), 1, fps[f]);
        fwrite(&ne64,    sizeof(uint64_t), 1, fps[f]);
    }
}

int
main(int argc, char *argv[])
{
    /* read arguments */
    if (argc < 4 || argc > 7) {
        fprintf(stderr, "Usage: %s <file list> <permutations file> <output file> [--two-tailed] [-b] [--fwer]\n", argv[0]);
        fprintf(stderr, "  --two-tailed : Enable two-tailed test (default: one-tailed)\n");
        fprintf(stderr, "  -b           : Write output in binary format instead of text\n");
        fprintf(stderr, "  --fwer       : Enable FWER max-statistic correction (two-pass)\n");
        exit(EXIT_FAILURE);
    }

    /* parse input arguments */
    char* filelist = argv[1];
    char* permutations = argv[2];
    char* outfile = argv[3];

    /* parse optional flags */
    int two_tailed = 0;
    int binary_output = 0;
    int fwer_mode = 0;
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--two-tailed") == 0) {
            two_tailed = 1;
        } else if (strcmp(argv[i], "-b") == 0) {
            binary_output = 1;
        } else if (strcmp(argv[i], "--fwer") == 0) {
            fwer_mode = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s <file list> <permutations file> <output file> [--two-tailed] [-b] [--fwer]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    printf("========================================\n");
    printf("permutationTest_cuda%s\n", fwer_mode ? " (FWER max-statistic)" : "");
    printf("========================================\n");
    printf("  File list    : %s\n", filelist);
    printf("  Permutations : %s\n", permutations);
    printf("  Output       : %s\n", outfile);
    printf("  Test type    : %s\n", two_tailed ? "two-tailed" : "one-tailed");
    printf("  Output format: %s\n", binary_output ? "binary (-b)" : "text");
    printf("  FWER mode    : %s\n", fwer_mode ? "YES (two-pass max-statistic)" : "no");
    printf("========================================\n");
    fflush(stdout);

    // Create t-stat output filename by inserting "_tstat" before the extension
    char* outfile_tstat = (char*)malloc(strlen(outfile) + 20);
    char* dot = strrchr(outfile, '.');
    if (dot != NULL) {
        size_t prefix_len = dot - outfile;
        strncpy(outfile_tstat, outfile, prefix_len);
        outfile_tstat[prefix_len] = '\0';
        strcat(outfile_tstat, "_tstat");
        strcat(outfile_tstat, dot);
    } else {
        strcpy(outfile_tstat, outfile);
        strcat(outfile_tstat, "_tstat");
    }
    printf("  Output tstat : %s\n", outfile_tstat);
    fflush(stdout);

    /* get dimensions */
    printf("Getting dimensions from input files...\n");
    fflush(stdout);
    ssize_t nr_r1vals = peekFileList(filelist);
    printf("Dimensions determined: %zd voxels per subject\n", nr_r1vals);
    fflush(stdout);
    ssize_t nr_vals = (nr_r1vals*(nr_r1vals-1))/2;
    printf("Counting permutations...\n");
    fflush(stdout);
    size_t nr_perm = getNumberLines(permutations);
    printf("Counting subjects...\n");
    fflush(stdout);
    size_t nr_subs = getNumberLines(filelist);
    printf("Number of permutations: %zd\n", nr_perm);
    printf("Number of subjects: %zu\n", nr_subs);
    printf("Number of connections in each subject: %zu\n", nr_vals);
    fflush(stdout);

    /* calculate how many voxels values we can test at once with our available memory */
    size_t device_free_mem, device_total_mem;
	cudaMemGetInfo(&device_free_mem, &device_total_mem);
	printf("GPU free mem: %lu, Total mem: %lu\n", device_free_mem, device_total_mem);
    fflush(stdout);
    // Memory needed per value: input (nr_subs floats) + output_pval (1 float) + output_tstat (1 float) = (nr_subs+2) floats
    size_t nr_vals_max = 0.9*((device_free_mem-(sizeof(int)*nr_perm*nr_subs))/(sizeof(float)*(nr_subs+2)));
    printf("Number of values we can load at once into device memory: %zu\n", nr_vals_max);
    fflush(stdout);

    if (nr_vals <= nr_vals_max) 
    {
        printf("can load the entire thing in one go!\n");
        fflush(stdout);

        printf("allocating memory buffers ...\n");
        fflush(stdout);
        int* perm_buff = (int*)malloc(sizeof(int)*nr_perm*nr_subs);
        /* we have to zero out the permutation buffer */
        for (int i=0; i<nr_perm*nr_subs; ++i)
        {
            perm_buff[i] = 0; 
        }
        float* device_buff = (float*)malloc(sizeof(float)*nr_vals*nr_subs);
        
        printf("parsing input files ...\n");
        fflush(stdout);
        parsePermutations(permutations, perm_buff, nr_subs);
        parseFileListNtoM(filelist, nr_subs, 0, nr_vals-1, device_buff);

        float* perm_test_res;
        float* perm_test_tstat;
        perm_test_res = (float*)malloc(sizeof(float)*nr_vals);
        perm_test_tstat = (float*)malloc(sizeof(float)*nr_vals);

        printf("performing permutation tests ...\n");
        fflush(stdout);
        ///* allocating gpu mem */
        int *d_perm;
        float *d_input;
        float *d_output_pval;
        float *d_output_tstat;
        cudaError_t err = cudaSuccess;;
        err = cudaMalloc((void **)&d_perm, sizeof(int)*nr_perm*nr_subs);
        err = cudaMalloc((void **)&d_input, sizeof(float)*nr_vals*nr_subs);
        err = cudaMalloc((void **)&d_output_pval, sizeof(float)*nr_vals );
        err = cudaMalloc((void **)&d_output_tstat, sizeof(float)*nr_vals );
        if (err!=cudaSuccess)
        {
            printf("CUDA ERROR! Failed to allocate device memory! (error code %s)\n", cudaGetErrorString(err));
        }
        /* copy input data host -> device */
        err = cudaMemcpy(d_input, device_buff, sizeof(float)*nr_vals*nr_subs, cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_perm, perm_buff, sizeof(int)*nr_perm*nr_subs, cudaMemcpyHostToDevice);
        if (err!=cudaSuccess)
        {
            printf("CUDA ERROR! Failed to copy memory from host to device (error code %s)\n", cudaGetErrorString(err));
        }

        int threads_per_block = 256;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        if (fwer_mode)
        {
            /* ── FWER: single-chunk path (both passes, data fits in GPU) ── */
            printf("[FWER] All data fits in GPU memory — running both passes without streaming.\n");
            fflush(stdout);

            /* Allocate max_t array on GPU (one float per permutation, init to 0) */
            float *d_max_t;
            err = cudaMalloc((void **)&d_max_t, sizeof(float) * nr_perm);
            if (err != cudaSuccess) {
                printf("CUDA ERROR! Failed to allocate d_max_t (error code %s)\n", cudaGetErrorString(err));
            }
            err = cudaMemset(d_max_t, 0, sizeof(float) * nr_perm);

            /* Pass 1: compute max |t| per permutation across all connections */
            printf("[FWER PASS 1] Computing max |t| per permutation ...\n");
            printf("  Launching %zu blocks x %d threads\n", nr_vals, threads_per_block);
            fflush(stdout);
            CUDA_perm_fwer_pass1<<<nr_vals, threads_per_block, shared_mem_size>>>(
                d_input, d_perm, nr_vals, nr_subs, nr_perm,
                d_max_t, d_output_tstat, two_tailed);
            err = cudaGetLastError();
            if (err != cudaSuccess) printf("CUDA ERROR! Pass 1 kernel launch failed (%s)\n", cudaGetErrorString(err));
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) printf("CUDA ERROR! Pass 1 kernel execution failed (%s)\n", cudaGetErrorString(err));
            printf("[FWER PASS 1] Complete.\n");
            fflush(stdout);

            /* Pass 2: compute FWER p-values */
            printf("[FWER PASS 2] Computing FWER-corrected p-values ...\n");
            printf("  Launching %zu blocks x %d threads\n", nr_vals, threads_per_block);
            fflush(stdout);
            CUDA_perm_fwer_pass2<<<nr_vals, threads_per_block, shared_mem_size>>>(
                d_input, d_perm, nr_vals, nr_subs, nr_perm,
                d_max_t, d_output_pval, d_output_tstat, two_tailed);
            err = cudaGetLastError();
            if (err != cudaSuccess) printf("CUDA ERROR! Pass 2 kernel launch failed (%s)\n", cudaGetErrorString(err));
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) printf("CUDA ERROR! Pass 2 kernel execution failed (%s)\n", cudaGetErrorString(err));
            printf("[FWER PASS 2] Complete.\n");
            fflush(stdout);

            cudaFree(d_max_t);
        }
        else
        {
            /* ── Standard per-connection p-values (original path) ── */
            printf("Launching kernel with %zu blocks, %d threads per block...\n", nr_vals, threads_per_block);
            fflush(stdout);
            CUDA_perm<<<nr_vals, threads_per_block, shared_mem_size>>>(d_input, d_perm, nr_vals, nr_subs, nr_perm, d_output_pval, d_output_tstat, two_tailed);
            
            // Check for kernel launch errors
            err = cudaGetLastError();
            if (err != cudaSuccess) 
            {
                printf("CUDA ERROR! Kernel launch failed (error code %s)\n", cudaGetErrorString(err));
            }
            
            // Wait for GPU to finish
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) 
            {
                printf("CUDA ERROR! Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
            }
        }
        
        /* copy results from the device back to host */
        err = cudaMemcpy(perm_test_res, d_output_pval, sizeof(float)*nr_vals, cudaMemcpyDeviceToHost);
        err = cudaMemcpy(perm_test_tstat, d_output_tstat, sizeof(float)*nr_vals, cudaMemcpyDeviceToHost);
        if (err!=cudaSuccess)
        {
            printf("CUDA ERROR! Failed to copy memory from device to host (error code %s)\n", cudaGetErrorString(err));
        }

        printf("writing to file ...\n");
        fflush(stdout);
        if (binary_output) {
            saveResToBinary(perm_test_res,   nr_r1vals, outfile);
            saveResToBinary(perm_test_tstat, nr_r1vals, outfile_tstat);
        } else {
            saveResToText(perm_test_res,   nr_r1vals, outfile);
            saveResToText(perm_test_tstat, nr_r1vals, outfile_tstat);
        }
        printf("done!\n");
        fflush(stdout);

        /* cleanup */
        free(device_buff);
        free(perm_buff);
        free(perm_test_res);
        free(perm_test_tstat);

        err = cudaFree(d_input);
        err = cudaFree(d_output_pval);
        err = cudaFree(d_output_tstat);
        err = cudaFree(d_perm);
        if (err!=cudaSuccess)
        {
            printf("CUDA ERROR! Failed to free device memory! (error code %s)\n", cudaGetErrorString(err));
        }
    }
    else 
    {
        printf("we cannot load the entire thing in one go ...\n");
        fflush(stdout);
        /* how many parts do we have to split stuff in? */
        int nr_parts = ceil(nr_vals/nr_vals_max);
        printf("... so we have to split the job into %i parts\n", nr_parts);
        fflush(stdout);

        /* calculate indices and sizes */
        size_t part_starts[nr_parts];
        size_t part_ends[nr_parts];
        size_t part_vals[nr_parts];
        for (int p=0; p<nr_parts; ++p)
        {
            if (p==0)
            {
                part_vals[p] = int(nr_vals/nr_parts);
                part_starts[p] = 0;
                part_ends[p] = part_vals[0]-1;
            }
            else 
            {
                part_starts[p] = part_ends[p-1]+1;
                part_vals[p] = part_vals[p-1];
                part_ends[p] = part_starts[p]+part_vals[p]-1;
            }    
        }
        int diffset;
        if (part_ends[nr_parts-1] < nr_vals-1) 
        {   /* correcting up */
            diffset = (nr_vals-1) - part_ends[nr_parts-1];
            part_ends[nr_parts-1] = nr_vals-1;
            part_vals[nr_parts-1] += diffset;
        } 
        else if (part_ends[nr_parts-1] > nr_vals-1) 
        {   /* correcting down */
            diffset = part_ends[nr_parts-1] - (nr_vals-1);
            part_ends[nr_parts-1] = nr_vals-1;
            part_vals[nr_parts-1] -= diffset;
        }

        /* dbg printout part divisions */ 
        printf("[DBG] part division indices:\n");
        fflush(stdout);
        for (int i=0; i<nr_parts; ++i) 
        {
            printf("[%zu, %zu] (%zu)\n", part_starts[i], part_ends[i], part_vals[i]);
        }
        fflush(stdout);

        /* ── Allocate and parse permutations (once, shared by both passes) ── */
        printf("Allocating and parsing permutations buffer (%zu MB)...\n", 
               (sizeof(int)*nr_perm*nr_subs)/(1024*1024));
        fflush(stdout);
        int* perm_buff = (int*)malloc(sizeof(int)*nr_perm*nr_subs);
        for (int i=0; i<nr_perm*nr_subs; ++i) perm_buff[i] = 0; 
        parsePermutations(permutations, perm_buff, nr_subs);
        printf("Permutations parsed successfully!\n");
        fflush(stdout);

        /* ── Shared device buffers for permutation matrix (same every part) ── */
        int *d_perm;
        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void **)&d_perm, sizeof(int)*nr_perm*nr_subs);
        if (err != cudaSuccess) {
            printf("CUDA ERROR! Failed to allocate d_perm (%s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(d_perm, perm_buff, sizeof(int)*nr_perm*nr_subs, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA ERROR! Failed to copy perm_buff to device (%s)\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        int threads_per_block = 256;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        if (fwer_mode)
        {
            /* ════════════════════════════════════════════════════════════════
             * FWER MAX-STATISTIC TWO-PASS STREAMING
             *
             * Pass 1: stream all parts → accumulate max |t| per permutation
             *         into host array h_max_t[nr_perm].
             * Pass 2: reopen files, stream all parts again → compute per-
             *         connection FWER p-value using the final h_max_t array.
             * ════════════════════════════════════════════════════════════════ */

            printf("========================================\n");
            printf("[FWER] Two-pass max-statistic permutation test\n");
            printf("[FWER] Pass 1: building max-|t| null distribution\n");
            printf("========================================\n");
            fflush(stdout);

            /* Allocate max_t arrays — host running accumulator + device chunk array */
            float *h_max_t      = (float*)calloc(nr_perm, sizeof(float));
            float *h_max_t_part = (float*)malloc(nr_perm * sizeof(float));
            float *d_max_t;
            err = cudaMalloc((void **)&d_max_t, sizeof(float) * nr_perm);
            if (err != cudaSuccess) {
                printf("CUDA ERROR! Failed to allocate d_max_t (%s)\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            /* Zero the device max_t — will be updated by atomicMaxFloat across all parts */
            err = cudaMemset(d_max_t, 0, sizeof(float) * nr_perm);

            /* Open subject files once for Pass 1 */
            printf("Opening all subject files for Pass 1...\n"); fflush(stdout);
            FileHandleArray* open_files_p1 = openAllSubjectFiles(filelist, nr_subs);

            /* ── Pass 1 loop ── */
            for (int p = 0; p < nr_parts; ++p)
            {
                printf("========================================\n");
                printf("[FWER P1] Part %i of %i\n", p+1, nr_parts);
                printf("========================================\n");
                fflush(stdout);

                float* device_buff = (float*)malloc(sizeof(float)*part_vals[p]*nr_subs);
                readRowsFromOpenFiles(open_files_p1, part_starts[p], part_ends[p], nr_subs, device_buff);
                printf("[FWER P1] Data loaded.\n"); fflush(stdout);

                /* Per-part t-stat output (we need to save t-stats for Pass 2
                 * recomputation — actually we re-derive t_obs in Pass 2, so
                 * this buffer here is just scratch for Pass 1 kernel output) */
                float *d_input, *d_tstat_scratch;
                err = cudaMalloc((void **)&d_input,         sizeof(float)*part_vals[p]*nr_subs);
                err = cudaMalloc((void **)&d_tstat_scratch, sizeof(float)*part_vals[p]);
                if (err != cudaSuccess) {
                    printf("CUDA ERROR! Pass 1 alloc failed (%s)\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
                err = cudaMemcpy(d_input, device_buff, sizeof(float)*part_vals[p]*nr_subs, cudaMemcpyHostToDevice);

                printf("[FWER P1] Launching %zu blocks x %d threads...\n",
                       part_vals[p], threads_per_block); fflush(stdout);
                CUDA_perm_fwer_pass1<<<part_vals[p], threads_per_block, shared_mem_size>>>(
                    d_input, d_perm, part_vals[p], nr_subs, nr_perm,
                    d_max_t, d_tstat_scratch, two_tailed);
                err = cudaGetLastError();
                if (err != cudaSuccess) printf("CUDA ERROR! Pass 1 kernel (%s)\n", cudaGetErrorString(err));
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) printf("CUDA ERROR! Pass 1 sync (%s)\n", cudaGetErrorString(err));

                /* Copy current d_max_t back to host and merge into h_max_t.
                 * The device array is already the running global max across all
                 * parts processed so far (atomicMax is persistent across launches
                 * since d_max_t stays allocated between parts). */
                err = cudaMemcpy(h_max_t_part, d_max_t, sizeof(float)*nr_perm, cudaMemcpyDeviceToHost);
                for (size_t i = 0; i < nr_perm; i++)
                    if (h_max_t_part[i] > h_max_t[i]) h_max_t[i] = h_max_t_part[i];

                printf("[FWER P1] Part %i done. Cleaning up...\n", p+1); fflush(stdout);
                free(device_buff);
                cudaFree(d_input);
                cudaFree(d_tstat_scratch);
            }

            closeAllSubjectFiles(open_files_p1);
            cudaFree(d_max_t);
            free(h_max_t_part);

            /* ── Compute null distribution stats ── */
            float max_t_min = h_max_t[0], max_t_max = h_max_t[0], max_t_mean = 0.f;
            for (size_t i = 0; i < nr_perm; i++) {
                if (h_max_t[i] < max_t_min) max_t_min = h_max_t[i];
                if (h_max_t[i] > max_t_max) max_t_max = h_max_t[i];
                max_t_mean += h_max_t[i];
            }
            max_t_mean /= nr_perm;
            printf("========================================\n");
            printf("[FWER P1] Max-|t| null distribution:\n");
            printf("  min=%.4f  mean=%.4f  max=%.4f\n", max_t_min, max_t_mean, max_t_max);
            printf("========================================\n");
            fflush(stdout);

            /* Upload final h_max_t to GPU for Pass 2 */
            float *d_max_t_final;
            err = cudaMalloc((void **)&d_max_t_final, sizeof(float) * nr_perm);
            err = cudaMemcpy(d_max_t_final, h_max_t, sizeof(float)*nr_perm, cudaMemcpyHostToDevice);
            free(h_max_t);

            /* ── Pass 2: open output files and stream all parts again ── */
            printf("========================================\n");
            printf("[FWER] Pass 2: computing FWER-corrected p-values\n");
            printf("========================================\n");
            fflush(stdout);

            printf("Opening all subject files for Pass 2...\n"); fflush(stdout);
            FileHandleArray* open_files_p2 = openAllSubjectFiles(filelist, nr_subs);

            /* Open output files */
            FILE *out_pval_bin  = NULL;
            FILE *out_tstat_bin = NULL;
            if (binary_output) {
                out_pval_bin  = fopen(outfile,       "wb");
                out_tstat_bin = fopen(outfile_tstat, "wb");
                if (!out_pval_bin || !out_tstat_bin) { perror("fopen binary output"); exit(EXIT_FAILURE); }
                write_permout_headers(out_pval_bin, out_tstat_bin,
                                      (size_t)nr_r1vals, (size_t)nr_vals);
            }

            for (int p = 0; p < nr_parts; ++p)
            {
                printf("========================================\n");
                printf("[FWER P2] Part %i of %i\n", p+1, nr_parts);
                printf("========================================\n");
                fflush(stdout);

                float* device_buff = (float*)malloc(sizeof(float)*part_vals[p]*nr_subs);
                readRowsFromOpenFiles(open_files_p2, part_starts[p], part_ends[p], nr_subs, device_buff);
                printf("[FWER P2] Data loaded.\n"); fflush(stdout);

                float* perm_test_res   = (float*)malloc(sizeof(float)*part_vals[p]);
                float* perm_test_tstat = (float*)malloc(sizeof(float)*part_vals[p]);

                float *d_input, *d_output_pval, *d_output_tstat;
                err = cudaMalloc((void **)&d_input,        sizeof(float)*part_vals[p]*nr_subs);
                err = cudaMalloc((void **)&d_output_pval,  sizeof(float)*part_vals[p]);
                err = cudaMalloc((void **)&d_output_tstat, sizeof(float)*part_vals[p]);
                if (err != cudaSuccess) {
                    printf("CUDA ERROR! Pass 2 alloc failed (%s)\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
                err = cudaMemcpy(d_input, device_buff, sizeof(float)*part_vals[p]*nr_subs, cudaMemcpyHostToDevice);

                printf("[FWER P2] Launching %zu blocks x %d threads...\n",
                       part_vals[p], threads_per_block); fflush(stdout);
                CUDA_perm_fwer_pass2<<<part_vals[p], threads_per_block, shared_mem_size>>>(
                    d_input, d_perm, part_vals[p], nr_subs, nr_perm,
                    d_max_t_final, d_output_pval, d_output_tstat, two_tailed);
                err = cudaGetLastError();
                if (err != cudaSuccess) printf("CUDA ERROR! Pass 2 kernel (%s)\n", cudaGetErrorString(err));
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) printf("CUDA ERROR! Pass 2 sync (%s)\n", cudaGetErrorString(err));

                err = cudaMemcpy(perm_test_res,   d_output_pval,  sizeof(float)*part_vals[p], cudaMemcpyDeviceToHost);
                err = cudaMemcpy(perm_test_tstat, d_output_tstat, sizeof(float)*part_vals[p], cudaMemcpyDeviceToHost);

                printf("[FWER P2] Saving results for part %i...\n", p+1); fflush(stdout);
                if (binary_output) {
                    appendPartialResultsBinary(out_pval_bin,  perm_test_res,   part_vals[p]);
                    appendPartialResultsBinary(out_tstat_bin, perm_test_tstat, part_vals[p]);
                    fflush(out_pval_bin);
                    fflush(out_tstat_bin);
                } else {
                    int is_first_write = (p == 0) ? 1 : 0;
                    appendPartialResults(perm_test_res,   part_starts[p], part_ends[p], nr_r1vals, outfile,       is_first_write);
                    appendPartialResults(perm_test_tstat, part_starts[p], part_ends[p], nr_r1vals, outfile_tstat, is_first_write);
                }

                printf("[FWER P2] Part %i done. Cleaning up...\n", p+1); fflush(stdout);
                free(device_buff);
                free(perm_test_res);
                free(perm_test_tstat);
                cudaFree(d_input);
                cudaFree(d_output_pval);
                cudaFree(d_output_tstat);
            }

            closeAllSubjectFiles(open_files_p2);
            cudaFree(d_max_t_final);

            if (binary_output) {
                if (out_pval_bin)  fclose(out_pval_bin);
                if (out_tstat_bin) fclose(out_tstat_bin);
            }
        }
        else
        {
            /* ════════════════════════════════════════════════════════════════
             * STANDARD per-connection permutation test (original streaming path)
             * ════════════════════════════════════════════════════════════════ */

            /* Check for existing results and determine where to resume */
            size_t existing_vals = binary_output
                ? countExistingResultsBinary(outfile, nr_r1vals)
                : countExistingResults(outfile, nr_r1vals);
            int start_part = 0;
            
            for (int p = 0; p < nr_parts; ++p)
            {
                if (existing_vals > part_ends[p]) start_part = p + 1;
                else break;
            }
            
            if (start_part > 0)
            {
                printf("========================================\n");
                printf("[RESUME] Resuming from part %d (already completed %d parts)\n", 
                       start_part + 1, start_part);
                printf("========================================\n");
                fflush(stdout);
            }
            else if (existing_vals > 0)
            {
                printf("========================================\n");
                printf("[RESUME] WARNING: Found partial results (%zu values) but they don't align with part boundaries.\n", 
                       existing_vals);
                printf("[RESUME] Starting from scratch and overwriting existing file.\n");
                printf("========================================\n");
                fflush(stdout);
            }

            printf("========================================\n");
            printf("Opening all subject files (happens ONCE)...\n");
            printf("========================================\n");
            fflush(stdout);
            FileHandleArray* open_files = openAllSubjectFiles(filelist, nr_subs);
            printf("All files are now open and ready for streaming!\n");
            fflush(stdout);

            FILE *out_pval_bin  = NULL;
            FILE *out_tstat_bin = NULL;
            if (binary_output) {
                int resuming = (start_part > 0);
                const char *open_mode = resuming ? "r+b" : "wb";
                out_pval_bin  = fopen(outfile,       open_mode);
                out_tstat_bin = fopen(outfile_tstat, open_mode);
                if (!out_pval_bin || !out_tstat_bin) { perror("fopen binary output"); exit(EXIT_FAILURE); }
                if (!resuming) {
                    write_permout_headers(out_pval_bin, out_tstat_bin,
                                          (size_t)nr_r1vals, (size_t)nr_vals);
                } else {
                    fseek(out_pval_bin,  0, SEEK_END);
                    fseek(out_tstat_bin, 0, SEEK_END);
                }
            }

            float *d_input, *d_output, *d_output_tstat;

            for (int p=start_part; p<nr_parts; ++p)
            {
                printf("========================================\n");
                printf("Processing part %i of %i\n", p+1, nr_parts);
                printf("========================================\n");
                fflush(stdout);

                float* device_buff = (float*)malloc(sizeof(float)*part_vals[p]*nr_subs);
                printf("Streaming data for part %i from open files (rows %zu to %zu)...\n", 
                       p+1, part_starts[p], part_ends[p]);
                fflush(stdout);
                readRowsFromOpenFiles(open_files, part_starts[p], part_ends[p], nr_subs, device_buff);
                printf("Data streaming complete for part %i!\n", p+1);
                fflush(stdout);

                float* perm_test_res   = (float*)malloc(sizeof(float)*part_vals[p]);
                float* perm_test_tstat = (float*)malloc(sizeof(float)*part_vals[p]);

                err = cudaMalloc((void **)&d_input,        sizeof(float)*part_vals[p]*nr_subs);
                err = cudaMalloc((void **)&d_output,       sizeof(float)*part_vals[p]);
                err = cudaMalloc((void **)&d_output_tstat, sizeof(float)*part_vals[p]);
                if (err!=cudaSuccess)
                    printf("CUDA ERROR! Failed to allocate device memory! (error code %s)\n", cudaGetErrorString(err));
                
                err = cudaMemcpy(d_input, device_buff, sizeof(float)*part_vals[p]*nr_subs, cudaMemcpyHostToDevice);
                if (err!=cudaSuccess)
                    printf("CUDA ERROR! Failed to copy memory from host to device (error code %s)\n", cudaGetErrorString(err));

                printf("  Launching %zu blocks × %d threads (%.2f million GPU threads)...\n", 
                       part_vals[p], threads_per_block, (part_vals[p] * threads_per_block) / 1e6);
                fflush(stdout);
                CUDA_perm<<<part_vals[p], threads_per_block, shared_mem_size>>>(
                    d_input, d_perm, part_vals[p], nr_subs, nr_perm,
                    d_output, d_output_tstat, two_tailed);
                
                err = cudaGetLastError();
                if (err != cudaSuccess)
                    printf("CUDA ERROR! Kernel launch failed (error code %s)\n", cudaGetErrorString(err));
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    printf("CUDA ERROR! Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
                
                err = cudaMemcpy(perm_test_res,   d_output,       sizeof(float)*part_vals[p], cudaMemcpyDeviceToHost);
                err = cudaMemcpy(perm_test_tstat, d_output_tstat, sizeof(float)*part_vals[p], cudaMemcpyDeviceToHost);
                if (err!=cudaSuccess)
                    printf("CUDA ERROR! Failed to copy memory from device to host (error code %s)\n", cudaGetErrorString(err));

                if (binary_output) {
                    appendPartialResultsBinary(out_pval_bin,  perm_test_res,   part_vals[p]);
                    appendPartialResultsBinary(out_tstat_bin, perm_test_tstat, part_vals[p]);
                    fflush(out_pval_bin);
                    fflush(out_tstat_bin);
                } else {
                    int is_first_write = (p == 0 && start_part == 0) ? 1 : 0;
                    appendPartialResults(perm_test_res,   part_starts[p], part_ends[p],
                                         nr_r1vals, outfile,       is_first_write);
                    appendPartialResults(perm_test_tstat, part_starts[p], part_ends[p],
                                         nr_r1vals, outfile_tstat, is_first_write);
                }

                free(device_buff);
                free(perm_test_res);
                free(perm_test_tstat);
                cudaFree(d_input);
                cudaFree(d_output);
                cudaFree(d_output_tstat);
                printf("Part %i complete!\n", p+1); fflush(stdout);
            }

            closeAllSubjectFiles(open_files);

            if (binary_output) {
                if (out_pval_bin)  fclose(out_pval_bin);
                if (out_tstat_bin) fclose(out_tstat_bin);
            }
        }

        /* shared cleanup */
        cudaFree(d_perm);
        free(perm_buff);

        printf("========================================\n");
        printf("All parts completed!\n");
        printf("P-values saved to: %s\n", outfile);
        printf("T-statistics saved to: %s\n", outfile_tstat);
        printf("done!\n");
        printf("========================================\n");
        fflush(stdout);
    }

    // Cleanup output filename buffer
    free(outfile_tstat);

    return(EXIT_SUCCESS);
}