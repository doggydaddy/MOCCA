// standard includes
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

// nifticlib includes
#include <nifti1.h>
#include <fslio.h>
#include <nifti1_io.h>

/* ── convenience macro: check CUDA calls and bail out on error ── */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ----------------- */
/* support functions */
/* ----------------- */

/* return the total system memory (*nix systems )*/
unsigned long long getTotalSystemMemory()
{
	long pages = sysconf(_SC_PHYS_PAGES);
	long page_size = sysconf(_SC_PAGE_SIZE);
	return pages*page_size;
}

/* return "Good-Values", or number of voxels inside the mask */
size_t getGoodValues(double ***mask, int dimx, int dimy, int dimz)
{
    int i, j, k;
    size_t gV = 0;
    for ( i=0; i<dimx; ++i) {
        for ( j=0; j<dimy; ++j ) {
            for (k=0; k<dimz; ++k) {
                if ( mask[i][j][k] != 0 )
                    gV++;
            }
        }
    }
    return gV;
}

/* allocate matrix of r rows and c columns */
float** createMatrix(int r, int c) {
    float **M;
    int i;
    M = (float**)malloc(r*sizeof(float*));
    if ( M==NULL) {
        printf("Error! Failed to allocate memory on host!\n");
        exit(-1);
    }
    for ( i=0; i<r; ++i ) {
        M[i] = (float*)malloc(c*sizeof(float));
        if ( M[i]==NULL ) {
            printf("Error! Failed to allocate memory on host!");
            exit(-1);
        }
    }
    return M;
}

/* reformat 4D array as obtained from fsl nifti-io library to 1d array
 * 
 * output 1d array is masked with what is only inside mask
 * 
 * indexing occurs from last to second index, i.e. i first, then j, finally k,
 * where input 4D data is input_data[time][k][j][i]
 */
void inputData4DtoArray(double ****input4D, float *h_arr, double ***mask, int dimx, int dimy, int dimz, int dimt)
{
    int i, j, k, t;
    size_t c = 0;
    for (k=0; k<dimz; ++k) {
        for (j=0; j<dimy; ++j) {
            for (i=0; i<dimx; ++i) {
                if (mask[k][j][i] != 0) {
                    for (t=0; t<dimt; ++t) {
                        h_arr[c] = (float)input4D[t][k][j][i];
                        c++;
                    }
                }
            }
        }
    }
}

void inputData4DtoArrayWithRanges(double ****input4D, float *h_arr, double ***mask, 
                                  int dimx, int dimy, int dimz, int dimt, 
                                  size_t start_idx, size_t end_idx)
{
    int i, j, k, t;
    size_t c = 0; // index for output "current" index
    size_t d = 0; // index for "n-th voxel within mask"
    for (k=0; k<dimz; ++k) 
    {
        for (j=0; j<dimy; ++j) 
        {
            for (i=0; i<dimx; ++i) 
            {
                if (mask[k][j][i] != 0) 
                {
                    if (d >= start_idx && d <= end_idx) 
                    {
                        for (t=0; t<dimt; ++t) 
                        {
                            h_arr[c] = (float)input4D[t][k][j][i];
                            c++;
                        }
                    }
                    d++;
                }
            }
        }
    }
}

int inputData4DtoArrayWith2Ranges(double ****input4D, float *h_arr, double ***mask, 
                                  int dimx, int dimy, int dimz, int dimt, 
                                  size_t start_idx1, size_t end_idx1, 
                                  size_t start_idx2, size_t end_idx2)
{
    int i, j, k, t;
    size_t c = 0; // index for output "current" index
    size_t d = 0; // index for "n-th voxel within mask"
    int div;
    for (k=0; k<dimz; ++k) 
    {
        for (j=0; j<dimy; ++j) 
        {
            for (i=0; i<dimx; ++i) 
            {
                if (mask[k][j][i] != 0) 
                {
                    if ((d >= start_idx1 && d <= end_idx1) || (d >= start_idx2 && d <= end_idx2)) 
                    {
                        if (d == start_idx2) {
                            div = c;
                        }
                        for (t=0; t<dimt; ++t) 
                        {
                            h_arr[c] = (float)input4D[t][k][j][i];
                            c++;
                        }
                    }
                    d++;
                }
            }
        }
    }
    return div;
}

/* save (2D full) cc-matrix to text file */
void saveToText_RECT(float *outputData, size_t nrows, size_t ncols, char *fileName)
{
    int i, j;
    FILE *output = fopen(fileName, "w");
    if (!output) { fprintf(stderr, "ERROR: could not open '%s'\n", fileName); return; }
    for( i=0; i<nrows; ++i ) {
        for( j=0; j<ncols; ++j ) {
            fprintf(output, "%lf ", outputData[i*ncols+j]);
        }
        fprintf(output, "\n");
    }
    fclose(output);
}

/* save (upper triangular without diagonal) cc-matrix to text file.
 * Uses a large write buffer to avoid per-number fwrite overhead. */
void saveToText_TRIA(float *outputData, size_t gV, char *fileName)
{
    FILE *output = fopen(fileName, "w");
    if (!output) {
        fprintf(stderr, "ERROR: could not open output file '%s'\n", fileName);
        return;
    }
    /* 4 MB write buffer – dramatically reduces fwrite syscall overhead */
    const size_t BUFSZ = 4 * 1024 * 1024;
    char *wbuf = (char *)malloc(BUFSZ);
    if (wbuf) setvbuf(output, wbuf, _IOFBF, BUFSZ);

	size_t c = 0;
    for (size_t i = 0; i < gV; ++i) {
        for (size_t j = i + 1; j < gV; ++j) {
            fprintf(output, "%.6f ", outputData[c]);
			c++;
        }
        fprintf(output, "\n");
    }
    fclose(output);
    free(wbuf);
}

/*
 * Binary ccmat format
 * -------------------
 * Offset  Size  Field
 *  0       4    magic number  0x43434D54  ("CCMT")
 *  4       4    format version (uint32, currently 1)
 *  8       8    gV  (uint64) — number of in-mask voxels
 * 16       8    n_elem = gV*(gV-1)/2  (uint64)
 * 24       n_elem * 4   upper-triangular float32 values, row-major
 *
 * The layout is identical to the text format: element [i,j] (j>i) is at
 * index  k = gV*(gV-1)/2 - (gV-i)*(gV-i-1)/2 + j - i - 1
 */
#define CCMAT_MAGIC   0x43434D54u
#define CCMAT_VERSION 1u

/* Save upper-triangular float array to a compact binary file.
 * Single fwrite → limited only by disk bandwidth. */
void saveToBinary_TRIA(float *outputData, size_t gV, char *fileName)
{
    FILE *output = fopen(fileName, "wb");
    if (!output) {
        fprintf(stderr, "ERROR: could not open binary output '%s'\n", fileName);
        return;
    }

    uint32_t magic   = CCMAT_MAGIC;
    uint32_t version = CCMAT_VERSION;
    uint64_t gV64    = (uint64_t)gV;
    uint64_t n_elem  = (uint64_t)(gV * (gV - 1) / 2);

    fwrite(&magic,   sizeof(uint32_t), 1, output);
    fwrite(&version, sizeof(uint32_t), 1, output);
    fwrite(&gV64,    sizeof(uint64_t), 1, output);
    fwrite(&n_elem,  sizeof(uint64_t), 1, output);
    fwrite(outputData, sizeof(float), n_elem, output);

    fclose(output);
}

/* converts input upper-triangular matrix without diagnoal to full 2D matrix
 * (for output to text file).
 * 
 * forces diagonal to be 0 (instead of 1 for pure cc-matrix)
 */
void upperTriangularToFull(float *input, float **output, size_t gV)
{
    int i, j;
    size_t c = 0;
    for( i=0; i<gV; ++i ) {
        for( j=i; j<gV; ++j ) {
            if ( i == j ) {
                output[i][j] = 1.0;
            } else {
                output[i][j] = input[c];
                output[j][i] = input[c];
                c++;
            }
        }
    }
}

/* retrieves n-th voxel from the from data of dimensions (nV x time) */
void getNthSeries(float *data, float *output, int n, size_t gV, int nt) 
{
    int c;
    int i;

	c = 0;
    for (i=n*nt; i<n*nt+nt; ++i) 
	{
        output[c] = data[i];
        c++; // all good cpp code needs at least one "c++"
    }
}

/*
 * calculate the cross-correlation matrix of data of size (gV x nt)
 * at n-th and m-th indices.
 * 
 * implementated for ease of reading and omp extensions.
 * 
 * there is a faster (fft-based) implementation, but this is only to check with
 * cuda results so, good enough?
*/
float calcCrossCorr_omp(float *data, int n, int m, size_t gV, int nt)
{
    float t1[nt]; 
    float t2[nt];
    getNthSeries(data, t1, n, gV, nt);
    getNthSeries(data, t2, m, gV, nt);

    float m1 = 0; 
    float m2 = 0;
    for (int i=0; i<nt; ++i) 
	{
        m1 += t1[i];
        m2 += t2[i];
    }
    m1 /= nt; 
    m2 /= nt;    

    float nom = 0; 
	float de1 = 0; 
	float de2 = 0;
    for (int i=0 ; i<nt ; ++i) 
	{
        nom += (t1[i] - m1) * (t2[i] - m2);     
        de1 += (t1[i] - m1) * (t1[i] - m1);  
        de2 += (t2[i] - m2) * (t2[i] - m2);
    }

    float output;
    output = nom / ( sqrt(de1*de2) ); 

    return output;
}

void calcCrossCorr_DIAG(float *data, float *result, size_t gV, int nt)
{
    printf("calculating diagonal block of size (%zu, %zu)\n", gV, gV); 
    #pragma omp parallel for
    for ( int i=0; i<gV; ++i ) 
	{
        for ( int j=i+1; j<gV; ++j ) 
		{
		    size_t k = (gV*(gV-1)/2)-(gV-i)*((gV-i)-1)/2+j-i-1;
            result[k] = calcCrossCorr_omp(data, i, j, gV, nt);  
        }
    }
}

void calcCrossCorr_OFFD(float *data, float *result, size_t gV, int nt, size_t div)
{
    size_t rows, cols;
    rows = div;
    cols = gV-div;
    printf("calculating off-diagonal block of size (%zu, %zu)\n", rows, cols); 
    #pragma omp parallel for
    for (size_t i=0; i<rows; ++i ) 
	{
        for (size_t j=0; j<cols; ++j ) 
		{
            result[(i*cols)+j] = calcCrossCorr_omp(data, i, j+rows, gV, nt);  
        }
    }
}

int sanityCheck(float *input, size_t input_size) 
{
    for(int i=0;i<input_size;++i)
    {
        if (input[i] == 0 || input[i] >= 1 || input[i] <= -1) 
        {
            printf("sanity check failed, at index %d with %f\n", i, input[i]);
            return 0;
        }
    }
    return 1;
}

/* 2D CUDA Kernel for calculating the cross-correlation matrix 
 * of data with size (gV x nt). 
 * Diagonal blocks: computes upper triangle (n < m).
 * Uses rsqrtf for speed; guards against zero-variance voxels (→ 0.0).
 */
__global__ void CUDA_CC_DIAG(float *data, float *answer, size_t gV, int nt) 
{
	float mean_t1 = 0;
	float mean_t2 = 0;
	float nom = 0;
	float de1 = 0;
	float de2 = 0;
	size_t k = 0;

	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;

	if ( n >= gV || m >= gV ) return;
	if ( m <= n ) return; 

	for (int i=0; i<nt; ++i) 
	{
		mean_t1 += data[nt*n+i];	
		mean_t2 += data[nt*m+i];
	}
	mean_t1 /= nt;
	mean_t2 /= nt;

	for (int i=0; i<nt; ++i) 
	{
		nom += (data[nt*n+i]-mean_t1)*(data[nt*m+i]-mean_t2);
		de1 += (data[nt*n+i]-mean_t1)*(data[nt*n+i]-mean_t1);
		de2 += (data[nt*m+i]-mean_t2)*(data[nt*m+i]-mean_t2);
	}

	/* guard: zero-variance voxel → correlation undefined, store 0 */
	k = (gV*(gV-1)/2)-(gV-n)*((gV-n)-1)/2+m-n-1;
	answer[k] = (de1 > 0.0f && de2 > 0.0f) ? nom * rsqrtf(de1*de2) : 0.0f;
}

/* 2D CUDA Kernel for calculating the cross-correlation matrix 
 * of data with size (gV x nt). 
 * Off-diagonal blocks: n ∈ [0, div), m ∈ [div, gV).
 * Uses rsqrtf for speed; guards against zero-variance voxels (→ 0.0).
 */
__global__ void CUDA_CC_OFFD(float *data, float *answer, size_t gV, int nt, size_t div) 
{
	float mean_t1 = 0;
	float mean_t2 = 0;
	float nom = 0;
	float de1 = 0;
	float de2 = 0;

	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;

	if ( n >= gV || m >= gV ) return;

    // boundary conditions:
    // n(i) = [0;div)
    // m(j) = [div;gV)
    if ( n >= div || m < div ) return;

	for (int i=0; i<nt; ++i) 
	{
		mean_t1 += data[nt*n+i];	
		mean_t2 += data[nt*m+i];
	}
	mean_t1 /= nt;
	mean_t2 /= nt;

	for (int i=0; i<nt; ++i) 
	{
		nom += (data[nt*n+i]-mean_t1)*(data[nt*m+i]-mean_t2);
		de1 += (data[nt*n+i]-mean_t1)*(data[nt*n+i]-mean_t1);
		de2 += (data[nt*m+i]-mean_t2)*(data[nt*m+i]-mean_t2);
	}

	/* guard: zero-variance voxel → correlation undefined, store 0 */
	answer[(n*(gV-div))+(m-div)] = (de1 > 0.0f && de2 > 0.0f) ? nom * rsqrtf(de1*de2) : 0.0f;
}

/* main function */
int main ( int argc , char * argv[] ) 
{
	/* 
	 * -------------------------------------------------------
     * Initialize, parse inputs, and calculate memory required
	 * -------------------------------------------------------
	 */

    /* ── argument check ── */
    if (argc < 4 || argc > 5)
    {
        fprintf(stderr, "Usage: %s <input_4d.nii> <mask.nii> <output.ccmat> [-b]\n", argv[0]);
        fprintf(stderr, "  -b   write binary .ccmat instead of text (faster, ~6x smaller)\n");
        return EXIT_FAILURE;
    }

    /* parsing input parametres */
    char *inputFile, *maskFile, *outputFile;
    int binary_output = 0;
    inputFile  = argv[1];
    maskFile   = argv[2];
    outputFile = argv[3];
    if (argc == 5 && strcmp(argv[4], "-b") == 0) binary_output = 1;

    printf("========================================\n");
    printf("cudaCC_div\n");
    printf("  input  : %s\n", inputFile);
    printf("  mask   : %s\n", maskFile);
    printf("  output : %s  (%s)\n", outputFile, binary_output ? "binary" : "text");
    printf("========================================\n");

    /* loading mask */
    FSLIO *fslio = FslInit();
    void *buffer = FslReadAllVolumes(fslio, maskFile);
    double ***mask = FslGetVolumeAsScaledDouble(fslio, 0);
    FslClose(fslio);
    free(buffer);

    /* loading the input dataset — capture dimensions BEFORE FslClose */
    fslio = FslInit();
    buffer = FslReadAllVolumes(fslio, inputFile);
    int nx_data = fslio->niftiptr->nx;
    int ny_data = fslio->niftiptr->ny;
    int nz_data = fslio->niftiptr->nz;
    int nt_data = fslio->niftiptr->nt;
    double ****data = FslGetBufferAsScaledDouble(fslio);
    FslClose(fslio);
    /* note: buffer (raw NIfTI allocation) is owned by niftilib; do not free here */

	/* Error code — kept for any legacy checks below */
	(void)cudaSuccess; /* suppress unused-var warning */

	/* Calculate sizes */
	/* calculate number of voxels inside the mask */
    size_t gV = getGoodValues(mask, nx_data, ny_data, nz_data);
	printf("number of voxels within the mask: %zu\n", gV);

	unsigned long num_elem_input  = (unsigned long)gV * nt_data;
	unsigned long num_elem_output = (unsigned long)((gV*(gV-1))/2);
	size_t input_size  = num_elem_input  * sizeof(float);
	size_t output_size = num_elem_output * sizeof(float);
	printf("Number of elements in input : %lu  (%zu bytes)\n", num_elem_input,  input_size);
	printf("Number of elements in output: %lu  (%zu bytes)\n", num_elem_output, output_size);

	/* print total memory needed */
	const size_t MiB = 1024ULL*1024;
	const size_t GiB = MiB*1024;
	printf("Input  size : %.1f MiB\n", (double)input_size  / MiB);
	printf("Output size : %.2f GiB\n", (double)output_size / GiB);

    /* total memory we have */
    printf("Getting system memory availability:\n");
    printf("===================================\n");
	unsigned long long system_memory = getTotalSystemMemory();
	size_t device_free_mem, device_total_mem;
	cudaMemGetInfo(&device_free_mem, &device_total_mem);
    printf("CPU available memory : %llu MiB\n", system_memory / MiB);
	printf("GPU free / total     : %zu / %zu MiB\n", device_free_mem / MiB, device_total_mem / MiB);

    /* buffer output image on host */ 
    printf("Allocating output buffer on host ...\n");
    float *output_buffer = (float*)malloc(output_size);
    if ( output_buffer == NULL ) 
    {
        fprintf(stderr, "Failed to allocate output buffer on host!\n");
        printf("Please upgrade your potato computer and try again!\n");
        return EXIT_FAILURE;
    }
    printf("...done!\n");

    /* For device memory:
     * We need approximately input_size + output_size 
     * to compute everything in one go */
    size_t device_req_mem = (size_t)(input_size + output_size);

    float num_divisions = 1.0;
    float num_runs = 1.0;
	if ( device_req_mem > device_free_mem ) 
	{
		printf("The job is too big to fit onto the device memory all at once ...\n");
		printf("... so we need to divide the computation into blocks.\n");

        /* max voxels we can fit: solve  blk*nt*4 + (blk*(blk-1)/2)*4 <= free
         * approximated conservatively as: blk^2/2 <= (free - input_size) / 4
         * → blk <= sqrt(2*(free-input_size)/sizeof(float))               */
        size_t available = (device_free_mem > input_size) ? (device_free_mem - input_size) : 0;
        size_t max_blk_size = (size_t)(sqrtf(2.0f * (float)available / sizeof(float)));
        num_divisions = ceilf((float)gV / (float)max_blk_size);
        num_runs = (num_divisions*(num_divisions+1)) / 2;
        printf("Block size        : %zu voxels\n", max_blk_size);
        printf("Number of blocks  : %d\n", (int)num_divisions);
        printf("Number of runs    : %d\n", (int)num_runs);

        int start_idx[(int)num_divisions];
        int end_idx[(int)num_divisions];
        for (int i=0; i<(int)num_divisions; ++i)
        {
            start_idx[i] = i * max_blk_size;
            end_idx[i]   = start_idx[i] + max_blk_size - 1;
            if ((size_t)end_idx[i] >= gV) end_idx[i] = (int)gV - 1;
        }
        printf("[DBG]: Division indices:\n");
        for (int i=0; i<(int)num_divisions; ++i)
            printf("  div %d: [%d, %d]\n", i, start_idx[i], end_idx[i]);

        /* free the large 4D data array now — not needed for the block path */
        free(data);
        data = NULL;
        /* reload input for block extraction */
        fslio = FslInit();
        FslReadAllVolumes(fslio, inputFile);
        data = FslGetBufferAsScaledDouble(fslio);
        FslClose(fslio);

        /* ========================= */
        /* performing the block runs */
        /* ========================= */

        float *h_data;
        float *gpu_ccmat;
        size_t Fi, Fj, Fk, bk;
        size_t blk_size_x, blk_size_y, blk_size;
        int current_run = 0;

        /* CUDA events for accurate GPU timing */
        cudaEvent_t t_start, t_stop;
        CUDA_CHECK(cudaEventCreate(&t_start));
        CUDA_CHECK(cudaEventCreate(&t_stop));
        CUDA_CHECK(cudaEventRecord(t_start));

        for (int blk_x = 0; blk_x < (int)num_divisions; ++blk_x) 
        {
            for (int blk_y = blk_x; blk_y < (int)num_divisions; ++blk_y) 
            {
                printf("----\nRun %d  block (%d,%d)\n", current_run, blk_x, blk_y);

                if (blk_x == blk_y) 
                {
                    /* ---- diagonal block ---- */
                    blk_size   = (size_t)(end_idx[blk_x] - start_idx[blk_x] + 1);
                    input_size  = blk_size * nt_data * sizeof(float);
                    output_size = ((blk_size*(blk_size-1))/2) * sizeof(float);
                    printf("Diagonal block, size %zu\n", blk_size);

                    h_data    = (float *)malloc(input_size);
                    gpu_ccmat = (float *)malloc(output_size);
                    if (!h_data || !gpu_ccmat) {
                        fprintf(stderr, "Failed to allocate host memory for block!\n");
                        return EXIT_FAILURE;
                    }

                    printf("Extracting data range [%d,%d] ...\n", start_idx[blk_x], end_idx[blk_x]);
                    inputData4DtoArrayWithRanges(data, h_data, mask, 
                                                 nx_data, ny_data, nz_data, nt_data, 
                                                 start_idx[blk_x], end_idx[blk_x]);
                    printf("...done!\n");

                    float *d_data = NULL, *d_result = NULL;
                    CUDA_CHECK(cudaMalloc((void **)&d_data,   input_size));
                    CUDA_CHECK(cudaMalloc((void **)&d_result, output_size));
                    CUDA_CHECK(cudaMemcpy(d_data, h_data, input_size, cudaMemcpyHostToDevice));

                    dim3 dimBlock(16, 16);
                    dim3 dimGrid((blk_size + dimBlock.x - 1) / dimBlock.x,
                                 (blk_size + dimBlock.y - 1) / dimBlock.y);
                    printf("Launching CUDA_CC_DIAG grid(%u,%u) block(16,16)\n",
                           dimGrid.x, dimGrid.y);
                    CUDA_CC_DIAG<<<dimGrid, dimBlock>>>(d_data, d_result, blk_size, nt_data);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());

                    CUDA_CHECK(cudaMemcpy(gpu_ccmat, d_result, output_size, cudaMemcpyDeviceToHost));

                    /* stitch into full output buffer */
                    printf("Stitching diagonal block into output buffer ...\n");
                    for (int bi = 0; bi < (int)blk_size; ++bi) 
                    {
                        for (int bj = bi+1; bj < (int)blk_size; ++bj) 
                        {
                            bk = (blk_size*(blk_size-1)/2)-(blk_size-bi)*((blk_size-bi)-1)/2+bj-bi-1;
                            Fi = (size_t)(blk_x*max_blk_size) + bi;
                            Fj = (size_t)(blk_y*max_blk_size) + bj;
                            Fk = (gV*(gV-1)/2)-(gV-Fi)*((gV-Fi)-1)/2+Fj-Fi-1;
                            output_buffer[Fk] = gpu_ccmat[bk];
                        }
                    }
                    printf("...done!\n");

                    CUDA_CHECK(cudaFree(d_data));
                    CUDA_CHECK(cudaFree(d_result));
                    free(h_data);
                    free(gpu_ccmat);
                }
                else 
                {
                    /* ---- off-diagonal block ---- */
                    blk_size_x = (size_t)(end_idx[blk_x] - start_idx[blk_x] + 1);
                    blk_size_y = (size_t)(end_idx[blk_y] - start_idx[blk_y] + 1);
                    printf("Off-diagonal block, size (%zu, %zu)\n", blk_size_x, blk_size_y);
                    input_size  = (blk_size_x + blk_size_y) * nt_data * sizeof(float);
                    output_size = blk_size_x * blk_size_y * sizeof(float);

                    h_data    = (float *)malloc(input_size);
                    gpu_ccmat = (float *)malloc(output_size);
                    if (!h_data || !gpu_ccmat) {
                        fprintf(stderr, "Failed to allocate host memory for block!\n");
                        return EXIT_FAILURE;
                    }

                    /* NOTE: blk_x range loaded first (rows), blk_y range second (cols).
                     * inputData4DtoArrayWith2Ranges stores range1 first, then range2.
                     * div = number of elements from range1 = blk_size_x. */
                    printf("Extracting data ranges [%d,%d] + [%d,%d] ...\n",
                           start_idx[blk_x], end_idx[blk_x],
                           start_idx[blk_y], end_idx[blk_y]);
                    int div = inputData4DtoArrayWith2Ranges(data, h_data, mask,
                                                            nx_data, ny_data, nz_data, nt_data,
                                                            start_idx[blk_x], end_idx[blk_x],
                                                            start_idx[blk_y], end_idx[blk_y]);
                    printf("...done!  div=%d\n", div);

                    float *d_data = NULL, *d_result = NULL;
                    CUDA_CHECK(cudaMalloc((void **)&d_data,   input_size));
                    CUDA_CHECK(cudaMalloc((void **)&d_result, output_size));
                    CUDA_CHECK(cudaMemcpy(d_data, h_data, input_size, cudaMemcpyHostToDevice));

                    size_t combined = blk_size_x + blk_size_y;
                    dim3 dimBlock(16, 16);
                    dim3 dimGrid((combined + dimBlock.x - 1) / dimBlock.x,
                                 (combined + dimBlock.y - 1) / dimBlock.y);
                    printf("Launching CUDA_CC_OFFD grid(%u,%u) block(16,16)\n",
                           dimGrid.x, dimGrid.y);
                    CUDA_CC_OFFD<<<dimGrid, dimBlock>>>(d_data, d_result, combined, nt_data, blk_size_x);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());

                    CUDA_CHECK(cudaMemcpy(gpu_ccmat, d_result, output_size, cudaMemcpyDeviceToHost));

                    /* stitch into full output buffer */
                    printf("Stitching off-diagonal block into output buffer ...\n");
                    for (size_t bi = 0; bi < blk_size_x; ++bi) 
                    {
                        for (size_t bj = 0; bj < blk_size_y; ++bj)
                        {
                            Fi = (size_t)(blk_x*max_blk_size) + bi;
                            Fj = (size_t)(blk_y*max_blk_size) + bj;
                            Fk = (gV*(gV-1)/2)-(gV-Fi)*((gV-Fi)-1)/2+Fj-Fi-1;
                            bk = bi * blk_size_y + bj;
                            output_buffer[Fk] = gpu_ccmat[bk];
                        }
                    }
                    printf("...done!\n");

                    CUDA_CHECK(cudaFree(d_data));
                    CUDA_CHECK(cudaFree(d_result));
                    free(h_data);
                    free(gpu_ccmat);
                }
                current_run++;
            }
        }

        CUDA_CHECK(cudaEventRecord(t_stop));
        CUDA_CHECK(cudaEventSynchronize(t_stop));
        float gpu_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t_start, t_stop));
        printf("GPU computation time: %.2f s\n", gpu_ms / 1000.0f);
        CUDA_CHECK(cudaEventDestroy(t_start));
        CUDA_CHECK(cudaEventDestroy(t_stop));
    }
    else
    {
	    printf("Whole dataset fits in GPU memory — single-pass computation.\n");
	    float *h_data = (float *)malloc(input_size);
	    if ( h_data == NULL ) 
        {
		    fprintf(stderr, "Failed to allocate memory on host!\n");
		    exit(EXIT_FAILURE);
	    }

        printf("Parsing input data into flat masked array on host ...\n");
        inputData4DtoArray(data, h_data, mask, nx_data, ny_data, nz_data, nt_data);
        printf("... freeing 4D source data ...\n");
        free(data);
        printf("...done!\n");

	    float *d_data = NULL;
	    float *d_result = NULL;
	    CUDA_CHECK(cudaMalloc((void **)&d_data,   input_size));
	    CUDA_CHECK(cudaMalloc((void **)&d_result, output_size));
	    printf("Copying input to device ...\n");

        cudaEvent_t t_start, t_stop;
        CUDA_CHECK(cudaEventCreate(&t_start));
        CUDA_CHECK(cudaEventCreate(&t_stop));
        CUDA_CHECK(cudaEventRecord(t_start));

        CUDA_CHECK(cudaMemcpy(d_data, h_data, input_size, cudaMemcpyHostToDevice));
        printf("...done!\n");

        dim3 dimBlock(16, 16);
        dim3 dimGrid((gV + dimBlock.x - 1) / dimBlock.x,
                     (gV + dimBlock.y - 1) / dimBlock.y);
        printf("Launching CUDA_CC_DIAG grid(%u,%u) block(16,16) ...\n",
               dimGrid.x, dimGrid.y);
        CUDA_CC_DIAG<<<dimGrid, dimBlock>>>(d_data, d_result, gV, nt_data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("...done!\n");

        CUDA_CHECK(cudaMemcpy(output_buffer, d_result, output_size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(t_stop));
        CUDA_CHECK(cudaEventSynchronize(t_stop));
        float gpu_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t_start, t_stop));
        printf("GPU computation time: %.2f s\n", gpu_ms / 1000.0f);
        CUDA_CHECK(cudaEventDestroy(t_start));
        CUDA_CHECK(cudaEventDestroy(t_stop));

        free(h_data);
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_result));
    }
	
	/* -------------------------------------------------------
	 * saving output
	 * -------------------------------------------------------
	 */

    struct timespec ts_save_start, ts_save_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_save_start);
    printf("Saving output to %s (%s) ...\n", outputFile,
           binary_output ? "binary" : "text");

    if (binary_output)
        saveToBinary_TRIA(output_buffer, gV, outputFile);
    else
        saveToText_TRIA(output_buffer, gV, outputFile);

    clock_gettime(CLOCK_MONOTONIC, &ts_save_end);
    double save_sec = (ts_save_end.tv_sec  - ts_save_start.tv_sec) +
                      (ts_save_end.tv_nsec - ts_save_start.tv_nsec) * 1e-9;
    printf("... done!  Save time: %.2f s\n", save_sec);

    free(output_buffer);
    printf("========================================\n");
    printf("cudaCC_div complete.\n");
    printf("========================================\n");
	return 0;
}

