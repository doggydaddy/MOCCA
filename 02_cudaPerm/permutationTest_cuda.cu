#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <omp.h>

/* ── Binary ccmat format (shared by input ccmats AND output permout files) ──
 *
 * INPUT  (.ccmat):  produced by cudaCC_div
 * OUTPUT (.permout) produced by permutationTest_cuda when -b flag is given
 *
 * Both share the same header layout:
 *
 * Offset  Size  Field
 *  0       4    magic  0x43434D54 ("CCMT")  [ccmat]
 *                      0x50455254 ("PERT")  [permout]
 *  4       4    version (uint32, currently 1)
 *  8       8    gV     (uint64) — number of voxels (NOT connections)
 * 16       8    n_elem = gV*(gV-1)/2  (uint64)
 * 24       n_elem * 4   upper-triangular float32, row-major
 * ─────────────────────────────────────────────────────────────────────────── */
#define CCMAT_MAGIC   0x43434D54u   /* input  ccmat  files ("CCMT") */
#define PERMOUT_MAGIC 0x50455254u   /* output permout files ("PERT") */
#define CCMAT_VERSION 1u
#define CCMAT_HDR_SIZE 24   /* bytes before the float data */

/* Returns 1 if the file starts with the binary magic number, 0 otherwise. */
static int isBinaryCCmat(const char* filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f) return 0;
    uint32_t magic = 0;
    fread(&magic, sizeof(uint32_t), 1, f);
    fclose(f);
    return (magic == CCMAT_MAGIC);
}

/* Read gV from binary header. */
static uint64_t binaryGetGV(const char* filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen"); exit(EXIT_FAILURE); }
    fseek(f, 8, SEEK_SET);   /* skip magic + version */
    uint64_t gV = 0;
    fread(&gV, sizeof(uint64_t), 1, f);
    fclose(f);
    return gV;
}

///
/// @brief returns number of lines (rows) a ccmat file has
/// @param filename input ccmat file (text or binary)
/// @return number of rows = gV
/// 
/// used to get number of voxels (not connections!) a subject data file has.
///
size_t getNumberLines(char* filename)
{
    /* Binary format: gV is stored directly in the header */
    if (isBinaryCCmat(filename))
        return (size_t)binaryGetGV(filename);

    /* Text format: count newlines */
    FILE *stream;
    size_t nlines = 0;
    int ch;

    stream = fopen(filename, "r");
    if (stream == NULL) 
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    // Fast line counting without getline (just count newlines)
    while ((ch = fgetc(stream)) != EOF)
    {
        if (ch == '\n')
        {
            nlines++;
        }
    }

    fclose(stream);

    return(nlines);
}

///
/// @brief returns the number of values a file has in the first line (row 0)
/// @param filename input ccmat file (text or binary)
/// @return number of values in row 0 = gV - 1
///
/// used to get number of voxels (not connections!) a subject data file has.
/// Sanity check getNumberLines() too as proper subject connectivity data should
/// have the same number of lines as values in the first line!
///
size_t getNumberValsFirstLine(char* filename)
{
    /* Binary format: first row has (gV-1) values */
    if (isBinaryCCmat(filename))
    {
        uint64_t gV = binaryGetGV(filename);
        return (size_t)(gV > 0 ? gV - 1 : 0);
    }

    /* Text format: parse the first line */
    FILE *stream;
    char *line = NULL;
    size_t len = 0;

    stream = fopen(filename, "r");
    if (stream == NULL) 
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    size_t line_length = 0;
    if (getline(&line, &len, stream) != -1) 
    {
        // Count values using pointer advancement instead of strtok
        char* ptr = line;
        char* end;
        
        while (*ptr != '\0' && *ptr != '\n')
        {
            // Skip whitespace
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            if (*ptr == '\0' || *ptr == '\n') break;
            
            // Try to parse a float (we just need to count, not store)
            strtof(ptr, &end);
            
            if (ptr == end) break; // No more numbers
            
            line_length++;
            ptr = end;
        }
    }

    fclose(stream);
    free(line);

    return(line_length);
}

///
/// @brief peek into file list and grabs the number of lines each subject has
/// @param filelist input file list 
/// @return number of lines each subject has
///
/// convenience routine to grab subject dimensions using the file list
/// references only
///
size_t peekFileList(char* filelist)
{
    FILE *fl;
    char* fl_line = NULL;
    size_t fl_len = 0;

    size_t output;

    fl = fopen(filelist, "r");
    if (fl == NULL) 
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    printf("[peekFileList] Opening first subject file to determine dimensions...\n");
    fflush(stdout);
    while( getline(&fl_line, &fl_len, fl) != -1 )
    {
        fl_line[strcspn(fl_line, "\n")] = 0;
        printf("[peekFileList] Attempting to open: %s\n", fl_line);
        fflush(stdout);
        output = getNumberLines(fl_line);
        printf("[peekFileList] File has %zu lines\n", output);
        fflush(stdout);
        break; // only need first file
    } 

    fclose(fl);
    free(fl_line);

    return(output);
}

///
/// @brief reads permutations file
/// @param filename generated permutations file from generatePermutations.py program
/// @param buffer array of ints to store the parsed permutations
/// @param nr_subs number of subjects in total for the test
///
/// note the expected input (as obtained from the output of
/// generatePermutations.py) is indices of a group (group A), and NOT one-hot
///
/// the output buffer contains one-hot labels of the subject permutations
///
void parsePermutations(char* filename, int* buffer, size_t nr_subs)
{
    /* variables to load permutations file */
    FILE *pt;
    char *pt_line = NULL;
    size_t pt_len = 0;
    ssize_t pt_lines;

    // opening permutations file
    pt = fopen(filename, "r");
    if (pt == NULL) 
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    size_t line_idx = 0;
    while ((pt_lines = getline(&pt_line, &pt_len, pt)) != -1) 
    {
        // Use strtol for faster integer parsing
        char* ptr = pt_line;
        char* end;
        
        while (*ptr != '\0' && *ptr != '\n') 
        {
            // Skip whitespace
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            if (*ptr == '\0' || *ptr == '\n') break;
            
            // Parse integer
            long val = strtol(ptr, &end, 10);
            
            // Check if parsing succeeded
            if (ptr == end) break;
            
            // Set one-hot encoding
            if (val >= 0 && val < nr_subs)
            {
                buffer[(line_idx * nr_subs) + val] = 1;
            }
            
            ptr = end;
        }
        line_idx++;
    } 

    fclose(pt);
    free(pt_line);
}

///
/// @brief Structure to hold open file handles for streaming reads
///
typedef struct {
    FILE** file_handles;
    size_t nr_files;
} FileHandleArray;

///
/// @brief Opens all subject files and returns array of file handles
/// @param filelist path to file list
/// @param nr_subs number of subjects
/// @return FileHandleArray structure with open file handles
///
FileHandleArray* openAllSubjectFiles(char* filelist, size_t nr_subs)
{
    FILE *fl;
    char* fl_line = NULL;
    size_t fl_len = 0;
    
    fl = fopen(filelist, "r");
    if (fl == NULL) 
    {
        perror("fopen filelist");
        exit(EXIT_FAILURE);
    }

    FileHandleArray* fha = (FileHandleArray*)malloc(sizeof(FileHandleArray));
    fha->nr_files = nr_subs;
    fha->file_handles = (FILE**)malloc(sizeof(FILE*) * nr_subs);
    
    size_t sub_idx = 0;
    printf("Opening all %zu subject files...\n", nr_subs);
    fflush(stdout);
    
    while(getline(&fl_line, &fl_len, fl) != -1 && sub_idx < nr_subs)
    {
        fl_line[strcspn(fl_line, "\n")] = 0;
        
        fha->file_handles[sub_idx] = fopen(fl_line, isBinaryCCmat(fl_line) ? "rb" : "r");
        if (fha->file_handles[sub_idx] == NULL) 
        {
            fprintf(stderr, "Error opening file %zu: %s\n", sub_idx+1, fl_line);
            perror("fopen");
            exit(EXIT_FAILURE);
        }
        
        if ((sub_idx + 1) % 50 == 0 || sub_idx == 0) {
            printf("  Opened %zu/%zu files\n", sub_idx+1, nr_subs);
            fflush(stdout);
        }
        
        sub_idx++;
    }
    
    fclose(fl);
    free(fl_line);
    
    printf("All %zu files opened successfully!\n", nr_subs);
    fflush(stdout);
    
    return fha;
}

///
/// @brief Closes all open file handles
/// @param fha FileHandleArray structure
///
void closeAllSubjectFiles(FileHandleArray* fha)
{
    printf("Closing all %zu subject files...\n", fha->nr_files);
    fflush(stdout);
    
    for (size_t i = 0; i < fha->nr_files; i++)
    {
        if (fha->file_handles[i] != NULL)
        {
            fclose(fha->file_handles[i]);
        }
    }
    
    free(fha->file_handles);
    free(fha);
    
    printf("All files closed.\n");
    fflush(stdout);
}

///
/// @brief Reads specific connection indices [N..M] from open file handles (streaming)
/// @param fha FileHandleArray with open file handles
/// @param N from (and including) flat index into upper-triangular array
/// @param M to (and including) flat index
/// @param nr_sub number of subjects
/// @param buffer output buffer
///
/// Supports both text and binary ccmat files transparently.
/// Binary path: single fseek + fread — orders of magnitude faster than text parsing.
///
void readRowsFromOpenFiles(FileHandleArray* fha, size_t N, size_t M, int nr_sub, float* buffer)
{
    size_t n_vals = M - N + 1;

    #pragma omp parallel for schedule(dynamic)
    for (int sub_idx = 0; sub_idx < nr_sub; sub_idx++)
    {
        FILE* stream = fha->file_handles[sub_idx];

        /* ── detect format by peeking at magic (non-destructive for binary,
         *    harmless for text since we rewind anyway)                      */
        rewind(stream);
        uint32_t magic = 0;
        fread(&magic, sizeof(uint32_t), 1, stream);
        int is_binary = (magic == CCMAT_MAGIC);

        if (is_binary)
        {
            /* Binary path:
             * Data starts at byte CCMAT_HDR_SIZE (24).
             * Element N is at offset  CCMAT_HDR_SIZE + N*sizeof(float).
             * Read exactly n_vals floats in one call.                       */
            long offset = (long)(CCMAT_HDR_SIZE + N * sizeof(float));
            fseek(stream, offset, SEEK_SET);

            /* Read directly into the correct column of the output buffer.
             * buffer layout: buffer[(conn_idx * nr_sub) + sub_idx]
             * We need a temporary array, then scatter.                      */
            float *tmp = (float *)malloc(n_vals * sizeof(float));
            if (!tmp) { fprintf(stderr, "OOM in readRowsFromOpenFiles\n"); exit(1); }
            size_t nread_vals = fread(tmp, sizeof(float), n_vals, stream);
            if (nread_vals != n_vals)
                fprintf(stderr, "WARNING: subject %d: expected %zu vals, got %zu\n",
                        sub_idx, n_vals, nread_vals);

            for (size_t r = 0; r < n_vals; r++)
                buffer[(r * nr_sub) + sub_idx] = tmp[r];

            free(tmp);
        }
        else
        {
            /* Text path: unchanged getline/strtof parser */
            rewind(stream);
            char *line = NULL;
            size_t len = 0;
            ssize_t nread;

            size_t k = 0;
            size_t row_counter = 0;

            while ((nread = getline(&line, &len, stream)) != -1) 
            {
                char* ptr = line;
                char* end;

                while (*ptr != '\0' && *ptr != '\n') 
                {
                    while (*ptr == ' ' || *ptr == '\t') ptr++;
                    if (*ptr == '\0' || *ptr == '\n') break;

                    float val = strtof(ptr, &end);
                    if (ptr == end) break;

                    if (k >= N && k <= M)
                    {
                        buffer[(row_counter * nr_sub) + sub_idx] = val;
                        row_counter++;
                    }

                    k++;
                    if (k > M) break;
                    ptr = end;
                }
                if (k > M) break;
            }
            free(line);
        }
    }
}

///
/// @brief Helper function to parse a single subject file (thread-safe)
/// @param filepath path to subject file
/// @param sub_idx subject index in the buffer
/// @param nr_sub total number of subjects
/// @param N from (and including) index 
/// @param M to (and including) index
/// @param buffer output buffer of subject connection values
///
void parseSingleSubjectFile(const char* filepath, int sub_idx, int nr_sub, size_t N, size_t M, float* buffer)
{
    FILE *stream;
    char *line = NULL;
    size_t len = 0;
    ssize_t nread;
    
    stream = fopen(filepath, "r");
    if (stream == NULL) 
    {
        fprintf(stderr, "Error opening file: %s\n", filepath);
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    size_t k = 0;
    size_t row_counter = 0;
    
    // Pre-allocate line buffer for better performance
    size_t line_buffer_size = 1024 * 1024; // 1MB line buffer
    if (line_buffer_size < len) line_buffer_size = len;
    
    while ((nread = getline(&line, &len, stream)) != -1) 
    {
        // Use strtof for faster parsing with pointer advancement
        char* ptr = line;
        char* end;
        
        while (*ptr != '\0' && *ptr != '\n') 
        {
            // Skip whitespace
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            if (*ptr == '\0' || *ptr == '\n') break;
            
            // Parse float
            float val = strtof(ptr, &end);
            
            // Check if parsing succeeded
            if (ptr == end) break;
            
            // Store value if in range
            if (k >= N && k <= M)
            {
                buffer[(row_counter*nr_sub)+sub_idx] = val;
                row_counter++;
            }
            
            k++;
            ptr = end;
        }
    }
    
    fclose(stream);
    free(line);
}

///
/// @brief reads file list and parses all subjects from index N to M
/// @param filename input file list
/// @param nr_sub number of subjects in total in file list
/// @param N from (and including) index 
/// @param M to (and including) index
/// @param buffer output buffer of subject connection values
///
/// Output buffer contains index N as first row to index M as last row of
/// values. For each row subject order is the same as specified in the file
/// list.
///
void parseFileListNtoM(char* filename, int nr_sub, size_t N, size_t M, float* buffer)
{
    /* variables to load file list */
    FILE *fl;
    char* fl_line = NULL;
    size_t fl_len = 0;

    fl = fopen(filename, "r");
    if (fl == NULL) 
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    // First, read all filenames into an array
    char** filenames = (char**)malloc(sizeof(char*) * nr_sub);
    size_t sub_idx = 0;
    
    while(getline(&fl_line, &fl_len, fl) != -1 && sub_idx < nr_sub)
    {
        fl_line[strcspn(fl_line, "\n")] = 0;
        filenames[sub_idx] = strdup(fl_line);
        sub_idx++;
    }
    
    fclose(fl);
    free(fl_line);

    printf("  Parsing %zu subject files in parallel...\n", sub_idx);
    fflush(stdout);

    // Parse files in parallel using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < nr_sub; i++)
    {
        #pragma omp critical
        {
            printf("  Loading subject %d/%d: %s\n", i+1, nr_sub, filenames[i]);
            fflush(stdout);
        }
        
        parseSingleSubjectFile(filenames[i], i, nr_sub, N, M, buffer);
    }

    // Cleanup filenames
    for(int i = 0; i < nr_sub; i++)
    {
        free(filenames[i]);
    }
    free(filenames);

    printf("done!\n");
    fflush(stdout);
}

///
/// @brief save results as upper triangular format
/// @param outputData data to be saved to file
/// @param nrows number of rows in the output file/nr voxels the output should have
/// @param fileName output file name
///
/// note that nrows specifies the number of voxels, not connections each subject
/// has.
///
void saveResToText(float *outputData, size_t nrows, char *fileName)
{
    FILE *output = fopen(fileName, "w");
    
    size_t c = 0;
    for (size_t i=0; i<nrows; ++i)
    {
        for (size_t j=i+1; j<nrows; ++j) 
        {
            fprintf(output, "%f ", outputData[c]);
            c++;
        }
        fprintf(output, "\n");
    }

    printf("[DBG] saved a total of %zu values\n", c);
    fclose(output);
}

///
/// @brief Save results as binary permout file (header + raw float32 array)
/// @param outputData full upper-triangular result array (n_elem floats)
/// @param nrows number of voxels (gV)
/// @param fileName output file name
///
void saveResToBinary(float *outputData, size_t nrows, char *fileName)
{
    FILE *output = fopen(fileName, "wb");
    if (!output) { perror("fopen (binary output)"); exit(EXIT_FAILURE); }

    uint32_t magic   = PERMOUT_MAGIC;
    uint32_t version = CCMAT_VERSION;
    uint64_t gV      = (uint64_t)nrows;
    uint64_t n_elem  = (uint64_t)(nrows * (nrows - 1) / 2);

    fwrite(&magic,   sizeof(uint32_t), 1, output);
    fwrite(&version, sizeof(uint32_t), 1, output);
    fwrite(&gV,      sizeof(uint64_t), 1, output);
    fwrite(&n_elem,  sizeof(uint64_t), 1, output);

    /* write the full float array in one call */
    size_t written = fwrite(outputData, sizeof(float), n_elem, output);
    if (written != n_elem)
        fprintf(stderr, "WARNING: saveResToBinary: wrote %zu / %zu floats\n", written, n_elem);

    fclose(output);
    printf("[DBG] binary: saved %zu values (%.3f GiB) to %s\n",
           n_elem, (double)(n_elem * sizeof(float)) / (1024.0*1024.0*1024.0), fileName);
}

///
/// @brief Count how many values already exist in a binary permout file
/// @param fileName output file name
/// @param nrows expected gV
/// @return n_elem from header, or 0 if file missing / wrong magic
///
size_t countExistingResultsBinary(char *fileName, size_t nrows)
{
    FILE *input = fopen(fileName, "rb");
    if (!input)
    {
        printf("[RESUME] No existing binary output file found. Starting from scratch.\n");
        fflush(stdout);
        return 0;
    }

    uint32_t magic = 0, version = 0;
    uint64_t gV = 0, n_elem = 0;
    fread(&magic,   sizeof(uint32_t), 1, input);
    fread(&version, sizeof(uint32_t), 1, input);
    fread(&gV,      sizeof(uint64_t), 1, input);
    fread(&n_elem,  sizeof(uint64_t), 1, input);
    fclose(input);

    if (magic != PERMOUT_MAGIC)
    {
        printf("[RESUME] Existing file has wrong magic (0x%08X). Starting from scratch.\n", magic);
        fflush(stdout);
        return 0;
    }

    printf("[RESUME] Found existing binary output: gV=%zu, n_elem=%zu\n", (size_t)gV, (size_t)n_elem);
    fflush(stdout);
    return (size_t)n_elem;
}

///
/// @brief Append partial float results to an open binary permout file
/// @param output  already-open binary file (positioned at EOF)
/// @param outputData partial result array (part_vals[p] floats)
/// @param n_vals number of floats to write
///
void appendPartialResultsBinary(FILE *output, float *outputData, size_t n_vals)
{
    size_t written = fwrite(outputData, sizeof(float), n_vals, output);
    if (written != n_vals)
        fprintf(stderr, "WARNING: appendPartialResultsBinary: wrote %zu / %zu floats\n", written, n_vals);
}

///
/// @brief Count how many values are already in an existing output file
/// @param fileName output file name
/// @param nrows expected number of rows (voxels)
/// @return number of values already computed, or 0 if file doesn't exist
///
size_t countExistingResults(char *fileName, size_t nrows)
{
    FILE *input = fopen(fileName, "r");
    if (input == NULL) 
    {
        printf("[RESUME] No existing output file found. Starting from scratch.\n");
        fflush(stdout);
        return 0;
    }

    printf("[RESUME] Found existing output file. Counting completed values...\n");
    fflush(stdout);

    size_t count = 0;
    char *line = NULL;
    size_t len = 0;
    ssize_t nread;
    
    while ((nread = getline(&line, &len, input)) != -1) 
    {
        char* ptr = line;
        char* end;
        
        while (*ptr != '\0' && *ptr != '\n') 
        {
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            if (*ptr == '\0' || *ptr == '\n') break;
            
            strtof(ptr, &end);
            if (ptr == end) break;
            
            count++;
            ptr = end;
        }
    }
    
    free(line);
    fclose(input);
    
    printf("[RESUME] Found %zu existing values in output file.\n", count);
    fflush(stdout);
    return count;
}

///
/// @brief Append partial results to output file
/// @param outputData partial results data (just this part)
/// @param start_idx starting connection index
/// @param end_idx ending connection index (inclusive)
/// @param nrows total number of voxels
/// @param fileName output file name
/// @param is_first_write whether this is the first write (use "w" vs "a")
///
void appendPartialResults(float *outputData, size_t start_idx, size_t end_idx, 
                          size_t nrows, char *fileName, int is_first_write)
{
    FILE *output = fopen(fileName, is_first_write ? "w" : "a");
    if (output == NULL) 
    {
        perror("Failed to open output file");
        exit(EXIT_FAILURE);
    }
    
    size_t c = 0;  // Overall connection counter
    size_t local_idx = 0;  // Index into outputData array
    size_t written = 0;
    int in_range = 0;
    
    // Iterate through upper triangular matrix
    for (size_t i = 0; i < nrows; ++i)
    {
        for (size_t j = i+1; j < nrows; ++j) 
        {
            if (c >= start_idx && c <= end_idx) 
            {
                if (!in_range) in_range = 1;
                fprintf(output, "%f ", outputData[local_idx]);
                written++;
                local_idx++;
            }
            c++;
        }
        
        // Add newline if we wrote anything on this row
        if (in_range && (c > end_idx || i == nrows - 1))
        {
            fprintf(output, "\n");
            if (c > end_idx) break;
        }
    }
    
    fclose(output);
    printf("[SAVE] Appended %zu values (indices %zu-%zu) to %s\n", 
           written, start_idx, end_idx, fileName);
    fflush(stdout);
}

///
/// @brief CUDA kernel to permform permutation test
/// @param input input buffer (subject values)
/// @param onehot permutations buffer in one-hot format
/// @param nr_vals number of values to process (connections)
/// @param nr_sub number of subjects in total
/// @param nr_perm number of permutations
/// @param output_pval output buffer for p-values
/// @param output_tstat output buffer for t-statistics
/// @param two_tailed is two-tailed test or not (1 = two-tailed, 0 = single-tailed)
///
/// OPTIMIZED VERSION: Parallelizes across both connections AND permutations
/// Each block processes one connection, multiple threads within block process permutations
///
__global__ 
void CUDA_perm(float *input, int* onehot, 
               size_t nr_vals, size_t nr_sub, size_t nr_perm, 
               float *output_pval, float *output_tstat, int two_tailed)
{
    int n = blockIdx.x;  // Connection index
    int tid = threadIdx.x;  // Thread within block (for permutation parallelization)
    int block_size = blockDim.x;
    
    if (n >= nr_vals) return;

    // Shared memory for reduction of p_val counts across threads
    extern __shared__ float shared_pval[];
    
    float t_obs = 0.;
    float local_pval = 0.;

    // First permutation is computed by first thread to get observed t-statistic
    if (tid == 0) 
    {
        float a_mean = 0.;
        float b_mean = 0.;
        float nA = 0;
        float nB = 0;
        
        for (int j=0; j<nr_sub; ++j)
        {
            if (onehot[j] == 0)
            {
                b_mean += input[(n*nr_sub)+j];
                nB++;
            } 
            else 
            {
                a_mean += input[(n*nr_sub)+j];
                nA++;
            }
        }
        if (nA > 0) a_mean /= nA;
        if (nB > 0) b_mean /= nB;
        t_obs = a_mean - b_mean;
    }
    
    // Broadcast t_obs to all threads in the block
    if (tid == 0) shared_pval[0] = t_obs;
    __syncthreads();
    t_obs = shared_pval[0];

    // Each thread processes a subset of permutations (starting from perm 1, not 0)
    for (int i = 1 + tid; i < nr_perm; i += block_size) 
    {
        float a_mean = 0.;
        float b_mean = 0.;
        float nA = 0;
        float nB = 0;
        float tstat = 0.0;
        
        for (int j=0; j<nr_sub; ++j)
        {
            if (onehot[(i*nr_sub)+j] == 0)
            {
                b_mean += input[(n*nr_sub)+j];
                nB++;
            } 
            else 
            {
                a_mean += input[(n*nr_sub)+j];
                nA++;
            }
        }
        if (nA > 0) a_mean /= nA;
        if (nB > 0) b_mean /= nB;
        tstat = a_mean - b_mean;

        // Count if permuted test statistic is more extreme
        if (two_tailed == 1) 
        {
            if (fabsf(tstat) >= fabsf(t_obs)) 
            {
                local_pval++;
            }
        }
        else 
        {
            if (tstat >= t_obs) 
            {
                local_pval++;
            }
        }
    }
    
    // Store local counts in shared memory for reduction
    shared_pval[tid] = local_pval;
    __syncthreads();
    
    // Reduction: sum all thread counts (simple sequential reduction by thread 0)
    if (tid == 0) 
    {
        float total_pval = 0.;
        for (int i = 0; i < block_size; i++) 
        {
            total_pval += shared_pval[i];
        }
        // Calculate p-value
        float p_val = (total_pval + 1.0f) / (float)(nr_perm + 1);
        output_pval[n] = p_val;
        output_tstat[n] = t_obs;
    }
}

int
main(int argc, char *argv[])
{
    /* read arguments */
    if (argc < 4 || argc > 6) {
        fprintf(stderr, "Usage: %s <file list> <permutations file> <output file> [--two-tailed] [-b]\n", argv[0]);
        fprintf(stderr, "  --two-tailed : Enable two-tailed test (default: one-tailed)\n");
        fprintf(stderr, "  -b           : Write output in binary format instead of text\n");
        exit(EXIT_FAILURE);
    }

    /* parse input arguments */
    char* filelist = argv[1];
    char* permutations = argv[2];
    char* outfile = argv[3];

    /* parse optional flags */
    int two_tailed = 0;
    int binary_output = 0;
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--two-tailed") == 0) {
            two_tailed = 1;
        } else if (strcmp(argv[i], "-b") == 0) {
            binary_output = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s <file list> <permutations file> <output file> [--two-tailed] [-b]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    printf("========================================\n");
    printf("permutationTest_cuda\n");
    printf("========================================\n");
    printf("  File list    : %s\n", filelist);
    printf("  Permutations : %s\n", permutations);
    printf("  Output       : %s\n", outfile);
    printf("  Test type    : %s\n", two_tailed ? "two-tailed" : "one-tailed");
    printf("  Output format: %s\n", binary_output ? "binary (-b)" : "text");
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

        /* setting up and perform calculations on gpu */
        // Optimized: Use 256 threads per block to parallelize permutation loop
        int threads_per_block = 256;
        size_t shared_mem_size = threads_per_block * sizeof(float);
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

        /* Check for existing results and determine where to resume */
        size_t existing_vals = binary_output
            ? countExistingResultsBinary(outfile, nr_r1vals)
            : countExistingResults(outfile, nr_r1vals);
        int start_part = 0;
        
        // Find which part to start from
        for (int p = 0; p < nr_parts; ++p)
        {
            if (existing_vals > part_ends[p])
            {
                start_part = p + 1;
            }
            else
            {
                break;
            }
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

        /* ===== STREAMING FILE APPROACH (NEW OPTIMIZED METHOD) ===== */
        printf("========================================\n");
        printf("OPTIMIZATION: Opening all files once, streaming data for each part\n");
        printf("========================================\n");
        fflush(stdout);
        
        // Allocate and parse permutations once (same for all parts)
        printf("Allocating and parsing permutations buffer (%zu MB)...\n", 
               (sizeof(int)*nr_perm*nr_subs)/(1024*1024));
        fflush(stdout);
        int* perm_buff = (int*)malloc(sizeof(int)*nr_perm*nr_subs);
        for (int i=0; i<nr_perm*nr_subs; ++i)
        {
            perm_buff[i] = 0; 
        }
        parsePermutations(permutations, perm_buff, nr_subs);
        printf("Permutations parsed successfully!\n");
        fflush(stdout);

        // Open all subject files ONCE and keep them open
        printf("========================================\n");
        printf("Opening all subject files (happens ONCE)...\n");
        printf("========================================\n");
        fflush(stdout);
        FileHandleArray* open_files = openAllSubjectFiles(filelist, nr_subs);
        printf("========================================\n");
        printf("All files are now open and ready for streaming!\n");
        printf("========================================\n");
        fflush(stdout);
        /* ============================================== */

        /* For binary output: open both output files once and write the header
         * on the first write (or skip header if resuming).                   */
        FILE *out_pval_bin  = NULL;
        FILE *out_tstat_bin = NULL;
        if (binary_output) {
            int resuming = (start_part > 0);
            const char *open_mode = resuming ? "r+b" : "wb";

            out_pval_bin  = fopen(outfile,       open_mode);
            out_tstat_bin = fopen(outfile_tstat, open_mode);
            if (!out_pval_bin || !out_tstat_bin) {
                perror("fopen binary output"); exit(EXIT_FAILURE);
            }

            if (!resuming) {
                /* Write header for a fresh file (gV and n_elem are now known) */
                uint32_t magic   = PERMOUT_MAGIC;
                uint32_t version = CCMAT_VERSION;
                uint64_t gV      = (uint64_t)nr_r1vals;
                uint64_t n_elem  = (uint64_t)nr_vals;
                for (FILE *fp : {out_pval_bin, out_tstat_bin}) {
                    fwrite(&magic,   sizeof(uint32_t), 1, fp);
                    fwrite(&version, sizeof(uint32_t), 1, fp);
                    fwrite(&gV,      sizeof(uint64_t), 1, fp);
                    fwrite(&n_elem,  sizeof(uint64_t), 1, fp);
                }
            } else {
                /* Seek to EOF to append after already-written floats */
                fseek(out_pval_bin,  0, SEEK_END);
                fseek(out_tstat_bin, 0, SEEK_END);
            }
        }

        /* performing calculation in parts */

        // device buffers
        int *d_perm;
        float *d_input;
        float *d_output;
        float *d_output_tstat;

        // counters
        cudaError_t err = cudaSuccess;

        for (int p=start_part; p<nr_parts; ++p)
        {
            printf("========================================\n");
            printf("Processing part %i of %i\n", p+1, nr_parts);
            printf("========================================\n");
            fflush(stdout);

            // Allocate buffer for this part only
            printf("Allocating buffer for part %i (%zu connections)...\n", p+1, part_vals[p]);
            fflush(stdout);
            float* device_buff = (float*)malloc(sizeof(float)*part_vals[p]*nr_subs);
            
            // Stream data from open files for this part
            printf("Streaming data for part %i from open files (rows %zu to %zu)...\n", 
                   p+1, part_starts[p], part_ends[p]);
            fflush(stdout);
            readRowsFromOpenFiles(open_files, part_starts[p], part_ends[p], nr_subs, device_buff);
            printf("Data streaming complete for part %i!\n", p+1);
            fflush(stdout);

            float* perm_test_res = (float*)malloc(sizeof(float)*part_vals[p]);
            float* perm_test_tstat = (float*)malloc(sizeof(float)*part_vals[p]);

            printf("Allocating GPU memory for part %i...\n", p+1);
            fflush(stdout);
            /* allocating gpu mem */
            err = cudaMalloc((void **)&d_perm, sizeof(int)*nr_perm*nr_subs);
            err = cudaMalloc((void **)&d_input, sizeof(float)*part_vals[p]*nr_subs);
            err = cudaMalloc((void **)&d_output, sizeof(float)*part_vals[p] );
            err = cudaMalloc((void **)&d_output_tstat, sizeof(float)*part_vals[p] );
            if (err!=cudaSuccess)
            {
                printf("CUDA ERROR! Failed to allocate device memory! (error code %s)\n", cudaGetErrorString(err));
            }
            
            printf("Copying data from CPU to GPU for part %i...\n", p+1);
            fflush(stdout);
            /* copy input data host -> device */
            err = cudaMemcpy(d_input, device_buff, sizeof(float)*part_vals[p]*nr_subs, cudaMemcpyHostToDevice);
            err = cudaMemcpy(d_perm, perm_buff, sizeof(int)*nr_perm*nr_subs, cudaMemcpyHostToDevice);
            if (err!=cudaSuccess)
            {
                printf("CUDA ERROR! Failed to copy memory from host to device (error code %s)\n", cudaGetErrorString(err));
            }

            printf("Running permutation tests on GPU for part %i...\n", p+1);
            fflush(stdout);
            /* setting up and perform calculations on gpu */
            // Optimized: Use 256 threads per block to parallelize permutation loop
            int threads_per_block = 256;
            size_t shared_mem_size = threads_per_block * sizeof(float);
            printf("  Launching %zu blocks × %d threads (%.2f million GPU threads)...\n", 
                   part_vals[p], threads_per_block, (part_vals[p] * threads_per_block) / 1e6);
            fflush(stdout);
            CUDA_perm<<<part_vals[p], threads_per_block, shared_mem_size>>>(d_input, d_perm, part_vals[p], nr_subs, nr_perm, d_output, d_output_tstat, two_tailed);
            
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
            
            printf("Copying results from GPU to CPU for part %i...\n", p+1);
            fflush(stdout);
            /* copy results from the device back to host */
            err = cudaMemcpy(perm_test_res, d_output, sizeof(float)*part_vals[p], cudaMemcpyDeviceToHost);
            err = cudaMemcpy(perm_test_tstat, d_output_tstat, sizeof(float)*part_vals[p], cudaMemcpyDeviceToHost);
            if (err!=cudaSuccess)
            {
                printf("CUDA ERROR! Failed to copy memory from device to host (error code %s)\n", cudaGetErrorString(err));
            }

            /* Save partial results immediately after computation */
            printf("Saving p-values for part %i to disk...\n", p+1);
            fflush(stdout);
            if (binary_output) {
                appendPartialResultsBinary(out_pval_bin,  perm_test_res,   part_vals[p]);
                appendPartialResultsBinary(out_tstat_bin, perm_test_tstat, part_vals[p]);
                fflush(out_pval_bin);
                fflush(out_tstat_bin);
            } else {
                int is_first_write = (p == 0 && start_part == 0) ? 1 : 0;
                appendPartialResults(perm_test_res,   part_starts[p], part_ends[p],
                                     nr_r1vals, outfile,       is_first_write);
                printf("Saving t-statistics for part %i to disk...\n", p+1);
                fflush(stdout);
                appendPartialResults(perm_test_tstat, part_starts[p], part_ends[p],
                                     nr_r1vals, outfile_tstat, is_first_write);
            }

            printf("Part %i complete! Cleaning up GPU memory...\n", p+1);
            fflush(stdout);
            // cleanup part-specific buffers
            free(device_buff);
            free(perm_test_res);
            free(perm_test_tstat);

            err = cudaFree(d_input);
            err = cudaFree(d_output);
            err = cudaFree(d_output_tstat);
            err = cudaFree(d_perm);
            if (err!=cudaSuccess)
            {
                printf("CUDA ERROR! Failed to free device memory! (error code %s)\n", cudaGetErrorString(err));
            }
            printf("Part %i memory cleaned up.\n", p+1);
            fflush(stdout);
        }

        printf("========================================\n");
        printf("All %i parts completed!\n", nr_parts);
        printf("Closing all open files...\n");
        printf("========================================\n");
        fflush(stdout);
        
        // Close all open files
        closeAllSubjectFiles(open_files);

        // Close binary output handles (if used)
        if (binary_output) {
            if (out_pval_bin)  fclose(out_pval_bin);
            if (out_tstat_bin) fclose(out_tstat_bin);
        }
        
        // Cleanup global buffers
        free(perm_buff);
        
        printf("P-values saved to: %s\n", outfile);
        printf("T-statistics saved to: %s\n", outfile_tstat);
        printf("done!\n");
        fflush(stdout);
    }

    // Cleanup output filename buffer
    free(outfile_tstat);

    return(EXIT_SUCCESS);
}