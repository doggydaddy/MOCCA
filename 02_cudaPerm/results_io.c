/*
 * results_io.c  —  Implementation of permutation test output file helpers
 */
#include "results_io.h"

/* ── Full-result saves ──────────────────────────────────────────────────────── */

///
/// @brief Saves the full result array in upper-triangular text format.
/// @param outputData  flat upper-triangular array (n_elem floats)
/// @param nrows       number of voxels (gV); determines the triangular shape
/// @param fileName    output file path
///
void saveResToText(float *outputData, size_t nrows, char *fileName)
{
    FILE *output = fopen(fileName, "w");
    if (!output) { perror("fopen (text output)"); exit(EXIT_FAILURE); }

    size_t c = 0;
    for (size_t i = 0; i < nrows; ++i)
    {
        for (size_t j = i + 1; j < nrows; ++j)
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
/// @brief Saves the full result array as a binary permout file.
/// @param outputData  flat upper-triangular array (n_elem floats)
/// @param nrows       number of voxels (gV)
/// @param fileName    output file path
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

    size_t written = fwrite(outputData, sizeof(float), n_elem, output);
    if (written != n_elem)
        fprintf(stderr, "WARNING: saveResToBinary: wrote %zu / %zu floats\n",
                written, n_elem);

    fclose(output);
    printf("[DBG] binary: saved %zu values (%.3f GiB) to %s\n",
           n_elem, (double)(n_elem * sizeof(float)) / (1024.0*1024.0*1024.0),
           fileName);
}

/* ── Resume detection ───────────────────────────────────────────────────────── */

///
/// @brief Counts how many values are already in an existing text output file.
/// @param fileName  output file path
/// @param nrows     expected number of voxels (unused beyond the signature)
/// @return number of float values found, or 0 if the file does not exist
///
size_t countExistingResults(char *fileName, size_t nrows)
{
    (void)nrows; /* parameter kept for API symmetry */
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

    while (getline(&line, &len, input) != -1)
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
/// @brief Counts how many values are declared in an existing binary permout file.
/// @param fileName  output file path
/// @param nrows     expected gV (used only for diagnostic consistency checks)
/// @return n_elem from the file header, or 0 if the file is missing / wrong magic
///
size_t countExistingResultsBinary(char *fileName, size_t nrows)
{
    (void)nrows;
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
        printf("[RESUME] Existing file has wrong magic (0x%08X). Starting from scratch.\n",
               magic);
        fflush(stdout);
        return 0;
    }

    printf("[RESUME] Found existing binary output: gV=%zu, n_elem=%zu\n",
           (size_t)gV, (size_t)n_elem);
    fflush(stdout);
    return (size_t)n_elem;
}

/* ── Partial appends ────────────────────────────────────────────────────────── */

///
/// @brief Appends a contiguous slice of flat-index results to a text output file.
/// @param outputData    result array for this slice (end_idx - start_idx + 1 values)
/// @param start_idx     first flat upper-triangular index in this slice
/// @param end_idx       last  flat upper-triangular index in this slice (inclusive)
/// @param nrows         total number of voxels (gV); defines the row structure
/// @param fileName      output file path
/// @param is_first_write  1 → open with "w" (truncate), 0 → open with "a" (append)
///
void appendPartialResults(float *outputData, size_t start_idx, size_t end_idx,
                          size_t nrows, char *fileName, int is_first_write)
{
    FILE *output = fopen(fileName, is_first_write ? "w" : "a");
    if (output == NULL) { perror("Failed to open output file"); exit(EXIT_FAILURE); }

    size_t c         = 0;   /* overall flat connection counter */
    size_t local_idx = 0;   /* index into outputData           */
    size_t written   = 0;
    int    in_range  = 0;

    for (size_t i = 0; i < nrows; ++i)
    {
        for (size_t j = i + 1; j < nrows; ++j)
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
        /* Emit a newline at the end of each row that overlaps the slice. */
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
/// @brief Appends raw floats to an already-open binary permout file.
/// @param output      open file handle positioned at the write point (typically EOF)
/// @param outputData  float array to write
/// @param n_vals      number of floats to write
///
void appendPartialResultsBinary(FILE *output, float *outputData, size_t n_vals)
{
    size_t written = fwrite(outputData, sizeof(float), n_vals, output);
    if (written != n_vals)
        fprintf(stderr,
                "WARNING: appendPartialResultsBinary: wrote %zu / %zu floats\n",
                written, n_vals);
}
