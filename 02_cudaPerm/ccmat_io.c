/*
 * ccmat_io.c  —  Implementation of ccmat/permout format I/O helpers
 */
#include "ccmat_io.h"

#include <math.h>
#include <float.h>
#include <omp.h>

/* ── Binary format helpers ─────────────────────────────────────────────────── */

/* Returns 1 if the file starts with the ccmat binary magic number, 0 otherwise. */
int isBinaryCCmat(const char* filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f) return 0;
    uint32_t magic = 0;
    fread(&magic, sizeof(uint32_t), 1, f);
    fclose(f);
    return (magic == CCMAT_MAGIC);
}

/* Read gV (number of voxels) from the binary header. */
uint64_t binaryGetGV(const char* filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen"); exit(EXIT_FAILURE); }
    fseek(f, 8, SEEK_SET);   /* skip magic + version */
    uint64_t gV = 0;
    fread(&gV, sizeof(uint64_t), 1, f);
    fclose(f);
    return gV;
}

/* ── Dimension queries ──────────────────────────────────────────────────────── */

///
/// @brief Returns number of rows a ccmat file has (= number of voxels gV).
/// @param filename  input ccmat file (text or binary)
/// @return number of rows = gV
///
size_t getNumberLines(char* filename)
{
    /* Binary format: gV is stored directly in the header */
    if (isBinaryCCmat(filename))
        return (size_t)binaryGetGV(filename);

    /* Text format: count newlines */
    FILE *stream = fopen(filename, "r");
    if (stream == NULL) { perror("fopen"); exit(EXIT_FAILURE); }

    size_t nlines = 0;
    int ch;
    while ((ch = fgetc(stream)) != EOF)
        if (ch == '\n') nlines++;

    fclose(stream);
    return nlines;
}

///
/// @brief Returns the number of values in the first row of a ccmat file (= gV - 1).
/// @param filename  input ccmat file (text or binary)
/// @return number of values in row 0
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
    FILE *stream = fopen(filename, "r");
    if (stream == NULL) { perror("fopen"); exit(EXIT_FAILURE); }

    char *line = NULL;
    size_t len = 0;
    size_t line_length = 0;

    if (getline(&line, &len, stream) != -1)
    {
        char* ptr = line;
        char* end;
        while (*ptr != '\0' && *ptr != '\n')
        {
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            if (*ptr == '\0' || *ptr == '\n') break;
            strtof(ptr, &end);
            if (ptr == end) break;
            line_length++;
            ptr = end;
        }
    }

    fclose(stream);
    free(line);
    return line_length;
}

///
/// @brief Peeks into the file list and returns the line count of the first subject file.
/// @param filelist  path to the file list
/// @return gV of the first subject file
///
size_t peekFileList(char* filelist)
{
    FILE *fl = fopen(filelist, "r");
    if (fl == NULL) { perror("fopen"); exit(EXIT_FAILURE); }

    char* fl_line = NULL;
    size_t fl_len = 0;
    size_t output = 0;

    printf("[peekFileList] Opening first subject file to determine dimensions...\n");
    fflush(stdout);

    while (getline(&fl_line, &fl_len, fl) != -1)
    {
        fl_line[strcspn(fl_line, "\n")] = 0;
        printf("[peekFileList] Attempting to open: %s\n", fl_line);
        fflush(stdout);
        output = getNumberLines(fl_line);
        printf("[peekFileList] File has %zu lines\n", output);
        fflush(stdout);
        break; /* only need the first file */
    }

    fclose(fl);
    free(fl_line);
    return output;
}

/* ── Permutation matrix parsing ─────────────────────────────────────────────── */

///
/// @brief Parses a permutations file into a one-hot integer buffer.
/// @param filename  output of generatePermutations.py (group-A indices per line)
/// @param buffer    zero-initialised int array of size [nr_perm * nr_subs]
/// @param nr_subs   total number of subjects
///
/// Input rows contain indices of group A; output buffer is one-hot encoded
/// so that buffer[(perm * nr_subs) + sub] == 1 iff subject sub is in group A
/// for permutation perm.
///
void parsePermutations(char* filename, int* buffer, size_t nr_subs)
{
    FILE *pt = fopen(filename, "r");
    if (pt == NULL) { perror("fopen"); exit(EXIT_FAILURE); }

    char *pt_line = NULL;
    size_t pt_len = 0;
    size_t line_idx = 0;

    while (getline(&pt_line, &pt_len, pt) != -1)
    {
        char* ptr = pt_line;
        char* end;
        while (*ptr != '\0' && *ptr != '\n')
        {
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            if (*ptr == '\0' || *ptr == '\n') break;
            long val = strtol(ptr, &end, 10);
            if (ptr == end) break;
            if (val >= 0 && (size_t)val < nr_subs)
                buffer[(line_idx * nr_subs) + val] = 1;
            ptr = end;
        }
        line_idx++;
    }

    fclose(pt);
    free(pt_line);
}

/* ── Streaming file I/O ──────────────────────────────────────────────────────── */

///
/// @brief Opens all subject files listed in filelist and returns a FileHandleArray.
/// @param filelist  path to the subject file list
/// @param nr_subs   number of subjects (= number of lines in filelist)
/// @return heap-allocated FileHandleArray; caller must call closeAllSubjectFiles()
///
FileHandleArray* openAllSubjectFiles(char* filelist, size_t nr_subs)
{
    FILE *fl = fopen(filelist, "r");
    if (fl == NULL) { perror("fopen filelist"); exit(EXIT_FAILURE); }

    FileHandleArray* fha = (FileHandleArray*)malloc(sizeof(FileHandleArray));
    fha->nr_files     = nr_subs;
    fha->file_handles = (FILE**)malloc(sizeof(FILE*) * nr_subs);

    char* fl_line = NULL;
    size_t fl_len = 0;
    size_t sub_idx = 0;

    printf("Opening all %zu subject files...\n", nr_subs);
    fflush(stdout);

    while (getline(&fl_line, &fl_len, fl) != -1 && sub_idx < nr_subs)
    {
        fl_line[strcspn(fl_line, "\n")] = 0;
        fha->file_handles[sub_idx] = fopen(fl_line, isBinaryCCmat(fl_line) ? "rb" : "r");
        if (fha->file_handles[sub_idx] == NULL)
        {
            fprintf(stderr, "Error opening file %zu: %s\n", sub_idx + 1, fl_line);
            perror("fopen");
            exit(EXIT_FAILURE);
        }
        if ((sub_idx + 1) % 50 == 0 || sub_idx == 0)
        {
            printf("  Opened %zu/%zu files\n", sub_idx + 1, nr_subs);
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
/// @brief Closes all file handles in a FileHandleArray and frees the structure.
/// @param fha  FileHandleArray returned by openAllSubjectFiles()
///
void closeAllSubjectFiles(FileHandleArray* fha)
{
    printf("Closing all %zu subject files...\n", fha->nr_files);
    fflush(stdout);
    for (size_t i = 0; i < fha->nr_files; i++)
        if (fha->file_handles[i] != NULL)
            fclose(fha->file_handles[i]);
    free(fha->file_handles);
    free(fha);
    printf("All files closed.\n");
    fflush(stdout);
}

///
/// @brief Reads connection indices [N..M] from already-open subject files.
/// @param fha     FileHandleArray with open file handles
/// @param N       first flat upper-triangular index to read (inclusive)
/// @param M       last  flat upper-triangular index to read (inclusive)
/// @param nr_sub  number of subjects
/// @param buffer  output buffer, layout: buffer[(conn - N) * nr_sub + sub]
///
/// Supports text and binary ccmat files transparently.
/// Binary path: single fseek + fread per subject — orders of magnitude faster
/// than the text path.
///
void readRowsFromOpenFiles(FileHandleArray* fha, size_t N, size_t M,
                           int nr_sub, float* buffer)
{
    size_t n_vals = M - N + 1;

    #pragma omp parallel for schedule(dynamic)
    for (int sub_idx = 0; sub_idx < nr_sub; sub_idx++)
    {
        FILE* stream = fha->file_handles[sub_idx];

        /* Detect format by peeking at the magic word. */
        rewind(stream);
        uint32_t magic = 0;
        fread(&magic, sizeof(uint32_t), 1, stream);
        int is_binary = (magic == CCMAT_MAGIC);

        if (is_binary)
        {
            /* Data starts at byte CCMAT_HDR_SIZE (24).
             * Element N is at offset CCMAT_HDR_SIZE + N*sizeof(float). */
            long offset = (long)(CCMAT_HDR_SIZE + N * sizeof(float));
            fseek(stream, offset, SEEK_SET);

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
            /* Text path: scan through the file counting flat indices. */
            rewind(stream);
            char *line = NULL;
            size_t len = 0;
            size_t k = 0;
            size_t row_counter = 0;

            while (getline(&line, &len, stream) != -1)
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

/* ── Batch parser (single-chunk path) ──────────────────────────────────────── */

///
/// @brief Parses connection indices [N..M] from a single subject file (thread-safe).
/// @param filepath  path to the subject ccmat file
/// @param sub_idx   column index for this subject in the output buffer
/// @param nr_sub    total number of subjects (buffer stride)
/// @param N         first flat index (inclusive)
/// @param M         last  flat index (inclusive)
/// @param buffer    output buffer, layout: buffer[(conn - N) * nr_sub + sub_idx]
///
void parseSingleSubjectFile(const char* filepath, int sub_idx, int nr_sub,
                            size_t N, size_t M, float* buffer)
{
    FILE *stream = fopen(filepath, "r");
    if (stream == NULL)
    {
        fprintf(stderr, "Error opening file: %s\n", filepath);
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    char *line = NULL;
    size_t len = 0;
    size_t k = 0;
    size_t row_counter = 0;

    while (getline(&line, &len, stream) != -1)
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
            ptr = end;
        }
    }

    fclose(stream);
    free(line);
}

///
/// @brief Reads connection indices [N..M] for all subjects listed in a file list.
/// @param filename  path to the subject file list
/// @param nr_sub    number of subjects
/// @param N         first flat index (inclusive)
/// @param M         last  flat index (inclusive)
/// @param buffer    output buffer, layout: buffer[(conn - N) * nr_sub + sub]
///
/// Opens and closes every subject file on each call.  Used by the single-chunk
/// path where all data fits in GPU memory.  For the streaming (multi-chunk) path
/// use openAllSubjectFiles() + readRowsFromOpenFiles() instead.
///
void parseFileListNtoM(char* filename, int nr_sub,
                       size_t N, size_t M, float* buffer)
{
    FILE *fl = fopen(filename, "r");
    if (fl == NULL) { perror("fopen"); exit(EXIT_FAILURE); }

    char** filenames = (char**)malloc(sizeof(char*) * nr_sub);
    char* fl_line = NULL;
    size_t fl_len = 0;
    size_t sub_idx = 0;

    while (getline(&fl_line, &fl_len, fl) != -1 && sub_idx < (size_t)nr_sub)
    {
        fl_line[strcspn(fl_line, "\n")] = 0;
        filenames[sub_idx] = strdup(fl_line);
        sub_idx++;
    }
    fclose(fl);
    free(fl_line);

    printf("  Parsing %zu subject files in parallel...\n", sub_idx);
    fflush(stdout);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nr_sub; i++)
    {
        #pragma omp critical
        {
            printf("  Loading subject %d/%d: %s\n", i + 1, nr_sub, filenames[i]);
            fflush(stdout);
        }
        parseSingleSubjectFile(filenames[i], i, nr_sub, N, M, buffer);
    }

    for (int i = 0; i < nr_sub; i++) free(filenames[i]);
    free(filenames);

    printf("done!\n");
    fflush(stdout);
}
