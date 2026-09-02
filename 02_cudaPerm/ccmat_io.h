/*
 * ccmat_io.h  —  Binary ccmat/permout format constants and subject file I/O
 */
#ifndef CCMAT_IO_H
#define CCMAT_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── Binary format constants ────────────────────────────────────────────────
 *
 * INPUT  (.ccmat):   produced by cudaCC_div
 * OUTPUT (.permout): produced by permutationTest_cuda_fwer when -b is given
 *
 * Header layout (24 bytes):
 *   Offset  Size  Field
 *    0       4    magic   0x43434D54 ("CCMT") [ccmat]
 *                         0x50455254 ("PERT") [permout]
 *    4       4    version (uint32, currently 1)
 *    8       8    gV      (uint64) — number of voxels (NOT connections)
 *   16       8    n_elem  = gV*(gV-1)/2  (uint64)
 *   24       n_elem * 4   upper-triangular float32, row-major
 * ─────────────────────────────────────────────────────────────────────────── */

/* These #defines must live OUTSIDE any extern "C" block so nvcc sees them
 * during device-code compilation. */
#define CCMAT_MAGIC   0x43434D54u   /* input  ccmat  files ("CCMT") */
#define PERMOUT_MAGIC 0x50455254u   /* output permout files ("PERT") */
#define CCMAT_VERSION 1u
#define CCMAT_HDR_SIZE 24           /* bytes before the float data   */

/* Struct definition also outside extern "C" for C++ type visibility. */
typedef struct {
    FILE** file_handles;
    size_t nr_files;
} FileHandleArray;

/* Function declarations use extern "C" so C translation units link correctly. */
#ifdef __cplusplus
extern "C" {
#endif

/* ── Binary format helpers ── */
int      isBinaryCCmat(const char* filename);
uint64_t binaryGetGV(const char* filename);

/* ── Dimension queries ── */
size_t getNumberLines(char* filename);
size_t getNumberValsFirstLine(char* filename);
size_t peekFileList(char* filelist);

/* ── Permutation matrix parsing ── */
void parsePermutations(char* filename, int* buffer, size_t nr_subs);

/* ── Streaming I/O (open files once, seek per chunk) ── */
FileHandleArray* openAllSubjectFiles(char* filelist, size_t nr_subs);
void             closeAllSubjectFiles(FileHandleArray* fha);
void             readRowsFromOpenFiles(FileHandleArray* fha, size_t N, size_t M,
                                       int nr_sub, float* buffer);
/* Same rows, transposed: buffer[sub_idx * n_vals + r] instead of
 * buffer[r * nr_sub + sub_idx]. Subject-major lets a CUDA thread per edge read
 * coalesced across a warp, which the edge-major layout cannot do. */
void             readRowsSubjectMajor(FileHandleArray* fha, size_t N, size_t M,
                                      int nr_sub, float* buffer);

/* ── Batch parser (opens/closes per call; used by single-chunk path) ── */
void parseSingleSubjectFile(const char* filepath, int sub_idx, int nr_sub,
                            size_t N, size_t M, float* buffer);
void parseFileListNtoM(char* filename, int nr_sub,
                       size_t N, size_t M, float* buffer);

#ifdef __cplusplus
}
#endif

#endif /* CCMAT_IO_H */
