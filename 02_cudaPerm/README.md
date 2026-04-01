# CUDA accelerated permutation testing

This subroutines performs connection-wise permutation tests.

The statistical tests are performed independently for each voxel, so the tests
can be greatly accelerated using GPU.

## ⚠️ IMPORTANT: Recent Bug Fix (March 2026)

A **critical bug** in the p-value calculation was fixed on March 7, 2026. If you have existing results computed before this date, they may be incorrect and should be rerun.

**What was fixed:**
- Permutation counting logic was incorrect
- Two-tailed tests were not working properly
- Many results showed exact zero p-values incorrectly

**Action required:** 
- Rerun analyses performed before March 7, 2026
- See `CHANGELOG.md` for complete details on all bug fixes and improvements

## ⚡ Performance (March 29, 2026)

The CUDA kernel has been **completely reworked** to exploit GPU parallelism properly.

**Previous behaviour:** 1 thread per block — each thread processed all permutations sequentially. GPU utilization was < 1%.

**Current behaviour:** 256 threads per block — permutation loop is divided across threads with a shared-memory reduction. GPU utilization is ~95%.

| Dataset | Parts | Before | After |
|---------|-------|--------|-------|
| 37 subjects, 1M perms | 185 | ~277 days | ~1.5–2 days |
| 257 subjects, 1M perms | 1299 | years | ~2–3 days |

See `CHANGELOG.md` [v3.2] for the full technical details.

# Details

For each test, all N subjects included in the tests must be loaded into memory.
For easy memory management, prior to performing the tests, the permutations are
generated *a-priori*.

## Generating permutations

The permutations are generated using a python program *createPerm.py*. It
generates one-hot labels of $n$ permutations of nA and nB subjects, where $nA$
and $nB$ is the number of subjects in group A and group B respectively.

Calling the program is simple:

        python createPerm.py <nr. permutations> <nA> <nB> <output txt file>

**Note** that the original statistical test groupings is NOT known to the
program. So the first row of the generated text file must be replaced with the
original labels, otherwise the test will not be as intended! The ordering is
from left-to-right from the top-to-down order of appearance in *filelist.txt*

## Performing permutation tests

### Parsing file list

The file list is a simple list of files to be processed in a single column with
one file name per row. The order does matter as the intended statistical test
groupings must be in order of the *first* row in the permutations file. 

For example, the filelist may be initiated using:

        find . -name 'groupA_subj*_connectivityMatrix.txt' | sort > filelist.txt
        find . -name 'groupB_subj*_connectivityMatrix.txt' | sort >> filelist.txt

The program *parseFileList* parses all subjects in the *filelist* from *start
index* to *end index*, and outputs it into a separate *output file*.

        parseFileList <filelist> <start index> <end index> <output file>

Each row contains the subject voxel data from start to end index, in order
left-to-right according to the input file lists top-to-down order.

# Simulated data for testing

The iPython notebook *genTestData.ipynb* can be used to generate simulated data
for testing purposes. Simulated data with their names can be found and changed
in *subject_list*. *N* is the number of simulated voxels (not connections).
*subject_list_A* contains a sub-list, where each subject have certain
connections artificially truncated to above a threshold (0.8). Otherwise all
simulated connection values are uniformly random between \[-1,1\].

# Generating permutations

Prior to running permutations tests, a separate routine needs to be called to
pre-generate permutations to be used for calculations. This is done to reduce
complexity in the actual calculations program. Simply call the python program
*createPerm.py* to generate permutations:

                python createPerm.py -nPerm <nP> -nA <X> -nB <Y> -o <output permutations txt file>
                
Where *nP* is the number of permutations (e.g 5000), *X* and *Y* are number of
subjects in group A and group B respectively.

Note that the output is the indices of one of the groups (group A) for each
permutation, and NOT one-hot labelling of the groups. This is done to save
space, and permutation calculation routines expects the input to be in this way.

# Running permutation tests

Performing permutation tests can be done simply by calling:

                ./<permutation_test_prog> <file list> <permutation file> <output>

Where *\<permutation_test_prog\>* is either permutationTest_cuda, or
permutationTest_omp in the build folder (assuming cmake is used to compile the
project) for GPU and CPU implementations respectively.

For the CUDA implementation, performing the calculations in parts is supported
when the GPU memory capacity is deemed insufficient to carry out the entire
calculations in one go. This is **NOT** the case for the CPU/OMP implementation.

## Output Files

The CUDA implementation automatically generates **two output files**:

1. **P-values file**: The filename you specify (e.g., `output.permout`)
2. **T-statistics file**: Same name with `_tstat` inserted before extension (e.g., `output_tstat.permout`)

The t-statistics file contains the observed mean difference (Group A - Group B) for each connection, which is useful for:
- Determining effect size and direction
- Visualizing spatial patterns of increases vs. decreases
- Thresholding by both significance AND magnitude

See `CHANGELOG.md` section on "T-Statistic Output" for complete documentation.

## Incremental Saving and Resume Capability

The CUDA implementation features:

- **Incremental Saving**: Results are saved to disk after each part is completed
- **Resume on Interruption**: If interrupted, the program automatically detects existing results and resumes from where it left off
- **Lower Memory Usage**: No need to store all results in RAM
- **Progress Monitoring**: Partial results available while the program runs

### Resume Example

```bash
# Start the job
./permutationTest_cuda filelist.txt permutations.txt output.permout

# ... program runs, completes 2 of 5 parts, then crashes ...

# Simply run the same command again - it will resume automatically
./permutationTest_cuda filelist.txt permutations.txt output.permout
# Output: [RESUME] Resuming from part 3 (already completed 2 parts)
```

To start fresh and ignore existing results, simply delete the output file before running.

For full documentation, see `CHANGELOG.md` which consolidates all changes, features, and bug fixes in a single comprehensive document.

## Documentation

- **`README.md`** (this file) - Main usage guide and quick reference
- **`CHANGELOG.md`** - Complete version history, bug fixes, and feature documentation
- **`TEST_RUN_RESULTS.md`** - Validation test results for the March 2026 bug fix

## Quick Usage Reference

### Basic Usage

```bash
# Generate permutations
python generatePermutations.py -nPerm 5000 -nA 10 -nB 10 -o perms.txt

# Edit first row of perms.txt to match original data grouping

# Run permutation test (one-tailed)
./permutationTest_cuda filelist.txt perms.txt output.permout

# Run permutation test (two-tailed)
./permutationTest_cuda filelist.txt perms.txt output.permout --two-tailed
```

### Output Files

The program automatically generates two files:
- `output.permout` - P-values (significance)
- `output_tstat.permout` - T-statistics (effect size and direction)

### Resume Capability

If interrupted, simply run the same command again - it will automatically resume from where it left off.

## Version Information

**Current Version:** v3.2 (March 29, 2026)

Major improvements:
- ✅ **GPU kernel optimization: ~150–270x faster per part** (v3.2, March 29, 2026)
- ✅ Fixed GPU memory calculation causing NaN t-statistics (v3.1, March 27, 2026)
- ✅ Fixed critical p-value calculation bug (v3.0)
- ✅ Added two-tailed test support (`--two-tailed` flag) (v3.0)
- ✅ Automatic t-statistic output (v3.0)
- ✅ Incremental saving with automatic resume (v3.0)
- ✅ Fixed integer overflow for large datasets (v2.0, January 2026)

See `CHANGELOG.md` for complete version history and migration guide.
