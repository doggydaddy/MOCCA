# Legacy edgewise permutation-p pipeline

Archived on 2026-08-26.  These files are retained as a frozen record of the
pipeline that generated per-edge, uncorrected permutation p-values and the
historical `*.permout` inputs to module 03.

Archived files:

- `permutationTest_cuda.cu`: CUDA edgewise permutation p-values.
- `permutationTest_omp.c`: OpenMP reference implementation.
- `validate_pvalues.sh`: legacy p-value output validation.
- `generateTestData/`: original test-data generator and fixtures.
- `CHANGELOG.md`: version history for the archived executables.

Shared inputs retained in the active parent directory:

- `generatePermutations.py`, because corrected and bundle-FWER runs use it.
- `ccmat_io.*`, `perm_kernels.*`, and `results_io.*`, because the corrected
  max-statistic executable uses them.
- `permutationTest_cuda_fwer.cu` and all bundle-FWER sources.

To build only the archived executables:

```bash
cmake -S . -B build
cmake --build build -j2
```
