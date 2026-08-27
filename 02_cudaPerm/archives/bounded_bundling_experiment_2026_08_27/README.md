# Bounded bundling experiment — rejected 2026-08-27

This records the non-chaining, strongest-seed endpoint-patch experiment. It
bounded every endpoint to `neighbor_dist=1` from a representative connection,
which capped the two endpoint patches at 3×3×3 voxels and prevented global
percolation.

The method was not adopted because manual visualization showed that the fixed
endpoint cubes were arbitrary and anatomically unconvincing. It also divided
the observed result into very many small bundles and produced no grid-FWER
discoveries at alpha 0.05.

The implementation remains available for provenance through
`bundle_fwer_bounded_omp` and `--bundle-method bounded`, but the active default
is again the historical transitive `strict` method. Do not use the bounded
method for production inference without an explicit new methodological
decision.

Completed experimental output:

`/mnt/storage/MOCCA_UCLA/bundle_fwer_LTLEvsRTLE_10k_pgrid_bounded_cpp/`

Its top-10 visualization caches are exploratory and non-significant.

The next intended investigation is a defensible cluster-forming threshold or
sparsity/percolation rule that lets the historical bundler operate below its
giant-component transition without selecting a threshold merely because the
observed result looks attractive.
