# Legacy uncorrected-p result preparation

Archived on 2026-08-26 as a frozen record of module 03 before development of
the multi-threshold bundle-FWER export path.

Contents:

- `permout_to_csv.py`: raw p/t `.permout` to thresholded edge CSV.
- `find_pvalue_threshold.py` and `quick_threshold_count.sh`: raw p-value
  distribution and cutoff inspection.
- `split_pos_neg_tstat.py`: positive/negative edge splitting.
- `apply_fdr.py`: post-hoc BH correction of raw edgewise p-values.
- `cudaPerm/`: original conversion notebook.
- `README.md`: complete historical usage documentation.

The `afni/` directory remains active because its utilities are not specific to
uncorrected permutation p-values.
