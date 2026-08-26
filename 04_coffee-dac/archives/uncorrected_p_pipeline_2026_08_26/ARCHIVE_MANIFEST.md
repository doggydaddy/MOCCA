# Legacy COFFEE-DAC uncorrected-p workflow

Archived on 2026-08-26.  This directory preserves both generations of the
CSV-driven bundle/network pipeline and the standalone visualization routines
used on edge CSVs selected by uncorrected permutation p-values.

Archived routines:

- `coffee_dac_pipeline.py`, `run_pipeline.py`: original implementation/CLI.
- `coffee_dac_pipeline_v2.py`, `run_pipeline_v2.py`: strict-bundle and
  provenance-aware implementation/CLI used for the historical result caches.
- `export_networks_to_gif.py`, `visualizer_local.py`, and `my_colormap.py`:
  original standalone visualization/export path.
- `README_coffee-dac.md` and `notes.txt`: historical method documentation.

Not archived:

- Input CSVs, processed caches, linkage arrays, result notes, and figures.
- `mocca_gui/`, `launch_gui.py`, meshes, and current GUI support files.
- Bundle-FWER C++ code in module 02.

The two pipeline implementations are still reachable through compatibility
shims at the module root so existing GUI/cache behavior remains unchanged.
