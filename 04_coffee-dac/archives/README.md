# Archived COFFEE-DAC modules

The original CSV-driven COFFEE-DAC processing and standalone visualization
routines are frozen in `uncorrected_p_pipeline_2026_08_26/`.

The active GUI imports the historical processing APIs through small top-level
compatibility shims.  This keeps existing caches and visualization functional
without returning the archived source files to the active module root.
