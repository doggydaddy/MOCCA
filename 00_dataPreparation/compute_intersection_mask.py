#!/usr/bin/env python3
"""
compute_intersection_mask.py

Derives a data-driven brain mask that is the logical AND of:
  1. The existing template mask  (mask2mm.nii)
  2. The non-zero (covered) voxels in EVERY resampled 4D functional image

Voxels that are zero across ALL time-points in ANY scan are excluded from the
new mask.  This eliminates the inferior FOV truncation artefact that causes
zero-variance time-courses and subsequent NaN correlations in the CC matrices.

Outputs (all in /mnt/islay/MOCCA/templates/):
  mask2mm_intersection.nii   – new binary mask (uint8, same grid as mask2mm.nii)

A summary printed to stdout lists:
  • original mask voxel count
  • how many voxels each scan removes (on top of what was already removed)
  • final intersection mask voxel count + percentage of original retained

Usage:
    python compute_intersection_mask.py [--funcdir DIR] [--mask FILE]
                                        [--output FILE] [--threshold VALUE]
"""

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np

# ── defaults ───────────────────────────────────────────────────────────────────
MASK_PATH  = "/mnt/islay/MOCCA/templates/mask2mm.nii"
FUNC_DIR   = "/mnt/highlands/data/MOCCA_UCLA/resampled_func_images_2mm"
OUT_PATH   = "/mnt/islay/MOCCA/templates/mask2mm_intersection.nii"
THRESHOLD  = 1e-6   # variance below this → voxel treated as zero / no-coverage


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask",      default=MASK_PATH,  help="Template mask NIfTI")
    p.add_argument("--funcdir",   default=FUNC_DIR,   help="Directory of resampled 4D NIfTIs")
    p.add_argument("--output",    default=OUT_PATH,   help="Output intersection mask path")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help="Variance threshold for zero-coverage detection (default %(default)s)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── load template mask ────────────────────────────────────────────────────
    mask_img  = nib.load(args.mask)
    mask_data = mask_img.get_fdata(dtype=np.float32)
    template  = mask_data > 0                          # bool (X, Y, Z)
    shape3    = template.shape

    n_orig = int(template.sum())
    print(f"Template mask : {args.mask}")
    print(f"  Grid shape  : {shape3}")
    print(f"  Voxel size  : {mask_img.header.get_zooms()}")
    print(f"  In-mask     : {n_orig:,} voxels\n")

    # ── accumulate coverage map ───────────────────────────────────────────────
    # coverage[x,y,z] = True  ↔  voxel has non-zero variance in EVERY scan
    coverage = template.copy()                          # start from template

    func_files = sorted(Path(args.funcdir).glob("*.nii"))
    print(f"Scanning {len(func_files)} functional files …\n")

    removed_cumulative = 0

    for i, fp in enumerate(func_files, 1):
        img  = nib.load(str(fp))
        data = img.get_fdata(dtype=np.float32)          # (X, Y, Z, T)

        if data.shape[:3] != shape3:
            print(f"  [{i:3d}] WARNING: shape mismatch {fp.name} "
                  f"{data.shape[:3]} vs mask {shape3} — skipping")
            continue

        if data.ndim != 4:
            print(f"  [{i:3d}] WARNING: not 4D ({fp.name}) — skipping")
            continue

        # variance map (computed only over time; 0 where coverage is missing)
        var_vol = np.var(data, axis=3, ddof=0)           # (X, Y, Z)

        # voxels WITH sufficient variance in this scan
        has_signal = var_vol >= args.threshold            # bool (X, Y, Z)

        # voxels that were previously covered but are NOT in this scan
        newly_lost = coverage & ~has_signal
        n_lost     = int(newly_lost.sum())

        # update running intersection
        coverage &= has_signal

        n_current  = int(coverage.sum())
        removed_cumulative += n_lost

        # report only scans that actually trim the mask
        if n_lost > 0:
            print(f"  [{i:3d}/{len(func_files)}]  {fp.name:<22s}  "
                  f"removed {n_lost:6,} voxels  →  {n_current:,} remaining")

    # ── apply template mask to coverage (belt-and-suspenders) ─────────────────
    intersection = template & coverage

    n_final   = int(intersection.sum())
    n_removed = n_orig - n_final
    pct_kept  = 100.0 * n_final / n_orig

    print(f"\n{'='*65}")
    print(f"INTERSECTION MASK SUMMARY")
    print(f"  Original template voxels : {n_orig:>10,}")
    print(f"  Voxels removed           : {n_removed:>10,}  ({100-pct_kept:.2f}%)")
    print(f"  Final intersection voxels: {n_final:>10,}  ({pct_kept:.2f}% of original)")
    print(f"{'='*65}\n")

    # ── save output ───────────────────────────────────────────────────────────
    out_data = intersection.astype(np.uint8)
    out_img  = nib.Nifti1Image(out_data, mask_img.affine, mask_img.header)
    out_img.set_data_dtype(np.uint8)
    out_img.header.set_data_shape(shape3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    nib.save(out_img, args.output)
    print(f"Saved intersection mask → {args.output}")

    # ── per-subject breakdown of what was lost ────────────────────────────────
    # Show which z-slices were trimmed
    removed_vol = template & ~intersection
    if removed_vol.any():
        print(f"\nRemoved voxels by z-slice (MNI z):")
        affine = mask_img.affine
        for z in range(shape3[2]):
            n_slice_removed = int(removed_vol[:, :, z].sum())
            n_slice_orig    = int(template[:, :, z].sum())
            if n_slice_removed > 0:
                mni_z = affine[2, 2] * z + affine[2, 3]
                pct   = 100.0 * n_slice_removed / n_slice_orig if n_slice_orig else 0
                bar   = "█" * int(pct / 2.5)
                print(f"  z={z:2d} (MNI {mni_z:+.0f}mm)  "
                      f"{n_slice_removed:5,}/{n_slice_orig:5,} removed ({pct:5.1f}%)  {bar}")


if __name__ == "__main__":
    main()
