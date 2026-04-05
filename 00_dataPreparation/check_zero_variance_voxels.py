#!/usr/bin/env python3
"""
check_zero_variance_voxels.py

Scans all resampled 4D functional NIfTI images against the brain mask and
reports any voxels inside the mask that have zero (or near-zero) temporal
variance.  Such constant time-courses will produce NaN correlations in the
connectivity matrices.

Usage:
    python check_zero_variance_voxels.py [--threshold 1e-6] [--output report.csv]
"""

import os
import sys
import argparse
import csv
import numpy as np
import nibabel as nib
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
MASK_PATH   = "/mnt/islay/MOCCA/templates/mask2mm.nii"
FUNC_DIR    = "/mnt/highlands/data/MOCCA_UCLA/resampled_func_images_2mm"
OUTPUT_CSV  = "/mnt/islay/MOCCA/03_prepResultsForVisualization/zero_variance_report.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mask",      default=MASK_PATH,  help="Brain mask NIfTI")
    p.add_argument("--funcdir",   default=FUNC_DIR,   help="Directory with resampled 4D NIfTIs")
    p.add_argument("--output",    default=OUTPUT_CSV, help="Output CSV report path")
    p.add_argument("--threshold", type=float, default=1e-6,
                   help="Variance threshold below which a voxel is flagged (default 1e-6)")
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


def load_mask(mask_path):
    img  = nib.load(mask_path)
    data = img.get_fdata()
    mask = data > 0
    print(f"[mask]  shape={mask.shape}  in-mask voxels={mask.sum()}")
    return mask, img.affine


def check_file(func_path, mask, threshold, verbose=False):
    """
    Returns a dict with statistics for one 4D file.
    """
    img  = nib.load(func_path)
    data = img.get_fdata(dtype=np.float32)          # (X, Y, Z, T)

    if data.ndim != 4:
        return {"error": f"Expected 4D image, got {data.ndim}D"}

    n_vox_mask = int(mask.sum())
    n_tp       = data.shape[3]

    # Extract only in-mask time-courses  →  (n_vox_mask, T)
    tc = data[mask]                                  # shape (N, T)

    # --- check for NaN / Inf in the raw data itself ---
    n_nan_raw  = int(np.isnan(tc).any(axis=1).sum())
    n_inf_raw  = int(np.isinf(tc).any(axis=1).sum())

    # --- variance check ---
    var = np.var(tc, axis=1, ddof=0)                # (N,)
    zero_var_mask = var < threshold
    n_zero_var    = int(zero_var_mask.sum())

    # xyz indices of offending voxels
    zero_var_xyz = np.argwhere(mask)[zero_var_mask]  # (n_zero, 3)

    result = {
        "file":          os.path.basename(func_path),
        "n_timepoints":  n_tp,
        "n_mask_voxels": n_vox_mask,
        "n_nan_raw":     n_nan_raw,
        "n_inf_raw":     n_inf_raw,
        "n_zero_var":    n_zero_var,
        "zero_var_voxel_xyz": zero_var_xyz.tolist() if n_zero_var else [],
    }

    if verbose and n_zero_var:
        print(f"  !! {n_zero_var} zero-variance voxels  |  "
              f"first 5 xyz: {zero_var_xyz[:5].tolist()}")

    return result


def main():
    args = parse_args()

    mask, mask_affine = load_mask(args.mask)

    func_files = sorted(Path(args.funcdir).glob("*.nii"))
    print(f"[scan]  found {len(func_files)} NIfTI files in {args.funcdir}")
    print(f"[scan]  zero-variance threshold = {args.threshold}\n")

    rows = []
    problem_files = []

    for i, fp in enumerate(func_files, 1):
        res = check_file(str(fp), mask, args.threshold, verbose=args.verbose)

        has_problem = (res.get("n_zero_var", 0) > 0 or
                       res.get("n_nan_raw",  0) > 0 or
                       res.get("n_inf_raw",  0) > 0 or
                       "error" in res)

        status = "OK"
        if "error" in res:
            status = f"ERROR: {res['error']}"
        elif has_problem:
            status = "PROBLEM"
            problem_files.append(fp.name)

        print(f"[{i:3d}/{len(func_files)}]  {fp.name:<20s}  "
              f"T={res.get('n_timepoints','?'):4}  "
              f"zero_var={res.get('n_zero_var', 0):5d}  "
              f"nan_raw={res.get('n_nan_raw', 0):4d}  "
              f"inf_raw={res.get('n_inf_raw', 0):4d}  "
              f"[{status}]")

        rows.append({
            "file":         res.get("file", fp.name),
            "n_timepoints": res.get("n_timepoints", ""),
            "n_mask_voxels":res.get("n_mask_voxels",""),
            "n_zero_var":   res.get("n_zero_var",   0),
            "n_nan_raw":    res.get("n_nan_raw",    0),
            "n_inf_raw":    res.get("n_inf_raw",    0),
            "status":       status,
            "zero_var_voxels_xyz": str(res.get("zero_var_voxel_xyz", [])),
        })

    # ── summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"SUMMARY  (threshold = {args.threshold})")
    print(f"  Total files checked  : {len(rows)}")
    n_problems = sum(1 for r in rows if r["status"] != "OK")
    print(f"  Files with problems  : {n_problems}")
    if problem_files:
        print(f"  Problem file list    :")
        for f in problem_files:
            r = next(x for x in rows if x["file"] == f)
            print(f"    {f:<25s}  zero_var={r['n_zero_var']:5d}  "
                  f"nan={r['n_nan_raw']:4d}  inf={r['n_inf_raw']:4d}")

    total_zero_var_files = sum(1 for r in rows if r["n_zero_var"] > 0)
    print(f"\n  Files with ≥1 zero-variance in-mask voxel: {total_zero_var_files}")
    total_zero_var = sum(r["n_zero_var"] for r in rows if isinstance(r["n_zero_var"], int))
    print(f"  Total zero-variance voxel-instances       : {total_zero_var}")
    print("="*70)

    # ── write CSV ──────────────────────────────────────────────────────────────
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[output]  report written to {args.output}")


if __name__ == "__main__":
    main()
