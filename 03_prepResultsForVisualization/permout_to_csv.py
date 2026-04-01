#!/usr/bin/env python3
"""
Convert permutation test results to CSV format for significant connections.

Reads permutation test output (.permout) and outputs a single CSV containing
every connection that passes the significance threshold, with both endpoint
coordinates, the p-value, and the t-statistic (when available).

Supports both formats — auto-detected via magic number:
  - Binary:  PERT header (0x50455254) written by permutationTest_cuda -b
             Accessed via numpy memmap — zero extra RAM regardless of file size.
  - Text:    legacy space-separated upper-triangular format

Binary format:
  Offset  Size  Field
   0       4    magic  0x50455254 ("PERT")
   4       4    version (uint32)
   8       8    gV     (uint64) — number of voxels
  16       8    n_elem = gV*(gV-1)/2  (uint64)
  24       n_elem * 4   upper-triangular float32, row-major

Input:
  - .permout file   : p-values  (binary or text)
  - _tstat.permout  : t-stats   (binary or text, auto-detected alongside p-values)
  - mask .dump file : voxel coordinates (i j k value, one row per voxel)

Output CSV columns: i1,j1,k1,i2,j2,k2,pvalue,tstat

Author: MOCCA Pipeline
Date: March 2026
"""

import argparse
import sys
import os
import struct
import numpy as np
from pathlib import Path
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install 'tqdm' for progress bar (pip install tqdm)")
    print("      Falling back to simple progress indicators\n")


# Must match permutationTest_cuda.cu and find_pvalue_threshold.py
PERMOUT_MAGIC = 0x50455254   # "PERT"
CCMAT_MAGIC   = 0x43434D54   # "CCMT"  (wrong file if seen here)
HDR_SIZE      = 24           # bytes: magic(4)+version(4)+gV(8)+n_elem(8)


def detect_binary_format(filepath):
    """
    Peek at the first 4 bytes.
    Returns (is_binary, gV, n_elem).
    Raises ValueError for raw ccmat files (wrong input).
    """
    with open(filepath, 'rb') as f:
        raw = f.read(HDR_SIZE)
    if len(raw) < HDR_SIZE:
        return False, 0, 0
    magic, version, gV, n_elem = struct.unpack_from('<IIQq', raw, 0)
    if magic == PERMOUT_MAGIC:
        return True, int(gV), int(n_elem)
    if magic == CCMAT_MAGIC:
        raise ValueError(
            f"{filepath} looks like a raw ccmat (input) file, not a permout result.")
    return False, 0, 0


def open_values(filepath, label="values"):
    """
    Return a flat array-like of float32/float64 values from a permout file.
    - Binary: numpy memmap (zero RAM cost, OS-level paging)
    - Text:   load into a float64 numpy array
    Also returns (is_binary, gV, n_elem).
    """
    try:
        is_binary, gV, n_elem = detect_binary_format(filepath)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if is_binary:
        print(f"  {label}: binary memmap  (gV={gV:,}, n_elem={n_elem:,})")
        arr = np.memmap(filepath, dtype=np.float32, mode='r',
                        offset=HDR_SIZE, shape=(n_elem,))
    else:
        print(f"  {label}: text — loading into memory...")
        raw = []
        with open(filepath, 'r') as f:
            for line in f:
                raw.extend([float(x) for x in line.split()])
        arr = np.array(raw, dtype=np.float64)
        gV = 0
        n_elem = len(arr)
        print(f"    loaded {n_elem:,} values")

    return arr, is_binary, gV, n_elem


def load_coordinates(coord_file):
    """Load voxel coordinates from brain template file."""
    print(f"Loading coordinates from {coord_file}...")
    coords = np.loadtxt(coord_file, usecols=(0, 1, 2), dtype=int)
    print(f"  Loaded {len(coords):,} voxel coordinates")
    return coords


def count_upper_triangular_elements(n):
    """Calculate number of elements in upper triangular matrix."""
    return n * (n - 1) // 2


def get_file_size_gb(filepath):
    """Get file size in GB."""
    return os.path.getsize(filepath) / (1024**3)


def process_permout(permout_file, coords, threshold, tstat_file=None, output_csv=None):
    """
    Process permutation output file and write significant connections to CSV.

    For binary files both pval and tstat arrays are opened as numpy memmaps —
    the OS pages in only the data actually touched, so RAM usage stays low even
    for 6+ GiB files.  For legacy text files the values are loaded into memory.

    Output CSV columns: i1,j1,k1,i2,j2,k2,pvalue,tstat
    """
    n_voxels = len(coords)
    expected_connections = count_upper_triangular_elements(n_voxels)

    print(f"\nOpening value arrays...")
    pval_arr, pval_bin, pval_gV, pval_nelem = open_values(permout_file, "p-values")

    use_tstat = tstat_file and Path(tstat_file).exists()
    tstat_arr = None
    if use_tstat:
        tstat_arr, _, _, _ = open_values(tstat_file, "t-statistics")
    else:
        print(f"  t-statistics: not found — CSV will have pvalue only")

    print(f"\nProcessing permutation results...")
    print(f"  N voxels:            {n_voxels:,}")
    print(f"  Expected connections:{expected_connections:,}")
    print(f"  P-value file size:   {get_file_size_gb(permout_file):.2f} GiB")
    print(f"  P-value threshold:   {threshold}")

    if len(pval_arr) != expected_connections:
        print(f"\nERROR: p-value count ({len(pval_arr):,}) != expected ({expected_connections:,})")
        sys.exit(1)
    if use_tstat and len(tstat_arr) != expected_connections:
        print(f"\nERROR: t-stat count ({len(tstat_arr):,}) != expected ({expected_connections:,})")
        sys.exit(1)

    if output_csv is None:
        output_csv = permout_file.replace(
            '.permout', f'_significant_p{str(threshold).replace("0.", "")}.csv')

    print(f"\nWriting significant connections to: {output_csv}\n")

    significant_count = 0
    connection_idx = 0

    # 4 MB write buffer
    BUF_LINES = 1 << 17   # ~131k rows before flushing
    row_buf = []

    with open(output_csv, 'w') as out_f:
        # Header
        if use_tstat:
            out_f.write("i1,j1,k1,i2,j2,k2,pvalue,tstat\n")
        else:
            out_f.write("i1,j1,k1,i2,j2,k2,pvalue\n")

        if HAS_TQDM:
            pbar = tqdm(total=n_voxels - 1, unit='voxels', desc="Processing",
                        ncols=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for v1 in range(n_voxels - 1):
            n_conn = n_voxels - v1 - 1
            s = connection_idx
            e = connection_idx + n_conn

            # Vectorised significance test over all connections from v1 at once
            local_pval = pval_arr[s:e]
            sig_mask   = local_pval < threshold
            sig_offsets = np.where(sig_mask)[0]

            if len(sig_offsets) > 0:
                i1, j1, k1 = coords[v1]
                v2_indices  = v1 + sig_offsets + 1
                p_vals      = local_pval[sig_offsets]

                if use_tstat:
                    t_vals = tstat_arr[s:e][sig_offsets]
                    for idx in range(len(sig_offsets)):
                        i2, j2, k2 = coords[v2_indices[idx]]
                        row_buf.append(
                            f"{i1},{j1},{k1},{i2},{j2},{k2},"
                            f"{p_vals[idx]:.6f},{t_vals[idx]:.6f}\n")
                else:
                    for idx in range(len(sig_offsets)):
                        i2, j2, k2 = coords[v2_indices[idx]]
                        row_buf.append(
                            f"{i1},{j1},{k1},{i2},{j2},{k2},{p_vals[idx]:.6f}\n")

                significant_count += len(sig_offsets)

                # Flush buffer periodically
                if len(row_buf) >= BUF_LINES:
                    out_f.writelines(row_buf)
                    row_buf.clear()

            connection_idx = e

            if HAS_TQDM:
                pbar.update(1)
                pbar.set_postfix({'significant': f'{significant_count:,}'})
            elif (v1 + 1) % 5000 == 0:
                pct = (v1 + 1) / n_voxels * 100
                print(f"  {v1+1:,}/{n_voxels:,} voxels ({pct:.1f}%) — "
                      f"{significant_count:,} significant", end='\r')

        # Flush remaining rows
        if row_buf:
            out_f.writelines(row_buf)

        if HAS_TQDM:
            pbar.close()

    print(f"\n\n{'='*60}")
    print(f"Completed!")
    print(f"  Total connections processed: {connection_idx:,}")
    print(f"  Significant (p < {threshold}):   {significant_count:,}")
    print(f"  Proportion significant:      {significant_count/connection_idx*100:.4f}%")
    print(f"  Output CSV: {output_csv}")
    print(f"{'='*60}\n")

    return significant_count


def main():
    parser = argparse.ArgumentParser(
        description='Convert permutation test results to CSV of significant connections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage — auto-detects binary/text, auto-finds _tstat file, threshold p<0.05
  %(prog)s results.permout

  # Custom threshold and mask
  %(prog)s results.permout -t 0.001 -m ../templates/mask3mm.dump

  # Explicit tstat file and output path
  %(prog)s results.permout -t 0.001 --tstat results_tstat.permout -o out.csv

Output CSV columns: i1,j1,k1,i2,j2,k2,pvalue,tstat
  (tstat column omitted if no tstat file found)

Memory note:
  Binary files are opened as numpy memmaps — the OS pages in only the
  data that is actually touched, so RAM stays low even for 6+ GiB files.
  Text files are loaded fully into memory.
        """
    )

    parser.add_argument('permout_file',
                        help='Path to .permout file with p-values (binary or text)')
    parser.add_argument('-t', '--threshold',
                        type=float, default=0.05,
                        help='P-value threshold for significance (default: 0.05)')
    parser.add_argument('-m', '--mask',
                        default='../templates/brain2mm.dump',
                        help='Path to mask/coordinate dump file (default: ../templates/brain2mm.dump)')
    parser.add_argument('--tstat',
                        help='Path to t-statistic file (default: auto-detect *_tstat.permout)')
    parser.add_argument('-o', '--output',
                        help='Output CSV path (default: auto-generate from input name)')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.permout_file).exists():
        print(f"ERROR: Input file not found: {args.permout_file}")
        sys.exit(1)
    if not Path(args.mask).exists():
        print(f"ERROR: Mask file not found: {args.mask}")
        sys.exit(1)

    # Auto-detect tstat file if not provided
    tstat_file = args.tstat
    if tstat_file is None:
        candidate = args.permout_file.replace('.permout', '_tstat.permout')
        if Path(candidate).exists():
            tstat_file = candidate

    # Detect pval format for banner
    try:
        is_binary, gV, n_elem = detect_binary_format(args.permout_file)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    fmt_str = "binary (PERT)" if is_binary else "text"

    print(f"\n{'='*60}")
    print(f"Permutation Results → CSV")
    print(f"{'='*60}")
    print(f"  P-value file : {args.permout_file}")
    print(f"  Format       : {fmt_str}")
    if is_binary:
        print(f"    gV         : {gV:,}")
        print(f"    n_elem     : {n_elem:,}")
    print(f"  File size    : {get_file_size_gb(args.permout_file):.2f} GiB")
    if tstat_file:
        print(f"  T-stat file  : {tstat_file}")
    else:
        print(f"  T-stat file  : (none found)")
    print(f"  Mask         : {args.mask}")
    print(f"  Threshold    : p < {args.threshold}")
    print(f"{'='*60}")

    # Load coordinates
    coords = load_coordinates(args.mask)

    # Run
    process_permout(
        args.permout_file,
        coords,
        args.threshold,
        tstat_file,
        args.output,
    )


if __name__ == '__main__':
    main()
