#!/usr/bin/env python3
"""
Apply Benjamini-Hochberg FDR correction to permutation test p-values.

Reads a raw .permout file (binary or text), applies the BH procedure at a
chosen FDR level (default q = 0.05), and writes a corrected .permout file
in the same format.  The companion _tstat.permout is symlinked (or copied)
alongside, since t-statistics are unchanged by FDR.

The output file contains **adjusted p-values** (q-values), not the original
raw p-values.  A connection is significant at FDR level q when its adjusted
p-value is ≤ q.

Benjamini-Hochberg procedure:
  1. Sort all m p-values: p(1) ≤ p(2) ≤ … ≤ p(m)
  2. Find the largest k such that  p(k) ≤ k/m * q
  3. Reject all hypotheses 1 … k
  Adjusted p-value (q-value) for rank i:
      q(i) = min( p(i) * m / i,  q(i+1) )   (enforce monotonicity)

Supports both binary (.permout with PERT header) and text formats.
Binary format is strongly preferred for large files — uses numpy memmap
so RAM usage is minimal regardless of file size.

Author: MOCCA Pipeline
Date: April 2026
"""

import argparse
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

# ── Constants (must match permutationTest_cuda.cu) ────────────────────────────
PERMOUT_MAGIC = 0x50455254   # "PERT"
CCMAT_MAGIC   = 0x43434D54   # "CCMT"
CCMAT_VERSION = 1
HDR_SIZE      = 24           # bytes: magic(4) + version(4) + gV(8) + n_elem(8)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def detect_binary(filepath):
    """Return (is_binary, gV, n_elem) by peeking at the header."""
    with open(filepath, 'rb') as f:
        raw = f.read(HDR_SIZE)
    if len(raw) < HDR_SIZE:
        return False, 0, 0
    magic, _ver, gV, n_elem = struct.unpack_from('<IIQq', raw, 0)
    if magic == PERMOUT_MAGIC:
        return True, int(gV), int(n_elem)
    if magic == CCMAT_MAGIC:
        raise ValueError(f"{filepath} looks like a ccmat input, not a permout.")
    return False, 0, 0


def load_pvalues(filepath):
    """
    Load p-values from a .permout file.
    Returns (pvals_array, is_binary, gV, n_elem).
    Binary → numpy memmap (read-only); Text → in-memory float64 array.
    """
    is_binary, gV, n_elem = detect_binary(filepath)

    if is_binary:
        print(f"  Format : binary  (gV={gV:,}, n_elem={n_elem:,})")
        arr = np.memmap(filepath, dtype=np.float32, mode='r',
                        offset=HDR_SIZE, shape=(n_elem,))
    else:
        print(f"  Format : text — loading into memory ...")
        vals = []
        with open(filepath, 'r') as f:
            for line in f:
                vals.extend(float(x) for x in line.split())
        arr = np.array(vals, dtype=np.float64)
        n_elem = len(arr)
        print(f"    loaded {n_elem:,} values")

    return arr, is_binary, gV, n_elem


def write_binary(filepath, qvals, gV, n_elem):
    """Write q-values as a binary .permout file."""
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<II', PERMOUT_MAGIC, CCMAT_VERSION))
        f.write(struct.pack('<QQ', gV, n_elem))
        qvals.astype(np.float32).tofile(f)
    size_gb = os.path.getsize(filepath) / (1024**3)
    print(f"  Written: {filepath}  ({n_elem:,} values, {size_gb:.2f} GiB)")


def write_text(filepath, qvals, gV):
    """
    Write q-values as a text .permout file (upper-triangular rows).
    If gV is unknown (0), write all values on a single line per value — but
    that won't match the original row structure.  Best to use binary.
    """
    with open(filepath, 'w') as f:
        if gV > 0:
            idx = 0
            for i in range(gV):
                row_len = gV - i - 1
                row = qvals[idx:idx + row_len]
                f.write(' '.join(f'{v:.6f}' for v in row) + '\n')
                idx += row_len
        else:
            # Fallback: one value per line
            for v in qvals:
                f.write(f'{v:.6f}\n')
    size_gb = os.path.getsize(filepath) / (1024**3)
    print(f"  Written: {filepath}  ({len(qvals):,} values, {size_gb:.2f} GiB)")


# ── BH-FDR ────────────────────────────────────────────────────────────────────

def benjamini_hochberg(pvals):
    """
    Apply Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : ndarray, shape (m,)
        Raw p-values.

    Returns
    -------
    qvals : ndarray, shape (m,)
        Adjusted p-values (q-values).  A connection is significant at
        FDR level q when its q-value is ≤ q.
    """
    m = len(pvals)
    print(f"  Applying BH-FDR to {m:,} p-values ...")

    # Work in float64 for precision
    p = np.array(pvals, dtype=np.float64)

    # 1. Sort indices
    order = np.argsort(p)           # indices that sort p ascending
    rank  = np.empty_like(order)
    rank[order] = np.arange(1, m + 1)   # 1-based ranks

    # 2. Adjusted p-value: q(i) = p(i) * m / rank(i)
    qvals = p * m / rank

    # 3. Enforce monotonicity (walk backwards through sorted order)
    #    q(i) = min( q(i),  q(i+1) )
    qvals_sorted = qvals[order]
    for i in range(m - 2, -1, -1):
        if qvals_sorted[i] > qvals_sorted[i + 1]:
            qvals_sorted[i] = qvals_sorted[i + 1]

    # 4. Cap at 1.0
    np.minimum(qvals_sorted, 1.0, out=qvals_sorted)

    # 5. Map back to original order
    qvals_out = np.empty(m, dtype=np.float64)
    qvals_out[order] = qvals_sorted

    return qvals_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply Benjamini-Hochberg FDR correction to .permout p-values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Default FDR q = 0.05, auto-detect binary/text
  python apply_fdr.py results.permout

  # Custom FDR level and output path
  python apply_fdr.py results.permout -q 0.01 -o results_fdr01.permout

  # The companion _tstat.permout is copied/symlinked automatically
""")
    parser.add_argument('input', help='Input .permout file (raw p-values)')
    parser.add_argument('-q', '--fdr', type=float, default=0.05,
                        help='FDR level (default: 0.05)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output .permout file (default: <input>_fdr.permout)')
    parser.add_argument('--no-copy-tstat', action='store_true',
                        help='Do not copy/symlink the _tstat.permout file')
    args = parser.parse_args()

    inpath = Path(args.input)
    if not inpath.exists():
        print(f"ERROR: file not found: {inpath}", file=sys.stderr)
        sys.exit(1)

    # Derive output path
    if args.output:
        outpath = Path(args.output)
    else:
        stem = inpath.stem          # e.g. "results" from "results.permout"
        suffix = inpath.suffix      # e.g. ".permout"
        outpath = inpath.with_name(f"{stem}_fdr{suffix}")

    print("=" * 60)
    print("Benjamini-Hochberg FDR Correction")
    print("=" * 60)
    print(f"  Input  : {inpath}")
    print(f"  Output : {outpath}")
    print(f"  FDR q  : {args.fdr}")
    print()

    # ── Load p-values ─────────────────────────────────────────────────────
    print("Loading p-values ...")
    pvals, is_binary, gV, n_elem = load_pvalues(inpath)
    print(f"  Total p-values: {n_elem:,}")
    print()

    # ── Quick stats on raw p-values ───────────────────────────────────────
    pmin = float(np.min(pvals))
    pmax = float(np.max(pvals))
    print(f"  Raw p-value range: [{pmin:.6e}, {pmax:.6e}]")

    # Count raw significant at common thresholds
    for thr in [0.05, 0.01, 0.001, 0.0001, 0.00001]:
        cnt = int(np.sum(np.array(pvals) < thr))
        pct = 100.0 * cnt / n_elem
        print(f"    raw p < {thr:<10g}: {cnt:>15,}  ({pct:.4f}%)")
    print()

    # ── Apply BH-FDR ─────────────────────────────────────────────────────
    qvals = benjamini_hochberg(pvals)

    # ── Stats on corrected q-values ───────────────────────────────────────
    qmin = float(np.min(qvals))
    qmax = float(np.max(qvals))
    print(f"  Adjusted q-value range: [{qmin:.6e}, {qmax:.6e}]")

    for thr in [0.05, 0.01, 0.001, 0.0001, 0.00001]:
        cnt = int(np.sum(qvals < thr))
        pct = 100.0 * cnt / n_elem
        print(f"    FDR q < {thr:<10g}: {cnt:>15,}  ({pct:.4f}%)")
    print()

    n_sig = int(np.sum(qvals <= args.fdr))
    print(f"  ★ Significant at FDR q ≤ {args.fdr}: {n_sig:,} / {n_elem:,}"
          f"  ({100.0 * n_sig / n_elem:.4f}%)")
    print()

    # ── Write output ──────────────────────────────────────────────────────
    print("Writing corrected p-values ...")
    if is_binary:
        write_binary(outpath, qvals, gV, n_elem)
    else:
        write_text(outpath, qvals, gV)

    # ── Copy / symlink tstat file ─────────────────────────────────────────
    if not args.no_copy_tstat:
        # Derive tstat paths
        tstat_in = inpath.with_name(
            inpath.stem.replace('.permout', '') + '_tstat' + inpath.suffix
        )
        # Handle case where stem doesn't contain .permout
        if not tstat_in.exists():
            tstat_in = inpath.with_name(inpath.stem + '_tstat' + inpath.suffix)

        if tstat_in.exists():
            tstat_out = outpath.with_name(
                outpath.stem.replace('.permout', '') + '_tstat' + outpath.suffix
            )
            if not tstat_out.exists():
                tstat_out_alt = outpath.with_name(outpath.stem + '_tstat' + outpath.suffix)
                tstat_out = tstat_out_alt

            print(f"  Copying t-stat file: {tstat_in.name} → {tstat_out.name}")
            shutil.copy2(tstat_in, tstat_out)
        else:
            print(f"  No companion _tstat file found (looked for {tstat_in.name})")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
