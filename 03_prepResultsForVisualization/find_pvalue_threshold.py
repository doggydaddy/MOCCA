#!/usr/bin/env python3
"""
Comprehensive p-value analysis for permutation test results.

This script performs a complete analysis of p-value distributions from
permutation test output files using ultra-low memory streaming. It calculates:
1. Percentage of connections below standard p-value thresholds
2. P-value thresholds for top percentiles of most significant connections
3. Complete p-value distribution for visualization

Supports both text (.permout from old pipeline) and binary (.permout with
PERMOUT_MAGIC header written by permutationTest_cuda -b).

Binary format (auto-detected via magic number):
  Offset  Size  Field
   0       4    magic  0x50455254 ("PERT")
   4       4    version (uint32)
   8       8    gV     (uint64) — number of voxels
  16       8    n_elem = gV*(gV-1)/2  (uint64)
  24       n_elem * 4   upper-triangular float32, row-major

Ultra-low memory streaming approach:
- Pass 1: Build histogram and basic statistics (only histogram in memory)
- Pass 2: Calculate percentile thresholds from histogram (no arrays)
- Maximum memory: ~8KB for histogram + minimal overhead
- No heaps, no arrays of p-values

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


# Must match permutationTest_cuda.cu
PERMOUT_MAGIC = 0x50455254   # "PERT"
CCMAT_MAGIC   = 0x43434D54   # "CCMT"  (input ccmat files — not expected here but detectable)
HDR_SIZE      = 24           # bytes: magic(4) + version(4) + gV(8) + n_elem(8)
CHUNK_FLOATS  = 1 << 22      # 4 M floats = 16 MiB per read — efficient streaming


def get_file_size_gb(filepath):
    """Get file size in GB."""
    return os.path.getsize(filepath) / (1024**3)


def detect_binary_format(filepath):
    """
    Peek at the first 4 bytes.
    Returns (is_binary, gV, n_elem) — gV/n_elem are valid only when is_binary=True.
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
            f"{filepath} looks like a raw ccmat (input) file, not a permout result file.")
    return False, 0, 0


class StreamingPvalueAnalyzer:
    """
    Ultra-low memory streaming analyzer for p-value files.
    Uses only histogram in memory (fixed size regardless of input).
    Supports both text and binary (.permout) formats — auto-detected.
    """
    
    def __init__(self, filepath, n_bins=10000):
        self.filepath = filepath
        
        # Detect format once at construction time
        self.is_binary, self.gV, self.n_elem = detect_binary_format(filepath)
        self.format_str = "binary" if self.is_binary else "text"
        
        # Basic statistics (minimal memory)
        self.n_values = 0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.sum_val = 0.0
        self.sum_sq_val = 0.0
        
        # Standard p-value thresholds to count
        self.standard_thresholds = [0.05, 0.01, 0.001, 0.0005, 0.0001]
        self.threshold_counts = {t: 0 for t in self.standard_thresholds}
        
        # Percentiles to calculate
        self.target_percentiles = [1.0, 5.0, 10.0, 15.0, 20.0]
        self.percentile_thresholds = {}
        
        # Histogram for distribution (fixed size - uses ~80KB for 10k bins)
        self.n_bins = n_bins
        self.hist_counts = None
        self.hist_edges = None

    def _process_chunk(self, vals):
        """Update statistics and histogram for a numpy array of float32/float64 values."""
        self.n_values += len(vals)
        self.min_val   = min(self.min_val,  float(vals.min()))
        self.max_val   = max(self.max_val,  float(vals.max()))
        self.sum_val   += float(vals.sum())
        self.sum_sq_val += float((vals.astype(np.float64) ** 2).sum())

        for thresh in self.standard_thresholds:
            self.threshold_counts[thresh] += int(np.sum(vals < thresh))

        # np.digitize is slower than searchsorted for sorted edges
        idxs = np.searchsorted(self.hist_edges[1:], vals)
        idxs = np.clip(idxs, 0, self.n_bins - 1)
        np.add.at(self.hist_counts, idxs, 1)

    def pass1_build_histogram(self):
        """
        First pass: Build histogram and collect basic statistics.
        Memory usage: Only histogram array (~80KB for 10k bins) + one chunk buffer.
        Automatically uses binary or text path based on detected format.
        """
        print(f"\nPass 1: Building histogram and basic statistics...")
        print(f"  Reading: {self.filepath}  [{self.format_str}]")

        # Create fixed log-scale bins (0 to 1, with focus on small values)
        self.hist_edges  = np.logspace(-20, 0, self.n_bins + 1)
        self.hist_counts = np.zeros(self.n_bins, dtype=np.int64)

        if self.is_binary:
            self._pass1_binary()
        else:
            self._pass1_text()

        # Derived statistics
        self.mean_val = self.sum_val / self.n_values if self.n_values > 0 else 0
        variance = (self.sum_sq_val / self.n_values) - (self.mean_val ** 2) if self.n_values > 0 else 0
        self.std_val = np.sqrt(max(0, variance))

        print(f"  Pass 1 complete!  ({self.n_values:,} values read)")

    def _pass1_binary(self):
        """Binary path: skip 24-byte header, then stream float32 chunks."""
        file_size   = os.path.getsize(self.filepath)
        data_bytes  = file_size - HDR_SIZE
        total_floats = data_bytes // 4

        if HAS_TQDM:
            pbar = tqdm(total=total_floats, unit=' vals', unit_scale=True,
                        desc="  Processing", ncols=100)

        with open(self.filepath, 'rb') as f:
            f.seek(HDR_SIZE)
            floats_read = 0
            while True:
                raw = f.read(CHUNK_FLOATS * 4)
                if not raw:
                    break
                chunk = np.frombuffer(raw, dtype=np.float32)
                self._process_chunk(chunk)
                floats_read += len(chunk)
                if HAS_TQDM:
                    pbar.update(len(chunk))
                else:
                    print(f"    {floats_read:,} / {total_floats:,} values", end='\r')

        if HAS_TQDM:
            pbar.close()
        else:
            print()

    def _pass1_text(self):
        """Text path: unchanged line-by-line float parsing."""
        file_size = os.path.getsize(self.filepath)

        with open(self.filepath, 'r') as f:
            if HAS_TQDM:
                pbar = tqdm(total=file_size, unit='B', unit_scale=True,
                            desc="  Processing", ncols=100)

            for line_num, line in enumerate(f, 1):
                vals = np.array([float(x) for x in line.split()], dtype=np.float64)
                if len(vals) == 0:
                    continue
                self._process_chunk(vals)

                if HAS_TQDM:
                    pbar.update(f.tell() - pbar.n)
                elif line_num % 10000 == 0:
                    print(f"    Line {line_num:,}, {self.n_values:,} values", end='\r')

            if HAS_TQDM:
                pbar.close()
            else:
                print(f"    Line {line_num:,}, {self.n_values:,} values")
        
    def pass2_calculate_percentiles(self):
        """
        Second pass: Calculate percentile thresholds from histogram.
        Uses cumulative counts to find thresholds without storing values.
        """
        print(f"\nPass 2: Calculating percentile thresholds from histogram...")
        
        # Calculate cumulative counts
        cumulative = np.cumsum(self.hist_counts)
        
        # Find threshold for each percentile
        for p in self.target_percentiles:
            target_count = int(np.ceil(self.n_values * p / 100.0))
            
            # Find first bin where cumulative count exceeds target
            bin_idx = np.searchsorted(cumulative, target_count)
            
            if bin_idx < self.n_bins:
                # Use the right edge of the bin as threshold
                self.percentile_thresholds[p] = self.hist_edges[bin_idx + 1]
            else:
                self.percentile_thresholds[p] = self.max_val
            
            print(f"    Top {p:>5.1f}%: threshold = {self.percentile_thresholds[p]:.6e}")
        
        print(f"  Pass 2 complete!")
    
    def save_distribution(self, output_file):
        """Save histogram distribution to file for later visualization."""
        print(f"\nSaving distribution to: {output_file}")
        
        # Prepare data
        bin_centers = (self.hist_edges[:-1] + self.hist_edges[1:]) / 2
        cumulative = np.cumsum(self.hist_counts)
        cumulative_pct = (cumulative / self.n_values) * 100 if self.n_values > 0 else 0
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write("# P-value distribution from permutation test\n")
            f.write(f"# Total connections: {self.n_values:,}\n")
            f.write(f"# File: {self.filepath}\n")
            f.write("#\n")
            f.write("# Columns:\n")
            f.write("#   bin_center: Center of histogram bin (log scale)\n")
            f.write("#   bin_left: Left edge of bin\n")
            f.write("#   bin_right: Right edge of bin\n")
            f.write("#   count: Number of p-values in this bin\n")
            f.write("#   density: Count / bin_width (for plotting)\n")
            f.write("#   cumulative: Cumulative count up to this bin\n")
            f.write("#   cumulative_pct: Cumulative percentage\n")
            f.write("#\n")
            f.write("bin_center,bin_left,bin_right,count,density,cumulative,cumulative_pct\n")
            
            for i in range(self.n_bins):
                bin_width = self.hist_edges[i+1] - self.hist_edges[i]
                density = self.hist_counts[i] / bin_width if bin_width > 0 else 0
                f.write(f"{bin_centers[i]:.6e},{self.hist_edges[i]:.6e},"
                       f"{self.hist_edges[i+1]:.6e},{self.hist_counts[i]},"
                       f"{density:.6e},{cumulative[i]},{cumulative_pct[i]:.6f}\n")
        
        print(f"  Saved {self.n_bins} histogram bins")
        print(f"  Use this file for plotting distribution")
    
    def print_report(self):
        """Print comprehensive analysis report."""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE P-VALUE ANALYSIS REPORT")
        print(f"{'='*80}")
        print(f"\nFile: {self.filepath}")
        print(f"Total connections: {self.n_values:,}")
        print(f"\n{'-'*80}")
        print(f"BASIC STATISTICS")
        print(f"{'-'*80}")
        print(f"Minimum p-value:       {self.min_val:.6e}")
        print(f"Maximum p-value:       {self.max_val:.6e}")
        print(f"Mean p-value:          {self.mean_val:.6e}")
        print(f"Std deviation:         {self.std_val:.6e}")
        
        print(f"\n{'-'*80}")
        print(f"STANDARD P-VALUE THRESHOLDS")
        print(f"{'-'*80}")
        print(f"{'Threshold':<15} {'Count':<20} {'Percentage':<15}")
        print(f"{'-'*15} {'-'*20} {'-'*15}")
        
        for thresh in sorted(self.standard_thresholds, reverse=True):
            count = self.threshold_counts[thresh]
            pct = (count / self.n_values) * 100 if self.n_values > 0 else 0
            print(f"p < {thresh:<10.4f}  {count:<20,} {pct:>12.4f}%")
        
        print(f"\n{'-'*80}")
        print(f"TOP PERCENTILE THRESHOLDS")
        print(f"{'-'*80}")
        print(f"{'Percentile':<15} {'Threshold':<20} {'Approx. Count':<20}")
        print(f"{'-'*15} {'-'*20} {'-'*20}")
        
        for p in sorted(self.target_percentiles):
            thresh = self.percentile_thresholds[p]
            count = int(self.n_values * p / 100.0)
            print(f"Top {p:<10.1f}%  {thresh:<20.6e} {count:<20,}")
        
        print(f"\n{'='*80}")
        print(f"Summary:")
        print(f"  - Most significant p-value: {self.min_val:.6e}")
        print(f"  - To keep top 1% connections: use p < {self.percentile_thresholds[1.0]:.6e}")
        print(f"  - To keep top 5% connections: use p < {self.percentile_thresholds[5.0]:.6e}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive p-value analysis for permutation test results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis with default settings
  %(prog)s ../02_cudaPerm/LTLEvsRTLE_run1.permout
  
  # Specify custom output prefix and more bins
  %(prog)s ../02_cudaPerm/LTLEvsRTLE_run1.permout -o my_analysis --bins 5000

Output:
  - Comprehensive report printed to console with:
    * Percentage of connections below standard p-value thresholds (0.05, 0.01, 0.001, 0.0005, 0.0001)
    * P-value thresholds for top percentiles (1%, 5%, 10%, 15%, 20%)
    * Basic statistics (min, max, mean, std dev)
  
  - Distribution file saved as CSV:
    * <prefix>_distribution.csv
    * Contains histogram of p-value distribution (log-scale bins)
    * Can be plotted later with a separate script

Memory efficiency:
  - Uses fixed-size histogram (~80KB for 10k bins)
  - No arrays of p-values stored
  - Suitable for files of any size
  - Total memory: <1 MB regardless of input size
        """
    )
    
    parser.add_argument('permout_file',
                       help='Path to .permout file with p-values')
    parser.add_argument('-o', '--output',
                       help='Output prefix for distribution file (default: auto-generate from input name)')
    parser.add_argument('--bins',
                       type=int,
                       default=10000,
                       help='Number of histogram bins (default: 10000, ~80KB memory)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.permout_file).exists():
        print(f"ERROR: Input file not found: {args.permout_file}")
        sys.exit(1)
    
    # Determine output prefix
    if args.output is None:
        output_prefix = args.permout_file.replace('.permout', '_pvalue_analysis')
    else:
        output_prefix = args.output
    
    dist_file = f"{output_prefix}_distribution.csv"

    # Detect format early so we can show it in the banner
    try:
        is_binary, gV, n_elem = detect_binary_format(args.permout_file)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    fmt_str    = "binary (PERT header)"  if is_binary else "text"
    n_elem_str = f"{n_elem:,}"           if is_binary else "(will count during pass 1)"

    # Show file info
    pval_size_gb = get_file_size_gb(args.permout_file)
    mem_usage_kb = args.bins * 8 / 1024  # 8 bytes per bin
    
    print(f"\n{'='*80}")
    print(f"Comprehensive P-value Analysis (Ultra-Low Memory)")
    print(f"{'='*80}")
    print(f"Input file:     {args.permout_file}")
    print(f"Format:         {fmt_str}")
    if is_binary:
        print(f"  gV (voxels):  {gV:,}")
        print(f"  n_elem:       {n_elem_str}")
    print(f"File size:      {pval_size_gb:.2f} GB")
    print(f"Histogram bins: {args.bins:,}")
    print(f"Memory usage:   ~{mem_usage_kb:.1f} KB")
    print(f"Output prefix:  {output_prefix}")
    
    # Create analyzer
    analyzer = StreamingPvalueAnalyzer(args.permout_file, n_bins=args.bins)
    
    # Pass 1: Build histogram and basic statistics
    analyzer.pass1_build_histogram()
    
    # Pass 2: Calculate percentile thresholds
    analyzer.pass2_calculate_percentiles()
    
    # Print comprehensive report
    analyzer.print_report()
    
    # Save distribution for plotting
    analyzer.save_distribution(dist_file)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"  Report: Displayed above")
    print(f"  Distribution: {dist_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
