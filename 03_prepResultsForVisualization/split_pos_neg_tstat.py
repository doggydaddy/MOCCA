#!/usr/bin/env python3
"""
Split a filtered connection CSV (e.g. from permout_to_csv.py) into
separate files for positive and negative t-statistics.

Usage:
    python split_pos_neg_tstat.py <input_csv>

Output files are written alongside the input:
    <basename>_pos.csv
    <basename>_neg.csv
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Split a connection CSV into positive and negative tstat subsets.")
    parser.add_argument("input_csv", help="Input CSV file with a 'tstat' column")
    args = parser.parse_args()

    inpath = Path(args.input_csv)
    if not inpath.exists():
        print(f"ERROR: file not found: {inpath}", file=sys.stderr)
        sys.exit(1)

    pos_path = inpath.with_name(inpath.stem + "_pos.csv")
    neg_path = inpath.with_name(inpath.stem + "_neg.csv")

    n_pos = 0
    n_neg = 0
    n_zero = 0

    with open(inpath) as f_in, \
         open(pos_path, "w") as f_pos, \
         open(neg_path, "w") as f_neg:

        # Read and validate header
        header = f_in.readline().rstrip("\n")
        cols = header.split(",")
        if "tstat" not in cols:
            print(f"ERROR: no 'tstat' column found in {inpath}", file=sys.stderr)
            print(f"       Columns found: {cols}", file=sys.stderr)
            sys.exit(1)
        tstat_idx = cols.index("tstat")

        f_pos.write(header + "\n")
        f_neg.write(header + "\n")

        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            val = float(line.split(",")[tstat_idx])
            if val > 0:
                f_pos.write(line + "\n")
                n_pos += 1
            elif val < 0:
                f_neg.write(line + "\n")
                n_neg += 1
            else:
                # exactly zero — write to both and count separately
                f_pos.write(line + "\n")
                f_neg.write(line + "\n")
                n_zero += 1

    total = n_pos + n_neg + n_zero
    print(f"Input  : {inpath.name}")
    print(f"Total connections : {total:>10,}")
    print(f"  Positive tstat  : {n_pos:>10,}  -> {pos_path.name}")
    print(f"  Negative tstat  : {n_neg:>10,}  -> {neg_path.name}")
    if n_zero:
        print(f"  Exactly zero    : {n_zero:>10,}  (written to both)")

if __name__ == "__main__":
    main()
