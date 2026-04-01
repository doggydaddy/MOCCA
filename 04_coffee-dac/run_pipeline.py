#!/usr/bin/env python3
"""
run_pipeline.py – command-line interface for the COFFEE-DAC pipeline.

Runs the full dual hierarchical clustering on an input edge CSV and writes:
  <output>              – processed CSV with bundle and network label columns
  <output_stem>.npy     – companion scipy linkage matrix (binary NumPy file)

Usage examples
--------------
# Minimal – output path is derived automatically from the input name:
  python run_pipeline.py LTLEvsRTLE_run1_3mm_p001.csv

# Explicit output path:
  python run_pipeline.py LTLEvsRTLE_run1_3mm_p001.csv -o results/processed.csv

# Custom clustering parameters:
  python run_pipeline.py input.csv --bundles 30 --networks 7

# Force reprocessing even if a cached result already exists:
  python run_pipeline.py input.csv --reprocess
"""

import os
import sys

# ---------------------------------------------------------------------------
# Bootstrap: if we are not already running inside the project venv, re-exec
# with the venv interpreter so all dependencies (numpy, pandas, sklearn …)
# are available without requiring the caller to activate the venv manually.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_PYTHON = os.path.join(SCRIPT_DIR, '.venv', 'bin', 'python3')

if os.path.isfile(_VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(_VENV_PYTHON):
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import argparse
import time

# ---------------------------------------------------------------------------
# Resolve import path: allow running from any directory as long as the
# 04_coffee-dac folder (where coffee_dac_pipeline.py lives) is findable.
# ---------------------------------------------------------------------------
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from coffee_dac_pipeline import (
    process_edge_data,
    cache_exists,
    load_cached_result,
    get_cache_paths,
    _CACHE_COLUMNS,
    save_result,
    BUNDLE_COL,
    NETWORK_COL,
    estimate_nr_bundles,
)
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def make_progress_callback(label_width=40):
    """Return a callback that prints a simple terminal progress bar."""
    start = time.time()

    def callback(pct):
        elapsed = time.time() - start
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f'\r  [{bar}] {pct:3d}%  ({elapsed:5.1f}s)', end='', flush=True)
        if pct >= 100:
            print()  # newline when done

    return callback


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description='COFFEE-DAC pipeline: cluster edges into bundles and FCNs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        'input',
        metavar='INPUT_CSV',
        help='Path to the input edge CSV '
             '(columns: i1,j1,k1,i2,j2,k2,pvalue,tstat)',
    )
    p.add_argument(
        '-o', '--output',
        metavar='OUTPUT_CSV',
        default=None,
        help='Path for the processed output CSV.  '
             'Defaults to <input_stem>_processed.csv next to the input file.',
    )
    p.add_argument(
        '--bundles', '-b',
        type=int,
        default=None,
        metavar='N',
        help='Number of edge bundles (hc1 clusters).  '
             'When omitted the number is estimated automatically from the data '
             'using the definition: "a connection belongs to a bundle if an '
             'endpoint of the connection is neighboring any connection already '
             'in the bundle" (i.e. connected components under voxel-neighbour '
             'adjacency).  See also --neighbor-dist.',
    )
    p.add_argument(
        '--neighbor-dist',
        type=float,
        default=1.0,
        metavar='D',
        help='Chebyshev distance threshold (voxels) used when estimating the '
             'number of bundles automatically (default 1 → 26-connected '
             'neighbours).  Ignored when --bundles is given explicitly.',
    )
    p.add_argument(
        '--networks', '-n',
        type=int,
        default=5,
        metavar='N',
        help='Number of functional connectivity networks (hc2 clusters).',
    )
    p.add_argument(
        '--max-exact',
        type=int,
        default=50_000,
        metavar='N',
        help='Edge count threshold for exact vs. approximate clustering.  '
             'Datasets larger than this use subsampled hierarchical clustering '
             'with label propagation.',
    )
    p.add_argument(
        '--reprocess',
        action='store_true',
        default=False,
        help='Force the full pipeline to run even if a cached result exists.',
    )
    return p


def resolve_output_path(input_csv, output_arg):
    """
    Determine the final output CSV path.
    If the caller supplied --output that differs from the default cache path,
    we run the pipeline normally and then copy/rename the result to that path.
    """
    default_csv, _ = get_cache_paths(input_csv)
    if output_arg is None:
        return default_csv
    return os.path.abspath(output_arg)


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_csv = os.path.abspath(args.input)
    if not os.path.isfile(input_csv):
        print(f'ERROR: input file not found: {input_csv}', file=sys.stderr)
        sys.exit(1)

    output_csv = resolve_output_path(input_csv, args.output)
    output_npy = os.path.splitext(output_csv)[0] + '_linkage.npy'

    print(f'Input  : {input_csv}')
    print(f'Output : {output_csv}')
    print(f'Linkage: {output_npy}')
    bundles_display = args.bundles if args.bundles is not None else 'auto'
    print(f'Bundles: {bundles_display}   Networks: {args.networks}   '
          f'max_exact: {args.max_exact:,}'
          + (f'   neighbor_dist: {args.neighbor_dist}' if args.bundles is None else ''))
    print()

    # ------------------------------------------------------------------
    # Fast path: load from cache
    # ------------------------------------------------------------------
    default_csv, default_npy = get_cache_paths(input_csv)
    use_cache = (
        not args.reprocess
        and cache_exists(input_csv)
        # Only use the cache when the output destination matches the default
        # cache location; otherwise we must (re)save to the requested path.
        and os.path.abspath(default_csv) == output_csv
    )

    if use_cache:
        print('Cached result found – loading without reprocessing.')
        print('  (pass --reprocess to force a full pipeline run)')
        result = load_cached_result(input_csv)
        print('Done.')
    else:
        # ------------------------------------------------------------------
        # Full pipeline
        # ------------------------------------------------------------------
        print('Running pipeline…')
        progress = make_progress_callback()

        result = process_edge_data(
            input_csv,
            nr_bundles=args.bundles,
            nr_networks=args.networks,
            progress_callback=progress,
            max_exact=args.max_exact,
            neighbor_dist=args.neighbor_dist,
        )

        # process_edge_data already saves to the default cache path.
        # If the user asked for a different output path, write it there too.
        if os.path.abspath(default_csv) != output_csv:
            edges_net = result['edges_net']
            n_cols = edges_net.shape[1]
            cols = _CACHE_COLUMNS[:n_cols]
            if n_cols > len(_CACHE_COLUMNS):
                cols += [f'col{i}' for i in range(len(_CACHE_COLUMNS), n_cols)]
            os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
            pd.DataFrame(edges_net, columns=cols).to_csv(output_csv, index=False)
            np.save(output_npy, result['linkage_matrix'])
            print(f'Also saved to: {output_csv}')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    edges_net = result['edges_net']
    n_edges = edges_net.shape[0]
    n_bundles = int(edges_net[:, BUNDLE_COL].max()) + 1
    n_networks = int(edges_net[:, NETWORK_COL].max()) + 1

    print()
    print('─' * 50)
    print(f'  Edges    : {n_edges:,}')
    print(f'  Bundles  : {n_bundles}')
    print(f'  Networks : {n_networks}')
    print('─' * 50)
    print(f'  Output CSV     → {output_csv}')
    print(f'  Linkage matrix → {output_npy}')


if __name__ == '__main__':
    main()
