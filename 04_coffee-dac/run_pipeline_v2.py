#!/usr/bin/env python3
"""
run_pipeline_v2.py – command-line interface for the COFFEE-DAC v2 pipeline.

V2 differs from v1 by adding:
  1. Optional tstat pre-filter (--top-n / --tstat-threshold).
  2. Isolation filter before bundling.
  3. Iterative intra-bundle pruning of isolated endpoints.
  4. Endpoint-cluster size pruning (--min-cluster-voxels).
  5. Bundling uses strict connected-components (not approximate hc1), so
     bundles are tight and density-robust.  Network-level clustering (hc2)
     then groups bundles into FCNs as in v1.

Writes:
  <output>   – processed CSV with bundle and network label columns
  <output_stem>_linkage.npy – scipy linkage matrix

Usage examples
--------------
# Minimal – output path derived automatically:
  python run_pipeline_v2.py LTLEvsRTLE_run1_3mm_p001.csv

# Keep only top 5000 connections, form 8 networks:
  python run_pipeline_v2.py input.csv --top-n 5000 --networks 8

# Explicit output path:
  python run_pipeline_v2.py input.csv -o results/processed_v2.csv

# Wider neighbourhood:
  python run_pipeline_v2.py input.csv --neighbor-dist 2.0

# Force reprocessing even if a cached result already exists:
  python run_pipeline_v2.py input.csv --reprocess
"""

import os
import sys

# ---------------------------------------------------------------------------
# Bootstrap: re-exec with the project venv interpreter when needed, so all
# dependencies (numpy, pandas, scipy …) are available without the caller
# having to activate the venv manually.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_PYTHON = os.path.join(SCRIPT_DIR, '.venv', 'bin', 'python3')

if os.path.isfile(_VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(_VENV_PYTHON):
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import argparse
import time

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from coffee_dac_pipeline_v2 import (
    process_edge_data_v2,
    cache_exists_v2,
    load_cached_result_v2,
    get_cache_paths_v2,
    save_result_v2,
    recut_networks,
    TSTAT_COL,
)
from coffee_dac_pipeline import (
    BUNDLE_COL,
    NETWORK_COL,
    _CACHE_COLUMNS,
)
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def make_progress_callback():
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
        description=(
            'COFFEE-DAC v2 pipeline: tstat filter → isolation filter → '
            'connected-component bundling → endpoint pruning → hc2 networks.'
        ),
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
             'Defaults to <input_stem>_v2_processed.csv next to the input file.',
    )
    p.add_argument(
        '--networks', '-n',
        type=int,
        default=5,
        metavar='N',
        help='Number of functional connectivity networks (hc2 clusters).',
    )
    p.add_argument(
        '--top-n',
        type=int,
        default=None,
        metavar='N',
        help='Before spatial filtering, keep only the N connections with the '
             'highest t-statistic.  Mutually exclusive with --tstat-threshold.',
    )
    p.add_argument(
        '--tstat-threshold',
        type=float,
        default=None,
        metavar='T',
        help='Before spatial filtering, keep only connections with '
             'tstat >= T.  Mutually exclusive with --top-n.',
    )
    p.add_argument(
        '--min-size',
        type=int,
        default=2,
        metavar='N',
        help='Drop bundles/networks with fewer than N edges (default 2).',
    )
    p.add_argument(
        '--min-cluster-voxels',
        type=int,
        default=3,
        metavar='N',
        help='Within each bundle, drop connections touching endpoint-voxel '
             'clusters smaller than N voxels (default 3).  Use 1 to disable.',
    )
    p.add_argument(
        '--neighbor-dist',
        type=float,
        default=1.0,
        metavar='D',
        help='Chebyshev distance threshold (voxels) used throughout '
             '(default 1 → 26-connected voxel neighbours).',
    )
    p.add_argument(
        '--reprocess',
        action='store_true',
        default=False,
        help='Force the full pipeline to run even if a cached result exists.',
    )
    p.add_argument(
        '--strict-bundles',
        action='store_true',
        default=False,
        help='Use the strict bundling mode: connections are grouped only when '
             'they share a common endpoint voxel AND their other endpoints '
             'are within --neighbor-dist.  Produces many smaller bundles on '
             'dense datasets; avoids giant components.',
    )
    p.add_argument(
        '--recut',
        type=int,
        default=None,
        metavar='N',
        help='Re-cut an existing cached result into N networks without '
             're-running the full pipeline.  Requires a v2 cache to exist. '
             'Overwrites the processed CSV with updated network labels.',
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_csv = os.path.abspath(args.input)
    if not os.path.isfile(input_csv):
        print(f'ERROR: input file not found: {input_csv}', file=sys.stderr)
        sys.exit(1)

    if args.top_n is not None and args.tstat_threshold is not None:
        print('ERROR: --top-n and --tstat-threshold are mutually exclusive.',
              file=sys.stderr)
        sys.exit(1)

    # Resolve output paths
    default_csv, default_npy = get_cache_paths_v2(input_csv)
    if args.output is None:
        output_csv = default_csv
    else:
        output_csv = os.path.abspath(args.output)

    # ------------------------------------------------------------------
    # Fast path: --recut  (re-cut an existing linkage without reprocessing)
    # ------------------------------------------------------------------
    if args.recut is not None:
        if not cache_exists_v2(input_csv):
            print('ERROR: --recut requires an existing v2 cache. '
                  'Run the pipeline first.', file=sys.stderr)
            sys.exit(1)
        print(f'Re-cutting cached result into {args.recut} network(s)…')
        cached = load_cached_result_v2(input_csv)
        edges_out, nr_net = recut_networks(
            cached['edges_net'], cached['linkage_matrix'], args.recut
        )
        cached['edges_net'] = edges_out
        cached['nr_networks_out'] = nr_net
        save_result_v2(input_csv, cached)
        print(f'  → {nr_net} network(s) saved to {default_csv}')
        return

    print(f'Input        : {input_csv}')
    print(f'Output       : {output_csv}')
    if args.top_n is not None:
        print(f'tstat filter : top {args.top_n:,} connections')
    elif args.tstat_threshold is not None:
        print(f'tstat filter : >= {args.tstat_threshold}')
    else:
        print(f'tstat filter : none')
    print(f'networks          : {args.networks}')
    print(f'min_size          : {args.min_size}')
    print(f'min_cluster_voxels: {args.min_cluster_voxels}')
    print(f'neighbor_dist     : {args.neighbor_dist}')
    print(f'strict_bundles    : {args.strict_bundles}')
    print()

    # ------------------------------------------------------------------
    # Fast path: load from cache
    # ------------------------------------------------------------------
    # Any tstat filter or non-default min_size/min_cluster_voxels makes cache stale
    tstat_filtering = args.top_n is not None or args.tstat_threshold is not None
    use_cache = (
        not args.reprocess
        and not tstat_filtering
        and not args.strict_bundles
        and args.networks == 5
        and args.min_size == 2
        and args.min_cluster_voxels == 3
        and cache_exists_v2(input_csv)
        # Only use the default cache when the output destination matches it;
        # otherwise we must rerun to write to the requested path.
        and os.path.abspath(default_csv) == output_csv
    )

    if use_cache:
        print('Cached v2 result found – loading without reprocessing.')
        print('  (pass --reprocess to force a full pipeline run)')
        result = load_cached_result_v2(input_csv)
        print('Done.')
    else:
        # ------------------------------------------------------------------
        # Full v2 pipeline
        # ------------------------------------------------------------------
        print('Running v2 pipeline…')
        progress = make_progress_callback()

        result = process_edge_data_v2(
            input_csv,
            progress_callback=progress,
            neighbor_dist=args.neighbor_dist,
            top_n=args.top_n,
            tstat_threshold=args.tstat_threshold,
            min_network_size=args.min_size,
            min_cluster_voxels=args.min_cluster_voxels,
            nr_networks=args.networks,
            strict_bundles=args.strict_bundles,
        )

        # process_edge_data_v2 already saves to the default v2 cache path.
        # If the user asked for a different output path, write there too.
        if os.path.abspath(default_csv) != output_csv:
            edges_net = result['edges_net']
            n_cols = edges_net.shape[1]
            cols = _CACHE_COLUMNS[:n_cols]
            if n_cols > len(_CACHE_COLUMNS):
                cols += [f'col{i}' for i in range(len(_CACHE_COLUMNS), n_cols)]
            os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
            pd.DataFrame(edges_net, columns=cols).to_csv(output_csv, index=False)
            print(f'Also saved to: {output_csv}')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    edges_net  = result['edges_net']
    n_edges    = edges_net.shape[0]
    n_networks = result.get('nr_networks_out', int(edges_net[:, NETWORK_COL].max()) + 1 if n_edges else 0)
    kept_mask  = result.get('kept_mask')
    tstat_mask = result.get('tstat_mask')

    n_original = int(kept_mask.shape[0]) if kept_mask is not None else '?'
    n_removed  = int((~kept_mask).sum()) if kept_mask is not None else '?'

    print()
    print('─' * 52)
    if tstat_mask is not None:
        n_tstat_kept = int(tstat_mask.sum())
        n_tstat_removed = int((~tstat_mask).sum())
        print(f'  Input edges       : {n_original:,}')
        print(f'  Removed (tstat)   : {n_tstat_removed:,}  → {n_tstat_kept:,} kept')
    else:
        print(f'  Input edges       : {n_original:,}')
    n_isolation_removed = n_removed - (int((~tstat_mask).sum()) if tstat_mask is not None else 0)
    print(f'  Removed (isolated): {n_isolation_removed:,}')
    n_size_removed = int(result['size_mask'].shape[0] - result['size_mask'].sum()) if result.get('size_mask') is not None else 0
    if n_size_removed:
        print(f'  Removed (size<{args.min_size:2d}) : {n_size_removed:,}')
    prune_mask = result.get('prune_mask')
    if prune_mask is not None:
        n_pruned = int((~prune_mask).sum())
        if n_pruned:
            print(f'  Removed (pruned)  : {n_pruned:,}')
    ep_mask = result.get('ep_cluster_mask')
    if ep_mask is not None:
        n_ep_dropped = int((~ep_mask).sum())
        if n_ep_dropped:
            print(f'  Removed (ep clust): {n_ep_dropped:,}')
    print(f'  Edges in networks : {n_edges:,}')
    print(f'  Networks (hc2)    : {n_networks}')
    print('─' * 52)
    print(f'  Output CSV  → {output_csv}')


if __name__ == '__main__':
    main()
