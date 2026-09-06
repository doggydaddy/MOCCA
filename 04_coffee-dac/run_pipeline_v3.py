#!/usr/bin/env python3
"""
run_pipeline_v3.py - command-line interface for the COFFEE-DAC v3 (divisive)
pipeline.

v3 is the companion to v1/v2 for the opposite problem: a single, already
FWER-significant "bundle" that is itself network-scale, with nothing above
it to merge into. Instead of clustering many small bundles UP into networks
(v1/v2's hc2), v3 builds one hierarchical-clustering tree directly over the
bundle's individual edges and cuts it DOWN into sub-bundles for
visualization. See coffee_dac_pipeline_v3.py's module docstring for the
full rationale.

The input is expected to be a bundle-level FWER visualization export (e.g.
from 03_prepResultsForVisualization/prepare_bundle_single_fwer.py or
prepare_bundle_grid_fwer.py) -- columns i1,j1,k1,i2,j2,k2,pvalue,tstat[,bundle,network].
The pvalue column is never modified: every sub-bundle keeps the same
whole-bundle FWER p-value it started with.

Writes:
  <output>                   - processed CSV with sub-bundle/network label columns
  <output_stem>_linkage.npy  - scipy linkage matrix over individual edges

Usage examples
--------------
# Minimal - output path derived automatically, split into 2 sub-bundles:
  python run_pipeline_v3.py controlsVSpatients_singleFWER.csv

# Split into 5 sub-bundles:
  python run_pipeline_v3.py controlsVSpatients_singleFWER.csv --bundles 5

# Re-cut an existing v3 cache into a different sub-bundle count (instant):
  python run_pipeline_v3.py controlsVSpatients_singleFWER.csv --recut 8
"""

import os
import sys

# ---------------------------------------------------------------------------
# Bootstrap: re-exec with the project venv interpreter when needed, so all
# dependencies (numpy, pandas, scipy ...) are available without the caller
# having to activate the venv manually.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MOCCA_ROOT = os.path.dirname(SCRIPT_DIR)
_VENV_PYTHON = os.path.join(_MOCCA_ROOT, '.venv', 'bin', 'python3')

if os.path.isfile(_VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(_VENV_PYTHON):
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import argparse
import time

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from coffee_dac_pipeline_v3 import (
    process_edge_data_v3,
    cache_exists_v3,
    cache_validation_v3,
    load_cached_result_v3,
    get_cache_paths_v3,
    is_processed_input_v3,
    save_result_v3,
    recut_subbundles,
)
from coffee_dac_pipeline import BUNDLE_COL, NETWORK_COL
import numpy as np


def _parse_per_parent_overrides(pairs):
    '''Parse repeated "PARENT_ID:N" strings into {parent_id: n}.'''
    overrides = {}
    for pair in pairs or []:
        try:
            parent_str, count_str = pair.split(':', 1)
            overrides[int(parent_str)] = int(count_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"expected 'PARENT_ID:N', got '{pair}'"
            )
    return overrides


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
            print()

    return callback


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            'COFFEE-DAC v3 (divisive) pipeline: build one hierarchical-'
            'clustering tree over an already-significant bundle\'s edges, '
            'then cut it into N sub-bundles for visualization only.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        'input',
        metavar='INPUT_CSV',
        help='Path to a bundle-level FWER visualization export '
             '(columns: i1,j1,k1,i2,j2,k2,pvalue,tstat[,bundle,network])',
    )
    p.add_argument(
        '-o', '--output',
        metavar='OUTPUT_CSV',
        default=None,
        help='Path for the processed output CSV. '
             'Defaults to <input_stem>_v3_processed.csv next to the input file.',
    )
    p.add_argument(
        '--bundles', '-n',
        type=int,
        default=2,
        metavar='N',
        help='Default target number of sub-bundles for every parent '
             '(inferential) bundle in the input. A single visualization '
             'export can contain more than one independently FWER-'
             'significant bundle; each is subdivided by its own '
             'independent edge-linkage tree (see coffee_dac_pipeline_v3.py) '
             'and never mixed with another parent bundle.',
    )
    p.add_argument(
        '--bundles-for',
        action='append',
        metavar='PARENT_ID:N',
        help='Override the sub-bundle count for one parent bundle, e.g. '
             '"94:3". Repeatable. Parent bundle ids are whatever is in the '
             'bundle/network columns of the input (or its sibling '
             '_v2_processed.csv) -- see the printed "parent bundle" summary '
             'from a first run to find them.',
    )
    p.add_argument(
        '--h1-flag',
        choices=('min', 'max', 'mean'),
        default='max',
        help="Edge-to-edge endpoint distance combination rule, matching "
             "v1's h1_dist (default 'max').",
    )
    p.add_argument(
        '--method',
        choices=('complete', 'average'),
        default='complete',
        help="scipy linkage method, matching v1's original edge-bundling "
             "call (default 'complete').",
    )
    p.add_argument(
        '--max-exact',
        type=int,
        default=50_000,
        metavar='N',
        help='Refuse to process more than this many edges: an exact edge-'
             'level linkage tree needs O(N^2) distances.',
    )
    p.add_argument(
        '--reprocess',
        action='store_true',
        default=False,
        help='Force the full pipeline to run even if a cached result exists.',
    )
    p.add_argument(
        '--recut',
        type=int,
        default=None,
        metavar='N',
        help='Re-cut an existing cached result: N sub-bundles for every '
             'parent bundle by default (override individual parents with '
             '--recut-for), without re-running the full pipeline. Requires '
             'a v3 cache to exist. Overwrites the processed CSV with '
             'updated sub-bundle labels.',
    )
    p.add_argument(
        '--recut-for',
        action='append',
        metavar='PARENT_ID:N',
        help='Override the re-cut sub-bundle count for one parent bundle, '
             'e.g. "94:3". Repeatable. Requires --recut to also be given '
             '(as the default for any parent not listed here).',
    )
    p.add_argument(
        '--allow-processed-input',
        action='store_true',
        default=False,
        help="Allow a file ending in '_v3_processed.csv' to be treated as "
             'raw input. Blocked by default to prevent nested cache names.',
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_csv = os.path.abspath(args.input)
    if not os.path.isfile(input_csv):
        print(f'ERROR: input file not found: {input_csv}', file=sys.stderr)
        sys.exit(1)

    if is_processed_input_v3(input_csv) and not args.allow_processed_input:
        print(
            "ERROR: refusing to process a '_v3_processed.csv' cache as raw "
            'input. Select the original CSV, or pass '
            '--allow-processed-input if this is intentional.',
            file=sys.stderr,
        )
        sys.exit(2)

    default_csv, default_npy = get_cache_paths_v3(input_csv)
    output_csv = default_csv if args.output is None else os.path.abspath(args.output)

    if args.recut is not None or args.recut_for:
        if not cache_exists_v3(input_csv):
            print('ERROR: --recut/--recut-for requires an existing v3 '
                  'cache. Run the pipeline first.', file=sys.stderr)
            sys.exit(1)
        recut_overrides = _parse_per_parent_overrides(args.recut_for)
        if recut_overrides and args.recut is None:
            print('ERROR: --recut-for requires --recut to also be given, '
                  'as the default for any parent bundle not listed.',
                  file=sys.stderr)
            sys.exit(2)
        default_recut = args.recut if args.recut is not None else 2
        print(f'Re-cutting cached result (default {default_recut} '
              f'sub-bundle(s) per parent bundle'
              + (f', overrides: {recut_overrides}' if recut_overrides else '')
              + ')...')
        cached = load_cached_result_v3(input_csv)
        requested = recut_overrides if recut_overrides else default_recut
        edges_out, nr_out_map = recut_subbundles(
            cached['edges_net'], cached['linkage_matrices'], requested,
            default_nr_bundles=default_recut,
        )
        cached['edges_net'] = edges_out
        cached['nr_bundles_out'] = nr_out_map
        save_result_v3(
            input_csv,
            cached,
            invocation='cli',
            recut={
                'requested_bundles': {
                    pid: recut_overrides.get(pid, default_recut)
                    for pid in nr_out_map
                },
                'actual_bundles': nr_out_map,
            },
        )
        for parent_id, nr_out in sorted(nr_out_map.items()):
            print(f'  -> parent bundle {parent_id}: {nr_out} sub-bundle(s)')
        print(f'  saved to {default_csv}')
        return

    bundle_overrides = _parse_per_parent_overrides(args.bundles_for)
    nr_bundles = bundle_overrides if bundle_overrides else int(args.bundles)

    print(f'Input        : {input_csv}')
    print(f'Output       : {output_csv}')
    print(f'bundles      : {args.bundles} (default per parent bundle)')
    if bundle_overrides:
        print(f'bundles-for  : {bundle_overrides}')
    print(f'h1_flag      : {args.h1_flag}')
    print(f'method       : {args.method}')
    print(f'max_exact    : {args.max_exact}')
    print()

    expected_parameters = {
        'nr_bundles': nr_bundles,
        'default_nr_bundles': int(args.bundles),
        'h1_flag': args.h1_flag,
        'method': args.method,
        'max_exact': int(args.max_exact),
    }
    cache_valid, cache_reason = cache_validation_v3(input_csv, expected_parameters)
    use_cache = (
        not args.reprocess
        and cache_valid
        and os.path.abspath(default_csv) == output_csv
    )

    if use_cache:
        print('Cached v3 result found - loading without reprocessing.')
        print('  (pass --reprocess to force a full pipeline run)')
        result = load_cached_result_v3(input_csv)
        print('Done.')
    else:
        print('Running v3 pipeline...')
        if cache_exists_v3(input_csv) and not args.reprocess:
            print(f'Existing cache not reused: {cache_reason}.')
        progress = make_progress_callback()

        result = process_edge_data_v3(
            input_csv,
            nr_bundles=nr_bundles,
            default_nr_bundles=int(args.bundles),
            h1_flag=args.h1_flag,
            method=args.method,
            max_exact=args.max_exact,
            progress_callback=progress,
            invocation='cli',
            allow_processed_input=args.allow_processed_input,
        )

        if os.path.abspath(default_csv) != output_csv:
            save_result_v3(
                input_csv,
                result,
                parameters=expected_parameters,
                invocation='cli',
                output_csv=output_csv,
            )
            print(f'Also saved result set for: {output_csv}')

    edges_net = result['edges_net']
    n_edges = edges_net.shape[0]
    nr_bundles_out = result.get('nr_bundles_out')
    if not nr_bundles_out and n_edges:
        # Loaded from a legacy/plain cache dict without nr_bundles_out --
        # derive it per parent bundle directly from the edge array.
        nr_bundles_out = {
            int(parent_id): len(np.unique(
                edges_net[edges_net[:, NETWORK_COL] == parent_id, BUNDLE_COL]
            ))
            for parent_id in np.unique(edges_net[:, NETWORK_COL])
        }
    nr_bundles_out = nr_bundles_out or {}

    print()
    print('-' * 52)
    print(f'  Input/output edges : {n_edges:,}')
    print(f'  Parent bundles     : {len(nr_bundles_out)}')
    for parent_id, nr_out in sorted(nr_bundles_out.items()):
        print(f'    parent {parent_id}: {nr_out} sub-bundle(s)')
    print('-' * 52)
    print(f'  Output CSV  -> {output_csv}')


if __name__ == '__main__':
    main()
