# coffee_dac_pipeline_v3.py
#
# Divisive companion to v1/v2's agglomerative bundle -> network clustering.
#
# v1/v2 assume you start with many small bundles that need merging UP into a
# handful of networks (hc2, average-linkage on bundle-to-bundle distance,
# cut via fcluster maxclust). FWER-corrected, percolation-calibrated
# inference produces the opposite shape: a few, but large, statistically
# significant "bundles" that are already network-scale, with nothing above
# them left to merge into.
#
# Hierarchical clustering has no native divisive primitive, so v3 goes the
# other direction the only way agglomerative clustering can: it treats the
# individual EDGE as the smallest available unit (exactly the leaves hc1
# already builds its tree from via h1_dist -- the same distance metric
# already used to form bundles from a raw thresholded edge pool in v1/v2),
# builds one linkage tree directly over the edges of an already-significant
# bundle, and cuts that tree at a chosen size to produce sub-bundles for
# visualization. This is the same "build the whole tree once, cut it
# wherever you like" trick as hc2/recut_networks, just one level down and
# skipping the bundle-formation step entirely, since the input here is
# already a single vetted, significant edge set.
#
# IMPORTANT: sub-bundling here is purely a rendering aid, exactly as the
# original bundle/network split always was. The p-value column is NEVER
# touched -- every edge keeps whatever whole-bundle FWER p-value it already
# carried from the statistical pipeline. Splitting a significant bundle into
# sub-bundles must never be read as assigning separate, uncorrected
# significance to any individual piece.

import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

from coffee_dac_pipeline import BUNDLE_COL, NETWORK_COL, _CACHE_COLUMNS, h1_dist
from coffee_dac_pipeline_v2 import _sha256_file, _utc_now, _software_metadata


# ---------------------------------------------------------------------------
# Cache paths specific to the v3 pipeline (different suffix so v1/v2/v3
# caches for the same input CSV never collide).
# ---------------------------------------------------------------------------

def get_cache_paths_v3(input_csv):
    '''
    Return the paths of the two v3 cache files that accompany *input_csv*:
      - <stem>_v3_processed.csv   - edge array with sub-bundle/network columns
      - <stem>_v3_linkage.npy     - scipy linkage matrix (Z) over individual edges
    '''
    base = os.path.splitext(input_csv)[0]
    return base + '_v3_processed.csv', base + '_v3_linkage.npy'


def get_params_path_v3(input_csv):
    '''Return the provenance sidecar path for a v3 cache set.'''
    base = os.path.splitext(input_csv)[0]
    return base + '_v3_params.json'


def is_processed_input_v3(input_csv):
    '''Return True when a v3 processed cache was supplied as raw input.'''
    return os.path.basename(os.fspath(input_csv)).lower().endswith(
        '_v3_processed.csv'
    )


def cache_exists_v3(input_csv):
    '''Return True if both v3 cache files exist for *input_csv*.'''
    csv_path, npy_path = get_cache_paths_v3(input_csv)
    return os.path.isfile(csv_path) and os.path.isfile(npy_path)


def load_params_v3(input_csv):
    '''Load a v3 provenance sidecar, returning None if absent/unreadable.'''
    path = get_params_path_v3(input_csv)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def cache_validation_v3(input_csv, expected_parameters=None):
    '''Return ``(valid, reason)`` for automatic reuse of a v3 cache.'''
    if not cache_exists_v3(input_csv):
        return False, 'processed CSV or linkage file is missing'

    manifest = load_params_v3(input_csv)
    if manifest is None:
        return False, 'parameter manifest is missing or unreadable'

    recorded_hash = manifest.get('input', {}).get('sha256')
    if not recorded_hash:
        return False, 'input checksum is missing from the manifest'
    try:
        if recorded_hash != _sha256_file(input_csv):
            return False, 'input CSV checksum differs from the manifest'
    except OSError as exc:
        return False, f'input CSV could not be hashed: {exc}'

    outputs = manifest.get('outputs', {})
    for output_path, hash_key, label in (
        (get_cache_paths_v3(input_csv)[0], 'processed_csv_sha256', 'processed CSV'),
        (get_cache_paths_v3(input_csv)[1], 'linkage_npy_sha256', 'linkage file'),
    ):
        expected_hash = outputs.get(hash_key)
        if not expected_hash:
            return False, f'{label} checksum is missing from the manifest'
        try:
            if expected_hash != _sha256_file(output_path):
                return False, f'{label} checksum differs from the manifest'
        except OSError as exc:
            return False, f'{label} could not be hashed: {exc}'

    if expected_parameters is not None:
        recorded = manifest.get('parameters')
        if recorded is not None and manifest.get('recuts'):
            recorded = dict(recorded)
            recorded['nr_bundles'] = manifest.get('results', {}).get(
                'bundles', recorded.get('nr_bundles')
            )
        if recorded != expected_parameters:
            return False, 'requested parameters differ from the manifest'

    return True, 'cache and manifest match'


def save_result_v3(input_csv, result, parameters=None, invocation='api',
                   started_at=None, recut=None, output_csv=None):
    '''
    Persist a v3 pipeline result dict to disk. Mirrors save_result_v2's
    on-disk shape exactly (same _CACHE_COLUMNS layout, same manifest schema)
    so existing v2 loading/rendering code needs no changes to also read v3
    caches -- only the file suffix and the pipeline label in the manifest
    differ.
    '''
    if output_csv is None:
        csv_path, npy_path = get_cache_paths_v3(input_csv)
        params_path = get_params_path_v3(input_csv)
    else:
        csv_path = os.path.abspath(output_csv)
        output_base = os.path.splitext(csv_path)[0]
        if output_base.endswith('_processed'):
            output_base = output_base[:-len('_processed')]
        npy_path = output_base + '_linkage.npy'
        params_path = output_base + '_params.json'

    edges_net = result['edges_net']
    n_cols = edges_net.shape[1]
    cols = _CACHE_COLUMNS[:n_cols]
    if n_cols > len(_CACHE_COLUMNS):
        cols += [f'col{i}' for i in range(len(_CACHE_COLUMNS), n_cols)]

    lm = result.get('linkage_matrix')
    cache_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(cache_dir, exist_ok=True)

    csv_tmp = tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', suffix='.csv', dir=cache_dir, delete=False,
    )
    npy_tmp = tempfile.NamedTemporaryFile(
        mode='wb', suffix='.npy', dir=cache_dir, delete=False,
    )
    try:
        with csv_tmp:
            pd.DataFrame(edges_net, columns=cols).to_csv(csv_tmp, index=False)
        with npy_tmp:
            np.save(npy_tmp, lm if lm is not None else np.array([]))
        os.replace(csv_tmp.name, csv_path)
        os.replace(npy_tmp.name, npy_path)
    finally:
        for temporary_path in (csv_tmp.name, npy_tmp.name):
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    manifest = None
    if parameters is None and os.path.isfile(params_path):
        try:
            with open(params_path, 'r', encoding='utf-8') as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            manifest = None
    new_manifest = manifest is None or parameters is not None
    if manifest is None:
        manifest = {
            'schema_version': 1,
            'pipeline': 'coffee-dac-v3-divisive',
            'parameters': parameters,
            'recuts': [],
        }

    input_rows = int(pd.read_csv(input_csv, usecols=[0]).shape[0])
    completed_at = _utc_now()
    manifest.update({
        'schema_version': 1,
        'pipeline': 'coffee-dac-v3-divisive',
        'last_updated_at': completed_at,
        'input': {
            'path': os.path.abspath(input_csv),
            'sha256': _sha256_file(input_csv),
            'rows': input_rows,
        },
        'results': {
            'retained_edges': int(edges_net.shape[0]),
            'bundles': int(len(np.unique(edges_net[:, BUNDLE_COL]))) if len(edges_net) else 0,
        },
        'outputs': {
            'processed_csv': os.path.basename(csv_path),
            'processed_csv_sha256': _sha256_file(csv_path),
            'linkage_npy': os.path.basename(npy_path),
            'linkage_npy_sha256': _sha256_file(npy_path),
        },
    })
    if new_manifest:
        manifest.update({
            'created_at': started_at or completed_at,
            'completed_at': completed_at,
            'invocation': invocation,
            'software': _software_metadata(),
        })
    if parameters is not None:
        manifest['parameters'] = parameters
        manifest['recuts'] = []
    if recut is not None:
        manifest.setdefault('recuts', []).append({
            'created_at': _utc_now(),
            'requested_bundles': int(recut['requested_bundles']),
            'actual_bundles': int(recut['actual_bundles']),
            'invocation': invocation,
        })

    json_tmp = tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', suffix='.json', dir=cache_dir, delete=False,
    )
    try:
        with json_tmp:
            json.dump(manifest, json_tmp, indent=2, sort_keys=True)
            json_tmp.write('\n')
        os.replace(json_tmp.name, params_path)
    finally:
        if os.path.exists(json_tmp.name):
            os.unlink(json_tmp.name)


def load_cached_result_v3(input_csv):
    '''
    Load a previously saved v3 pipeline result.

    Returns the same dict shape as process_edge_data_v3:
      { "edges_net": ndarray, "linkage_matrix": ndarray, "provenance": dict }
    '''
    csv_path, npy_path = get_cache_paths_v3(input_csv)
    edges_net = pd.read_csv(csv_path).to_numpy()
    linkage_matrix = np.load(npy_path)
    return {
        "edges_net": edges_net,
        "linkage_matrix": linkage_matrix,
        "provenance": load_params_v3(input_csv),
    }


# ---------------------------------------------------------------------------
# Divisive sub-bundling
# ---------------------------------------------------------------------------

def build_edge_linkage(edges, h1_flag='max', method='complete'):
    '''
    Build one hierarchical-clustering tree with individual EDGES as leaves,
    using the exact same edge-to-edge distance (h1_dist) that v1/v2 already
    use to form bundles from a raw thresholded edge pool. Only columns
    0:6 (the two endpoints) are read; any existing bundle/network columns
    on `edges` are ignored as input -- every edge is on equal footing as a
    leaf regardless of which upstream bundle it came from.

    Returns the scipy linkage matrix Z, shape (N-1, 4).
    '''
    condensed = h1_dist(edges, h1_flag)
    return linkage(condensed, method=method)


def recut_subbundles(edges, linkage_matrix, nr_bundles):
    '''
    Re-assign sub-bundle labels from an existing edge-level linkage matrix
    without recomputing distances. Mirrors recut_networks() in
    coffee_dac_pipeline_v2.py, but cuts directly to per-EDGE labels (there is
    no bundle-to-network indirection here) and writes BUNDLE_COL instead of
    NETWORK_COL. NETWORK_COL, pvalue, and tstat are left untouched -- this
    never invents or changes a significance value.

    Parameters
    ----------
    edges          : ndarray, shape (N, >=8), columns 0:8 = i1,j1,k1,i2,j2,k2,pvalue,tstat
    linkage_matrix : ndarray, shape (N-1, 4), from build_edge_linkage()
    nr_bundles     : int, desired number of sub-bundles (>=1, <=N)

    Returns
    -------
    edges_out  : ndarray with BUNDLE_COL (and NETWORK_COL if absent) set
    nr_out     : actual number of sub-bundles produced
    '''
    n_edges = edges.shape[0]
    edges_out = edges.copy()
    if edges_out.shape[1] <= BUNDLE_COL:
        edges_out = np.c_[
            edges_out, np.zeros((n_edges, BUNDLE_COL + 1 - edges_out.shape[1]))
        ]
    if edges_out.shape[1] <= NETWORK_COL:
        edges_out = np.c_[edges_out, np.zeros(edges_out.shape[0])]

    if linkage_matrix is None or linkage_matrix.shape[0] == 0 or n_edges <= 1:
        edges_out[:, BUNDLE_COL] = 0.0
        return edges_out, min(1, n_edges)

    max_bundles = int(linkage_matrix.shape[0]) + 1  # N-1 merges -> N leaves
    nr_out = max(1, min(nr_bundles, max_bundles))

    labels = fcluster(linkage_matrix, nr_out, criterion='maxclust') - 1  # zero-indexed

    # Re-index so sub-bundle 0 is the largest by edge count, matching the
    # convention hc1/hc2 already use.
    unique, counts = np.unique(labels, return_counts=True)
    order = sorted(zip(unique, counts), key=lambda item: -item[1])
    remap = {old: new for new, (old, _) in enumerate(order)}
    labels = np.array([remap[label] for label in labels], dtype=np.float64)

    edges_out[:, BUNDLE_COL] = labels
    print(f"recut_subbundles: {nr_out} sub-bundle(s) from {n_edges} edge(s)")
    return edges_out, nr_out


def process_edge_data_v3(input_csv, nr_bundles=2, h1_flag='max', method='complete',
                         max_exact=50_000, progress_callback=None,
                         invocation='api', allow_processed_input=False):
    '''
    V3 pipeline: divide an already-significant, already-bundled edge set
    (typically one FWER-significant bundle's exported edges) into sub-bundles
    for visualization only.

      1. Build one linkage tree over the individual edges (build_edge_linkage).
      2. Cut it into nr_bundles sub-bundles (recut_subbundles).

    Unlike v1/v2 there is no isolation filter, size filter, or endpoint-
    cluster pruning here: the input edges are assumed to already be the
    final, statistically validated set from the FWER pipeline. v3 only
    reorganizes them for legibility.

    Parameters
    ----------
    input_csv         : str, path to a bundle-level FWER visualization export
                         (e.g. from prepare_bundle_single_fwer.py), columns
                         i1,j1,k1,i2,j2,k2,pvalue,tstat[,bundle,network]
    nr_bundles        : int, target number of sub-bundles
    h1_flag           : 'min' | 'max' | 'mean', edge-to-edge endpoint distance
                         combination rule (default 'max', matching v1's
                         default bundle-forming distance)
    method            : 'complete' | 'average', scipy linkage method
                         (default 'complete', matching v1's original
                         edge-bundling call)
    max_exact         : if the input has more edges than this, refuse rather
                         than silently degrade to an approximate tree -- at
                         the scale this pipeline targets (single significant
                         bundles, historically tens of thousands of edges)
                         the exact O(N^2) distance matrix is affordable, and
                         an approximate tree would not support instant recut.

    Returns
    -------
    result : dict with keys edges_net, linkage_matrix, nr_bundles_out
    '''
    if is_processed_input_v3(input_csv) and not allow_processed_input:
        raise ValueError(
            "Refusing to process a '_v3_processed.csv' cache as raw input. "
            "Select the original CSV instead, or explicitly allow processed "
            "input if this is intentional."
        )

    started_at = _utc_now()
    parameters = {
        'nr_bundles': int(nr_bundles),
        'h1_flag': h1_flag,
        'method': method,
        'max_exact': int(max_exact),
    }

    edges_ijk = pd.read_csv(input_csv)
    edges = edges_ijk.to_numpy()
    n_edges = edges.shape[0]
    print(f"process_edge_data_v3: loaded {n_edges} edge(s) from '{input_csv}'")
    if progress_callback:
        progress_callback(5)

    if n_edges > max_exact:
        raise ValueError(
            f"process_edge_data_v3: {n_edges} edges exceeds max_exact="
            f"{max_exact}. An exact edge-level linkage tree needs O(N^2) "
            "distances; pass a smaller significant-bundle export, or raise "
            "max_exact explicitly if you have verified the memory/time cost."
        )

    linkage_matrix = build_edge_linkage(edges, h1_flag=h1_flag, method=method)
    if progress_callback:
        progress_callback(70)

    edges_out, nr_out = recut_subbundles(edges, linkage_matrix, nr_bundles)
    if progress_callback:
        progress_callback(90)

    result = {
        "edges_net": edges_out,
        "linkage_matrix": linkage_matrix,
        "nr_bundles_out": nr_out,
    }
    save_result_v3(input_csv, result, parameters=parameters, invocation=invocation,
                   started_at=started_at)
    if progress_callback:
        progress_callback(100)
    return result
