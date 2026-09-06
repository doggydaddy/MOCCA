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
#
# PARENT-BUNDLE AWARENESS: a single visualization export (e.g. from
# prepare_bundle_single_fwer.py) can legitimately contain more than one
# independently-significant FWER bundle -- e.g. two bundles both survive
# FWER at alpha=0.05. These are separate inferential findings that happen to
# ride in the same CSV; they must never be pooled into one edge-distance
# tree together, since that silently treats "close enough in the pooled
# tree" as if it meant something, when the only thing that made either set
# of edges significant was the whole-bundle FWER test performed upstream.
# v3 therefore builds and cuts one independent linkage tree PER parent
# bundle, and never lets an edge from one parent influence which sub-bundle
# an edge from a different parent lands in. Column convention for v3 output:
#   NETWORK_COL - the parent (inferential) bundle id this edge belongs to.
#   BUNDLE_COL  - the display subdivision id within that parent, independently
#                 zero-indexed largest-first per parent (so, unlike v1/v2,
#                 BUNDLE_COL values repeat across different parents -- always
#                 pair it with NETWORK_COL, exactly as the rest of the GUI's
#                 FCN-then-bundle tree already assumes).

import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

from coffee_dac_pipeline import BUNDLE_COL, NETWORK_COL, _CACHE_COLUMNS, h1_dist
from coffee_dac_pipeline_v2 import (
    _sha256_file, _utc_now, _software_metadata, get_cache_paths_v2,
)


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
            last_recut = manifest['recuts'][-1]
            # A later --recut overrides the originally-processed nr_bundles;
            # compare against what it actually requested per parent bundle,
            # not the summed subdivision count (which can't be inverted back
            # into a per-parent request).
            recorded['nr_bundles'] = {
                int(parent_id): count
                for parent_id, count in last_recut.get('requested_bundles', {}).items()
            }
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

    ``result['linkage_matrices']`` is a dict {parent_bundle_id: Z or None},
    one independent edge-linkage tree per parent bundle (see module
    docstring) -- saved as a single pickled object array so the existing
    one-file-per-cache-slot layout (``_v3_linkage.npy``) doesn't need to grow
    a second file per parent.
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

    linkage_matrices = result.get('linkage_matrices') or {}
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
            np.save(npy_tmp, np.asarray(linkage_matrices, dtype=object),
                    allow_pickle=True)
        os.replace(csv_tmp.name, csv_path)
        os.replace(npy_tmp.name, npy_path)
    finally:
        for temporary_path in (csv_tmp.name, npy_tmp.name):
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    parent_ids = edges_net[:, NETWORK_COL].astype(int) if len(edges_net) else np.array([], dtype=int)
    parent_summaries = []
    for parent_id in sorted(np.unique(parent_ids).tolist()):
        parent_rows = edges_net[parent_ids == parent_id]
        parent_summaries.append({
            'parent_bundle_id': int(parent_id),
            'edge_count': int(len(parent_rows)),
            'subdivisions': int(len(np.unique(parent_rows[:, BUNDLE_COL]))),
        })

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
            # Total distinct (parent, subdivision) pairs across every parent
            # bundle -- BUNDLE_COL alone under-counts this since subdivision
            # ids restart at 0 within each parent (see module docstring).
            'bundles': sum(entry['subdivisions'] for entry in parent_summaries),
            'parent_bundle_count': len(parent_summaries),
            'parent_bundles': parent_summaries,
            'parent_label_source': result.get('parent_label_source'),
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
        # requested_bundles/actual_bundles are per-parent: {parent_id: n}.
        manifest.setdefault('recuts', []).append({
            'created_at': _utc_now(),
            'requested_bundles': {
                str(k): int(v) for k, v in recut['requested_bundles'].items()
            },
            'actual_bundles': {
                str(k): int(v) for k, v in recut['actual_bundles'].items()
            },
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
      { "edges_net": ndarray, "linkage_matrices": dict, "provenance": dict }
    linkage_matrices maps {parent_bundle_id (int): Z ndarray or None}, one
    independent edge-linkage tree per parent bundle (see module docstring).
    '''
    csv_path, npy_path = get_cache_paths_v3(input_csv)
    edges_net = pd.read_csv(csv_path).to_numpy()
    linkage_matrices = np.load(npy_path, allow_pickle=True).item()
    return {
        "edges_net": edges_net,
        "linkage_matrices": linkage_matrices,
        "provenance": load_params_v3(input_csv),
    }


# ---------------------------------------------------------------------------
# Divisive sub-bundling
# ---------------------------------------------------------------------------

def _resolve_parent_ids(input_csv, edges):
    '''
    Recover the parent (inferential) bundle id for every edge, so subdivision
    can stay independent per parent (see module docstring). Priority:

      1. `edges` already carries a bundle column (>= 9 columns) -- e.g. the
         input is a `_v2_processed.csv`-shaped export, or any CSV that
         already labels each edge's parent bundle. Used directly.
      2. A sibling `<stem>_v2_processed.csv` next to a bare raw export
         (prepare_bundle_single_fwer.py / prepare_bundle_grid_fwer.py write
         both files from the same source rows in the same loop iteration,
         so they align positionally row-for-row). Used only if its row
         count matches `edges` exactly.
      3. Otherwise, one implicit parent bundle (id 0) spanning every edge --
         the original single-bundle behavior, unchanged.

    Returns (parent_ids, source), where source is one of
    'input_bundle_column', 'sibling_v2_processed_csv', or
    'implicit_single_parent'.
    '''
    if edges.shape[1] > BUNDLE_COL:
        return edges[:, BUNDLE_COL].astype(int), 'input_bundle_column'

    sibling_csv, _ = get_cache_paths_v2(input_csv)
    if os.path.isfile(sibling_csv):
        sibling = pd.read_csv(sibling_csv, usecols=['bundle'])
        if len(sibling) == len(edges):
            return (
                sibling['bundle'].to_numpy().astype(int),
                'sibling_v2_processed_csv',
            )
        print(
            f"process_edge_data_v3: sibling '{sibling_csv}' has "
            f"{len(sibling)} row(s) but input has {len(edges)} -- cannot use "
            "it to recover parent-bundle labels; falling back to one "
            "implicit parent bundle."
        )

    return np.zeros(edges.shape[0], dtype=int), 'implicit_single_parent'


def build_edge_linkage(edges, h1_flag='max', method='complete'):
    '''
    Build one hierarchical-clustering tree with individual EDGES as leaves,
    using the exact same edge-to-edge distance (h1_dist) that v1/v2 already
    use to form bundles from a raw thresholded edge pool. Only columns 0:6
    (the two endpoints) are read.

    Returns the scipy linkage matrix Z, shape (N-1, 4), or None when `edges`
    has fewer than 2 rows (nothing to link).
    '''
    if edges.shape[0] < 2:
        return None
    condensed = h1_dist(edges, h1_flag)
    return linkage(condensed, method=method)


def build_edge_linkage_per_parent(edges, parent_ids, h1_flag='max', method='complete'):
    '''
    Build one independent edge-linkage tree PER parent bundle (see module
    docstring): edges belonging to different parent bundles never influence
    each other's distances or tree structure, so one parent's internal
    structure can never leak into how another parent gets subdivided.

    Returns {parent_bundle_id (int): Z ndarray or None}.
    '''
    linkage_matrices = {}
    for parent_id in np.unique(parent_ids):
        parent_edges = edges[parent_ids == parent_id]
        linkage_matrices[int(parent_id)] = build_edge_linkage(
            parent_edges, h1_flag=h1_flag, method=method
        )
    return linkage_matrices


def _cut_one_parent(parent_edges, linkage_matrix, nr_bundles):
    '''Cut one parent bundle's edge-linkage tree into nr_bundles sub-bundles.

    Returns (labels, nr_out), labels zero-indexed with 0 = the largest
    sub-bundle by edge count (matching the convention hc1/hc2 already use).
    '''
    n_edges = parent_edges.shape[0]
    if linkage_matrix is None or linkage_matrix.shape[0] == 0 or n_edges <= 1:
        return np.zeros(n_edges, dtype=np.float64), min(1, n_edges)

    max_bundles = int(linkage_matrix.shape[0]) + 1  # N-1 merges -> N leaves
    nr_out = max(1, min(int(nr_bundles), max_bundles))

    labels = fcluster(linkage_matrix, nr_out, criterion='maxclust') - 1  # zero-indexed

    unique, counts = np.unique(labels, return_counts=True)
    order = sorted(zip(unique, counts), key=lambda item: -item[1])
    remap = {old: new for new, (old, _) in enumerate(order)}
    labels = np.array([remap[label] for label in labels], dtype=np.float64)
    return labels, nr_out


def recut_subbundles(edges, linkage_matrices, nr_bundles, default_nr_bundles=2):
    '''
    Re-assign sub-bundle labels from existing edge-level linkage matrices
    without recomputing distances, independently per parent bundle (NEVER
    pooling edges across parents -- see module docstring). Mirrors
    recut_networks() in coffee_dac_pipeline_v2.py, but cuts directly to
    per-EDGE labels within each parent (there is no bundle-to-network
    indirection here) and writes BUNDLE_COL; NETWORK_COL (the parent bundle
    id), pvalue, and tstat are left untouched -- this never invents or
    changes a significance value.

    Parameters
    ----------
    edges              : ndarray, shape (N, >=8). Parent-bundle membership is
                         read from NETWORK_COL if present; if `edges` lacks a
                         NETWORK_COL, every edge is treated as one implicit
                         parent bundle (id 0).
    linkage_matrices    : dict {parent_bundle_id: Z or None}, from
                         build_edge_linkage_per_parent(). A bare Z ndarray
                         (or None) is also accepted as shorthand for "use
                         this same tree for every parent bundle present" --
                         only sound when there is in fact a single parent.
    nr_bundles          : int (applied to every parent bundle) or dict
                         {parent_bundle_id: int} for independent per-parent
                         sub-bundle counts.
    default_nr_bundles  : fallback sub-bundle count for a parent bundle not
                         named in an `nr_bundles` dict.

    Returns
    -------
    edges_out  : ndarray with BUNDLE_COL (and NETWORK_COL if absent) set
    nr_out_map : dict {parent_bundle_id: actual number of sub-bundles produced}
    '''
    n_edges = edges.shape[0]
    edges_out = edges.copy()
    if edges_out.shape[1] <= BUNDLE_COL:
        edges_out = np.c_[
            edges_out, np.zeros((n_edges, BUNDLE_COL + 1 - edges_out.shape[1]))
        ]
    if edges_out.shape[1] <= NETWORK_COL:
        edges_out = np.c_[
            edges_out, np.zeros((n_edges, NETWORK_COL + 1 - edges_out.shape[1]))
        ]

    if n_edges == 0:
        return edges_out, {}

    parent_ids = edges_out[:, NETWORK_COL].astype(int)

    if not isinstance(linkage_matrices, dict):
        # Shorthand: one Z (or None) applied to every parent present.
        linkage_matrices = {
            int(parent_id): linkage_matrices for parent_id in np.unique(parent_ids)
        }

    nr_out_map = {}
    for parent_id in np.unique(parent_ids):
        parent_id = int(parent_id)
        mask = parent_ids == parent_id
        if isinstance(nr_bundles, dict):
            requested = nr_bundles.get(parent_id, default_nr_bundles)
        else:
            requested = nr_bundles
        labels, nr_out = _cut_one_parent(
            edges_out[mask], linkage_matrices.get(parent_id), requested
        )
        edges_out[mask, BUNDLE_COL] = labels
        nr_out_map[parent_id] = nr_out
        print(
            f"recut_subbundles: parent bundle {parent_id}: {nr_out} "
            f"sub-bundle(s) from {int(mask.sum())} edge(s)"
        )

    return edges_out, nr_out_map


def process_edge_data_v3(input_csv, nr_bundles=2, default_nr_bundles=2,
                         h1_flag='max', method='complete',
                         max_exact=50_000, progress_callback=None,
                         invocation='api', allow_processed_input=False):
    '''
    V3 pipeline: divide one or more already-significant, already-bundled
    edge sets (e.g. every FWER-significant bundle in a single visualization
    export) into sub-bundles for visualization only.

      1. Recover which parent (inferential) bundle each edge belongs to
         (_resolve_parent_ids) -- a visualization export commonly contains
         more than one independently FWER-significant bundle at once.
      2. Build one independent linkage tree per parent bundle
         (build_edge_linkage_per_parent). Edges from different parent
         bundles never share a tree or influence each other's distances.
      3. Cut each parent bundle's tree into its own number of sub-bundles
         (recut_subbundles).

    Unlike v1/v2 there is no isolation filter, size filter, or endpoint-
    cluster pruning here: the input edges are assumed to already be the
    final, statistically validated set from the FWER pipeline. v3 only
    reorganizes them for legibility, strictly within each parent bundle.

    Parameters
    ----------
    input_csv          : str, path to a bundle-level FWER visualization
                         export (e.g. from prepare_bundle_single_fwer.py),
                         columns i1,j1,k1,i2,j2,k2,pvalue,tstat[,bundle,network].
                         When `bundle` is absent, a sibling
                         `_v2_processed.csv` is used to recover parent-bundle
                         labels if present (see _resolve_parent_ids);
                         otherwise every edge is treated as one implicit
                         parent bundle, matching the original single-bundle
                         behavior.
    nr_bundles         : int (applied to every parent bundle) or dict
                         {parent_bundle_id: int} for independent per-parent
                         sub-bundle counts -- e.g. a small parent bundle
                         might need only 2 sub-bundles to read clearly while
                         a much larger one needs 6.
    default_nr_bundles : fallback sub-bundle count for any parent bundle not
                         named in an `nr_bundles` dict.
    h1_flag            : 'min' | 'max' | 'mean', edge-to-edge endpoint
                         distance combination rule (default 'max', matching
                         v1's default bundle-forming distance)
    method             : 'complete' | 'average', scipy linkage method
                         (default 'complete', matching v1's original
                         edge-bundling call)
    max_exact          : if any single parent bundle has more edges than
                         this, refuse rather than silently degrade to an
                         approximate tree -- at the scale this pipeline
                         targets, an exact O(N^2) distance matrix is
                         affordable per parent bundle, and an approximate
                         tree would not support instant recut.

    Returns
    -------
    result : dict with keys edges_net, linkage_matrices, nr_bundles_out,
             parent_label_source
    '''
    if is_processed_input_v3(input_csv) and not allow_processed_input:
        raise ValueError(
            "Refusing to process a '_v3_processed.csv' cache as raw input. "
            "Select the original CSV instead, or explicitly allow processed "
            "input if this is intentional."
        )

    started_at = _utc_now()
    parameters = {
        'nr_bundles': nr_bundles if isinstance(nr_bundles, dict) else int(nr_bundles),
        'default_nr_bundles': int(default_nr_bundles),
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

    parent_ids, parent_label_source = _resolve_parent_ids(input_csv, edges)
    if n_edges:
        unique_parents, parent_counts = np.unique(parent_ids, return_counts=True)
        largest_parent = int(np.max(parent_counts))
    else:
        unique_parents, largest_parent = np.array([], dtype=int), 0
    print(
        f"process_edge_data_v3: {len(unique_parents)} parent bundle(s) "
        f"(source: {parent_label_source})"
    )

    if largest_parent > max_exact:
        raise ValueError(
            f"process_edge_data_v3: the largest parent bundle has "
            f"{largest_parent} edges, exceeding max_exact={max_exact}. An "
            "exact edge-level linkage tree needs O(N^2) distances per "
            "parent bundle; pass a smaller significant-bundle export, or "
            "raise max_exact explicitly if you have verified the "
            "memory/time cost."
        )

    if edges.shape[1] <= BUNDLE_COL:
        edges = np.c_[edges, np.zeros((n_edges, BUNDLE_COL + 1 - edges.shape[1]))]
    if edges.shape[1] <= NETWORK_COL:
        edges = np.c_[edges, np.zeros((n_edges, NETWORK_COL + 1 - edges.shape[1]))]
    edges[:, NETWORK_COL] = parent_ids
    if progress_callback:
        progress_callback(20)

    linkage_matrices = build_edge_linkage_per_parent(
        edges, parent_ids, h1_flag=h1_flag, method=method
    )
    if progress_callback:
        progress_callback(70)

    edges_out, nr_out_map = recut_subbundles(
        edges, linkage_matrices, nr_bundles, default_nr_bundles=default_nr_bundles
    )
    if progress_callback:
        progress_callback(90)

    result = {
        "edges_net": edges_out,
        "linkage_matrices": linkage_matrices,
        "nr_bundles_out": nr_out_map,
        "parent_label_source": parent_label_source,
    }
    save_result_v3(input_csv, result, parameters=parameters, invocation=invocation,
                   started_at=started_at)
    if progress_callback:
        progress_callback(100)
    return result
