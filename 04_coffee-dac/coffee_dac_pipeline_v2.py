# coffee_dac_pipeline_v2.py
#
# Pipeline combining the best of v1 and spatial filtering:
#
#   0. TSTAT PRE-FILTER (optional) — keep only the top-N or tstat >= T
#      connections before any spatial processing.
#
#   1. ISOLATION FILTER — remove connections whose neither endpoint
#      neighbours any other connection's endpoint (26-connected, Chebyshev).
#
#   2. BUNDLING — two modes selectable via strict_bundles=:
#
#      a) CC BUNDLING (strict_bundles=False, default) — Union-Find on the
#         endpoint voxel graph.  Two connections land in the same bundle if
#         *any* endpoint of one is voxel-adjacent to *any* endpoint of the
#         other.  Transitive via Union-Find, so on very dense datasets this
#         can still produce large components.
#
#      b) STRICT BUNDLING (strict_bundles=True) — enforces the definition:
#           "all connections in a bundle share a common endpoint voxel, and
#            their other endpoints are mutually neighbouring."
#         Concretely, two connections A and B are merged only when they agree
#         on ONE shared endpoint (same voxel, after checking both orientations
#         A.ep1==B.ep1 / A.ep2==B.ep2 and A.ep1==B.ep2 / A.ep2==B.ep1) AND
#         their remaining endpoints are within Chebyshev distance neighbor_dist.
#         There is NO chaining: A and C can only share a bundle if they
#         themselves satisfy the criterion, not just because both are near some
#         intermediate connection B.  This produces many small, star-shaped
#         bundles and avoids giant components on dense datasets.
#
#   3. PRUNING (optional) — iteratively remove connections with isolated
#      endpoints within their bundle, then drop connections touching small
#      endpoint-voxel clusters.
#
#   4. NETWORK CLUSTERING — hierarchical clustering (hc2, average linkage)
#      on bundle-to-bundle distances, exactly as in v1.  The number of
#      networks is an explicit parameter and can be changed after the fact
#      via recut_networks() without re-running the full pipeline.

import numpy as np
import os
import pandas as pd
from collections import defaultdict, Counter

from scipy.cluster.hierarchy import fcluster

# ---------------------------------------------------------------------------
# Re-use distance/clustering helpers and column constants from v1.
# ---------------------------------------------------------------------------
from coffee_dac_pipeline import (
    BUNDLE_COL,
    NETWORK_COL,
    _CACHE_COLUMNS,
    h2_dist,
    hc2,
)

# ---------------------------------------------------------------------------
# Cache paths specific to the v2 pipeline (different suffix so the two
# variants' caches never collide).
# ---------------------------------------------------------------------------

def get_cache_paths_v2(input_csv):
    '''
    Return the paths of the two v2 cache files that accompany *input_csv*:
      - <stem>_v2_processed.csv   – edge array with bundle/network columns
      - <stem>_v2_linkage.npy     – scipy linkage matrix (Z) as binary numpy
    '''
    base = os.path.splitext(input_csv)[0]
    return base + '_v2_processed.csv', base + '_v2_linkage.npy'


def cache_exists_v2(input_csv):
    '''Return True if both v2 cache files exist for *input_csv*.'''
    csv_path, npy_path = get_cache_paths_v2(input_csv)
    return os.path.isfile(csv_path) and os.path.isfile(npy_path)


def save_result_v2(input_csv, result):
    '''
    Persist a v2 pipeline result dict to disk.

    Files written
    -------------
    <stem>_v2_processed.csv  – edges_net array as CSV with named columns
    <stem>_v2_linkage.npy    – linkage matrix as a binary .npy file, or a
                               zero-length placeholder when linkage_matrix
                               is None (v2 pipeline does not produce one).
    '''
    csv_path, npy_path = get_cache_paths_v2(input_csv)

    edges_net = result['edges_net']
    n_cols = edges_net.shape[1]
    cols = _CACHE_COLUMNS[:n_cols]
    if n_cols > len(_CACHE_COLUMNS):
        cols += [f'col{i}' for i in range(len(_CACHE_COLUMNS), n_cols)]

    pd.DataFrame(edges_net, columns=cols).to_csv(csv_path, index=False)

    lm = result.get('linkage_matrix')
    np.save(npy_path, lm if lm is not None else np.array([]))


def load_cached_result_v2(input_csv):
    '''
    Load a previously saved v2 pipeline result.

    Returns the same dict shape as process_edge_data_v2:
      { "edges_net": ndarray, "linkage_matrix": ndarray }
    '''
    csv_path, npy_path = get_cache_paths_v2(input_csv)
    edges_net = pd.read_csv(csv_path).to_numpy()
    linkage_matrix = np.load(npy_path)
    return {"edges_net": edges_net, "linkage_matrix": linkage_matrix}


# ---------------------------------------------------------------------------
# Step 0 – tstat pre-filter (optional)
# ---------------------------------------------------------------------------

# Column index of the tstat in the raw input CSV (0-indexed).
# Input format: i1,j1,k1,i2,j2,k2,pvalue,tstat → tstat is column 7.
TSTAT_COL = 7

def filter_top_tstat(edges, top_n=None, tstat_threshold=None):
    '''
    Keep only the connections with the highest t-statistics.

    Exactly one of ``top_n`` or ``tstat_threshold`` must be provided.

    Parameters
    ----------
    edges            : ndarray, shape (N, >=8)
                       Column TSTAT_COL (7) must contain the t-statistic.
    top_n            : int or None
                       Keep the ``top_n`` connections with the largest tstat.
                       Ties at the boundary are all included, so the actual
                       count may be marginally larger than top_n.
    tstat_threshold  : float or None
                       Keep all connections with tstat >= this value.

    Returns
    -------
    filtered  : ndarray, shape (M, >=8)
    kept_mask : ndarray of bool, shape (N,)
    '''
    if (top_n is None) == (tstat_threshold is None):
        raise ValueError("Provide exactly one of top_n or tstat_threshold.")

    tstats = edges[:, TSTAT_COL]

    if top_n is not None:
        if top_n >= edges.shape[0]:
            kept_mask = np.ones(edges.shape[0], dtype=bool)
        else:
            threshold = np.partition(tstats, -top_n)[-top_n]
            kept_mask = tstats >= threshold
    else:
        kept_mask = tstats >= tstat_threshold

    filtered = edges[kept_mask]
    print(f"filter_top_tstat: kept {filtered.shape[0]:,} / {edges.shape[0]:,} connections "
          f"(tstat >= {tstats[kept_mask].min():.6f})")
    return filtered, kept_mask


# ---------------------------------------------------------------------------
# Step 1 – Isolation filter
# ---------------------------------------------------------------------------

def filter_isolated_edges(edges, neighbor_dist=1.0):
    '''
    Remove connections that have no neighbours.

    A connection is *isolated* when neither of its two endpoints falls within
    Chebyshev distance ``neighbor_dist`` of *any* other connection's endpoint
    (i.e. the edge has no neighbour in the 26-connected voxel sense when
    neighbor_dist=1).  Such connections cannot belong to any bundle by the
    bundling definition and are therefore discarded before clustering.

    Parameters
    ----------
    edges         : ndarray, shape (N, >=6)
                    Columns 0:3 = ep1 (i,j,k), columns 3:6 = ep2 (i,j,k).
    neighbor_dist : float
                    Chebyshev distance threshold (default 1 → 26-connected
                    voxel neighbours).

    Returns
    -------
    filtered : ndarray, shape (M, >=6)   M <= N
        The subset of edges that have at least one endpoint neighbouring at
        least one endpoint of some *other* edge.
    kept_mask : ndarray of bool, shape (N,)
        Boolean mask over the original rows; True means the edge was kept.
    '''

    N = edges.shape[0]
    ep1 = np.round(edges[:, 0:3]).astype(np.int32)
    ep2 = np.round(edges[:, 3:6]).astype(np.int32)

    # Build voxel → set of edge indices lookup
    voxel_map = defaultdict(set)
    for idx in range(N):
        voxel_map[tuple(ep1[idx])].add(idx)
        voxel_map[tuple(ep2[idx])].add(idx)

    d = int(np.ceil(neighbor_dist))
    offsets = [
        (di, dj, dk)
        for di in range(-d, d + 1)
        for dj in range(-d, d + 1)
        for dk in range(-d, d + 1)
    ]

    def _has_neighbour(coord, self_idx):
        '''True if any voxel in the neighbourhood contains an edge other than self_idx.'''
        ci, cj, ck = coord
        for di, dj, dk in offsets:
            for other_idx in voxel_map.get((ci + di, cj + dj, ck + dk), ()):
                if other_idx != self_idx:
                    return True
        return False

    kept_mask = np.zeros(N, dtype=bool)
    for idx in range(N):
        if _has_neighbour(tuple(ep1[idx]), idx) or _has_neighbour(tuple(ep2[idx]), idx):
            kept_mask[idx] = True

    filtered = edges[kept_mask]
    n_removed = N - int(kept_mask.sum())
    print(f"filter_isolated_edges: removed {n_removed} isolated connection(s) "
          f"({N} → {filtered.shape[0]})")
    return filtered, kept_mask


# ---------------------------------------------------------------------------
# Step 2a – Connected-component bundling  (replaces greedy flood-fill)
# ---------------------------------------------------------------------------

def assign_bundle_labels_cc(edges, neighbor_dist=1.0):
    '''
    Assign connections to bundles using connected components on the endpoint
    *voxel* graph (Union-Find on voxels, not on edges).

    Algorithm
    ---------
    1. Collect every unique endpoint voxel across all edges.
    2. Build connected components of those voxels under Chebyshev adjacency
       (distance <= neighbor_dist).  Two voxels that are spatially adjacent
       land in the same voxel-component.
    3. Each edge is assigned to the bundle whose voxel-component contains
       *both* of the edge's endpoints.  If the two endpoints belong to
       *different* voxel-components, the edge is placed in its own singleton
       bundle (this should be rare after the isolation filter).

    Because components are formed over voxels rather than over edges, the
    chaining is limited to the spatial extent of contiguous voxel clusters.
    On dense but spatially separated datasets this produces multiple
    spatially-tight bundles rather than one giant component.

    Parameters
    ----------
    edges         : ndarray, shape (N, >=6)
    neighbor_dist : float  (default 1 → 26-connected voxels)

    Returns
    -------
    edges_out  : ndarray, shape (N, original_cols + 2)
        Original array with zero-indexed bundle labels appended as col BUNDLE_COL
        and duplicated into col NETWORK_COL (so downstream pruning functions that
        read NETWORK_COL work without modification).
    nr_bundles : int
    '''
    N = edges.shape[0]

    ep1 = np.round(edges[:, 0:3]).astype(np.int32)
    ep2 = np.round(edges[:, 3:6]).astype(np.int32)

    # Collect all unique endpoint voxels
    all_voxels = list({tuple(v) for v in np.vstack([ep1, ep2])})
    voxel_to_idx = {v: i for i, v in enumerate(all_voxels)}
    V = len(all_voxels)

    # Union-Find over *voxels*
    parent = np.arange(V, dtype=np.int32)
    rank   = np.zeros(V,  dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    voxel_set = set(all_voxels)
    d = int(np.ceil(neighbor_dist))
    offsets = [
        (di, dj, dk)
        for di in range(-d, d + 1)
        for dj in range(-d, d + 1)
        for dk in range(-d, d + 1)
        if not (di == 0 and dj == 0 and dk == 0)
    ]

    # Union neighbouring voxels
    for v in all_voxels:
        vi = voxel_to_idx[v]
        ci, cj, ck = v
        for di, dj, dk in offsets:
            nbr = (ci + di, cj + dj, ck + dk)
            if nbr in voxel_set:
                union(vi, voxel_to_idx[nbr])

    # Map voxel root → compact bundle label
    root_to_label: dict = {}
    def voxel_label(v):
        root = find(voxel_to_idx[v])
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        return root_to_label[root]

    # Assign each edge to the bundle of its ep1 voxel-component.
    # If ep1 and ep2 are in different voxel-components (isolated edge that
    # slipped through the isolation filter), give the edge its own bundle.
    labels = np.empty(N, dtype=np.int32)
    for idx in range(N):
        v1 = tuple(ep1[idx])
        v2 = tuple(ep2[idx])
        lbl1 = voxel_label(v1)
        lbl2 = voxel_label(v2)
        if lbl1 == lbl2:
            labels[idx] = lbl1
        else:
            # Rare: endpoints in different components — assign a new singleton bundle
            new_lbl = len(root_to_label)
            root_to_label[f'singleton_{idx}'] = new_lbl
            labels[idx] = new_lbl

    nr_bundles = len(set(labels))
    print(f"assign_bundle_labels_cc: {nr_bundles} bundle(s) across {N} edge(s)")

    lbl = labels.astype(np.float64)
    edges_out = np.c_[edges, lbl, lbl]   # col BUNDLE_COL, col NETWORK_COL (both = bundle label)
    return edges_out, nr_bundles


# ---------------------------------------------------------------------------
# Step 2b – Strict bundling  (shared-endpoint + neighbouring free-endpoint)
# ---------------------------------------------------------------------------

def assign_bundle_labels_strict(edges, neighbor_dist=1.0):
    '''
    Assign connections to bundles using the strictest spatial definition:

        "All connections in a bundle share a common endpoint voxel, and their
         other (free) endpoints are mutually neighbouring (Chebyshev distance
         <= neighbor_dist)."

    Two connections A and B are placed in the same bundle only when:
      - they share an endpoint voxel (same integer voxel coordinate after
        rounding), considering both possible orientations
        (A.ep1==B.ep1 / A.ep2==B.ep2  OR  A.ep1==B.ep2 / A.ep2==B.ep1), AND
      - their *remaining* (free) endpoints are within Chebyshev distance
        ``neighbor_dist`` of each other.

    Implemented as Union-Find over edges.  Critically, transitivity does NOT
    create large components here: A and C can only merge if they themselves
    satisfy both criteria simultaneously.  A shared intermediate connection B
    is NOT sufficient to bridge A and C unless A–C also share an endpoint
    and have neighbouring free ends.  On dense datasets this produces many
    small, star-shaped bundles instead of one giant component.

    Parameters
    ----------
    edges         : ndarray, shape (N, >=6)
                    Columns 0:3 = ep1 (i,j,k), columns 3:6 = ep2 (i,j,k).
    neighbor_dist : float  (default 1 → 26-connected voxels)

    Returns
    -------
    edges_out  : ndarray, shape (N, original_cols + 2)
        Original array with zero-indexed bundle labels appended as col
        BUNDLE_COL and duplicated into col NETWORK_COL.
    nr_bundles : int
    '''
    N = edges.shape[0]

    ep1 = np.round(edges[:, 0:3]).astype(np.int32)
    ep2 = np.round(edges[:, 3:6]).astype(np.int32)

    # Union-Find over edges
    parent = np.arange(N, dtype=np.int32)
    rank   = np.zeros(N, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    # Index: shared_voxel → list of (edge_idx, free_endpoint_coord)
    # For each edge we register it twice — once as (ep1 shared, free=ep2)
    # and once as (ep2 shared, free=ep1) — so we can look up all edges that
    # share a given endpoint regardless of which "role" it plays.
    shared_map = defaultdict(list)   # voxel_tuple → [(idx, free_coord), ...]

    for idx in range(N):
        v1 = tuple(ep1[idx])
        v2 = tuple(ep2[idx])
        shared_map[v1].append((idx, ep2[idx]))
        shared_map[v2].append((idx, ep1[idx]))

    d = int(np.ceil(neighbor_dist))

    # For each shared voxel, find pairs whose free endpoints are neighbours
    for shared_voxel, entries in shared_map.items():
        # entries is a list of (idx, free_coord) sharing this voxel.
        # Check every pair; merge if their free endpoints are within
        # Chebyshev distance neighbor_dist.
        n = len(entries)
        if n < 2:
            continue
        for i in range(n):
            idx_a, free_a = entries[i]
            for j in range(i + 1, n):
                idx_b, free_b = entries[j]
                # Chebyshev distance between the two free endpoints
                if (abs(int(free_a[0]) - int(free_b[0])) <= d and
                        abs(int(free_a[1]) - int(free_b[1])) <= d and
                        abs(int(free_a[2]) - int(free_b[2])) <= d):
                    union(idx_a, idx_b)

    # Compact zero-indexed labels
    root_to_label = {}
    labels = np.empty(N, dtype=np.int32)
    for idx in range(N):
        root = find(idx)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels[idx] = root_to_label[root]

    nr_bundles = len(root_to_label)
    print(f"assign_bundle_labels_strict: {nr_bundles} bundle(s) across {N} edge(s)")

    lbl = labels.astype(np.float64)
    edges_out = np.c_[edges, lbl, lbl]   # col BUNDLE_COL, col NETWORK_COL
    return edges_out, nr_bundles


# ---------------------------------------------------------------------------
# Step 3 – Size filter
# ---------------------------------------------------------------------------

def filter_small_networks(edges_net, min_size=2):
    '''
    Remove networks whose total edge count is strictly less than ``min_size``
    and re-index the remaining network (and bundle) labels compactly from 0.

    Networks are re-numbered in descending size order so that network 0 is
    always the largest.

    Parameters
    ----------
    edges_net : ndarray, shape (N, >=10)
                Output of a bundler (assign_bundle_labels_cc or
                assign_bundle_labels_strict); NETWORK_COL and BUNDLE_COL carry
                identical labels.
    min_size  : int
                Minimum number of edges a network must have to be kept
                (default 2 — drops strict singletons).

    Returns
    -------
    edges_out   : ndarray  – rows whose network survived, with labels re-indexed.
    size_mask   : ndarray of bool, shape (N,)  – True for kept rows.
    nr_networks : int  – number of networks after filtering.
    '''
    labels = edges_net[:, NETWORK_COL].astype(int)
    unique, counts = np.unique(labels, return_counts=True)

    # Sort survivors by descending size so network 0 == largest
    survivors = [(lbl, cnt) for lbl, cnt in zip(unique, counts) if cnt >= min_size]
    survivors.sort(key=lambda x: -x[1])

    n_dropped_nets = len(unique) - len(survivors)
    n_dropped_edges = int(counts[counts < min_size].sum())

    # Build old-label → new-label mapping
    remap = {old: new for new, (old, _) in enumerate(survivors)}

    size_mask = np.array([lbl in remap for lbl in labels], dtype=bool)
    edges_kept = edges_net[size_mask].copy()

    # Re-index both BUNDLE_COL and NETWORK_COL
    old_labels = edges_kept[:, NETWORK_COL].astype(int)
    new_labels  = np.array([remap[l] for l in old_labels], dtype=np.float64)
    edges_kept[:, BUNDLE_COL]  = new_labels
    edges_kept[:, NETWORK_COL] = new_labels

    nr_networks = len(survivors)
    print(f"filter_small_networks: dropped {n_dropped_nets} network(s) "
          f"({n_dropped_edges} edge(s)), {nr_networks} network(s) remaining")
    return edges_kept, size_mask, nr_networks


def prune_intra_network_isolated(edges_net, neighbor_dist=1.0):
    '''
    Within each network, iteratively remove connections that have at least one
    endpoint that does not neighbour any other connection's endpoint *inside
    the same network*.

    The pruning is repeated until no more connections can be removed (fixed
    point), because removing a connection may expose new isolated endpoints in
    surviving connections.  Networks that become empty after pruning are
    dropped entirely and all surviving networks are re-indexed in descending
    size order (network 0 = largest).

    Parameters
    ----------
    edges_net     : ndarray, shape (N, >=10)
                    Output of filter_small_networks(); NETWORK_COL labels are
                    compact zero-indexed integers.
    neighbor_dist : float
                    Chebyshev distance threshold (same value used everywhere
                    else in the pipeline).

    Returns
    -------
    edges_out   : ndarray  – surviving edges with re-indexed labels.
    prune_mask  : ndarray of bool, shape (N,)  – True for kept rows.
    nr_networks : int  – number of non-empty networks after pruning.
    '''

    d = int(np.ceil(neighbor_dist))
    offsets = [
        (di, dj, dk)
        for di in range(-d, d + 1)
        for dj in range(-d, d + 1)
        for dk in range(-d, d + 1)
        if not (di == 0 and dj == 0 and dk == 0)   # exclude self
    ]

    N = edges_net.shape[0]
    keep = np.ones(N, dtype=bool)   # global keep mask (indices into edges_net)

    network_labels = np.unique(edges_net[:, NETWORK_COL].astype(int))

    total_pruned = 0

    for net_lbl in network_labels:
        # Indices (into edges_net) belonging to this network
        net_idx = np.where(edges_net[:, NETWORK_COL].astype(int) == net_lbl)[0]

        # Local keep mask (indices into net_idx)
        local_keep = np.ones(len(net_idx), dtype=bool)

        while True:
            active = net_idx[local_keep]
            if len(active) == 0:
                break

            ep1 = np.round(edges_net[active, 0:3]).astype(np.int32)
            ep2 = np.round(edges_net[active, 3:6]).astype(np.int32)

            # Build voxel → set of local indices (into `active`) for this network
            voxel_map = defaultdict(set)
            for loc, (a1, a2) in enumerate(zip(ep1, ep2)):
                voxel_map[tuple(a1)].add(loc)
                voxel_map[tuple(a2)].add(loc)

            def _has_network_neighbour(coord, self_loc):
                ci, cj, ck = coord
                for di, dj, dk in offsets:
                    for other_loc in voxel_map.get((ci+di, cj+dj, ck+dk), ()):
                        if other_loc != self_loc:
                            return True
                # Also check the same voxel (different edge)
                for other_loc in voxel_map.get(coord, ()):
                    if other_loc != self_loc:
                        return True
                return False

            # Mark connections to drop: any connection where ep1 OR ep2 has
            # no neighbour among the other active connections in this network
            to_drop_local = []
            for loc in range(len(active)):
                ep1_isolated = not _has_network_neighbour(tuple(ep1[loc]), loc)
                ep2_isolated = not _has_network_neighbour(tuple(ep2[loc]), loc)
                if ep1_isolated or ep2_isolated:
                    to_drop_local.append(loc)

            if not to_drop_local:
                break   # fixed point reached

            # Map local drop indices back to net_idx positions
            active_positions = np.where(local_keep)[0]
            for loc in to_drop_local:
                local_keep[active_positions[loc]] = False

        pruned_in_net = int((~local_keep).sum())
        total_pruned += pruned_in_net
        # Write back into the global keep mask
        keep[net_idx[~local_keep]] = False

    n_kept = int(keep.sum())
    print(f"prune_intra_network_isolated: pruned {total_pruned} connection(s), "
          f"{n_kept} remaining")

    edges_pruned = edges_net[keep].copy()
    prune_mask = keep

    # Re-index networks by descending size (drop empty ones automatically)
    if edges_pruned.shape[0] == 0:
        return edges_pruned, prune_mask, 0

    labels = edges_pruned[:, NETWORK_COL].astype(int)
    unique, counts = np.unique(labels, return_counts=True)
    survivors = sorted(zip(unique, counts), key=lambda x: -x[1])
    remap = {old: new for new, (old, _) in enumerate(survivors)}
    new_labels = np.array([remap[l] for l in labels], dtype=np.float64)
    edges_pruned[:, BUNDLE_COL]  = new_labels
    edges_pruned[:, NETWORK_COL] = new_labels

    nr_networks = len(survivors)
    return edges_pruned, prune_mask, nr_networks


def prune_small_endpoint_clusters(edges_net, min_cluster_voxels=3, neighbor_dist=1.0):
    '''
    Within each network, cluster the *endpoint voxels* by spatial adjacency
    (Chebyshev distance <= neighbor_dist), then drop every connection whose
    either endpoint belongs to a cluster that contains fewer than
    ``min_cluster_voxels`` unique voxels.

    Rationale
    ---------
    After the flood-fill and iterative pruning steps the surviving connections
    are spatially coherent, but a network can still contain small "peninsulas"
    of endpoint voxels that touch the main mass only through a single chain.
    Clustering the endpoints (rather than the connections) and thresholding on
    cluster size gives a natural way to remove those peripheral fragments in a
    single, deterministic pass.

    Algorithm (per network)
    -----------------------
    1. Collect all unique endpoint voxels (both ep1 and ep2 of all edges).
    2. Build connected components of those voxels under Chebyshev adjacency.
    3. Any component with fewer than ``min_cluster_voxels`` voxels is "small".
    4. Remove every connection that has at least one endpoint in a small
       component.
    5. Networks that become empty are dropped; survivors are re-indexed by
       descending edge count.

    Parameters
    ----------
    edges_net           : ndarray, shape (N, >=10)
    min_cluster_voxels  : int   – minimum voxels a component must have to
                                  keep its connections (default 3).
    neighbor_dist       : float – Chebyshev radius (default 1 → 26-connected).

    Returns
    -------
    edges_out   : ndarray  – surviving edges with re-indexed labels.
    ep_mask     : ndarray of bool, shape (N,) – True for kept rows.
    nr_networks : int
    '''

    d = int(np.ceil(neighbor_dist))
    offsets = [
        (di, dj, dk)
        for di in range(-d, d + 1)
        for dj in range(-d, d + 1)
        for dk in range(-d, d + 1)
        if not (di == 0 and dj == 0 and dk == 0)
    ]

    N = edges_net.shape[0]
    keep = np.ones(N, dtype=bool)
    total_dropped = 0

    for net_lbl in np.unique(edges_net[:, NETWORK_COL].astype(int)):
        net_idx = np.where(edges_net[:, NETWORK_COL].astype(int) == net_lbl)[0]
        net_edges = edges_net[net_idx]

        ep1 = np.round(net_edges[:, 0:3]).astype(np.int32)
        ep2 = np.round(net_edges[:, 3:6]).astype(np.int32)

        # All unique endpoint voxels in this network
        all_voxels = list({tuple(v) for v in np.vstack([ep1, ep2])})
        voxel_set  = set(all_voxels)
        V = len(all_voxels)
        voxel_to_idx = {v: i for i, v in enumerate(all_voxels)}

        # Union-Find over voxels
        parent = list(range(V))
        rank   = [0] * V

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        for v in all_voxels:
            vi = voxel_to_idx[v]
            ci, cj, ck = v
            for di, dj, dk in offsets:
                nbr = (ci+di, cj+dj, ck+dk)
                if nbr in voxel_set:
                    union(vi, voxel_to_idx[nbr])

        # Count voxels per component
        root_counts = Counter(find(i) for i in range(V))

        # Small component roots
        small_roots = {r for r, cnt in root_counts.items() if cnt < min_cluster_voxels}

        if not small_roots:
            continue   # nothing to drop in this network

        # Mark connections whose ep1 OR ep2 is in a small component
        small_voxels = {v for v in all_voxels if find(voxel_to_idx[v]) in small_roots}

        for local_i, (a1, a2) in enumerate(zip(ep1, ep2)):
            if tuple(a1) in small_voxels or tuple(a2) in small_voxels:
                keep[net_idx[local_i]] = False
                total_dropped += 1

    n_kept = int(keep.sum())
    print(f"prune_small_endpoint_clusters: dropped {total_dropped} connection(s) "
          f"(min_cluster_voxels={min_cluster_voxels}), {n_kept} remaining")

    edges_pruned = edges_net[keep].copy()
    ep_mask = keep

    if edges_pruned.shape[0] == 0:
        return edges_pruned, ep_mask, 0

    # Re-index by descending size
    labels = edges_pruned[:, NETWORK_COL].astype(int)
    unique, counts = np.unique(labels, return_counts=True)
    survivors = sorted(zip(unique, counts), key=lambda x: -x[1])
    remap = {old: new for new, (old, _) in enumerate(survivors)}
    new_labels = np.array([remap[l] for l in labels], dtype=np.float64)
    edges_pruned[:, BUNDLE_COL]  = new_labels
    edges_pruned[:, NETWORK_COL] = new_labels

    nr_networks = len(survivors)
    return edges_pruned, ep_mask, nr_networks


def recut_networks(edges_net, linkage_matrix, nr_networks):
    '''
    Re-assign network labels from an existing linkage matrix without
    re-running the full pipeline.

    Because hc2 uses hierarchical clustering, the entire tree is encoded in
    ``linkage_matrix`` (the scipy Z matrix saved alongside the processed CSV).
    Cutting it at a different number of clusters is instantaneous — there is
    no need to reprocess the data.

    Parameters
    ----------
    edges_net      : ndarray, shape (N, 10)
                     Output of process_edge_data_v2(); col BUNDLE_COL holds
                     the bundle labels that were fed into hc2.
    linkage_matrix : ndarray, shape (B-1, 4)
                     The scipy linkage matrix Z returned by process_edge_data_v2()
                     (or loaded from the _v2_linkage.npy cache file).
    nr_networks    : int
                     Desired number of networks (FCNs) to cut the tree into.
                     Must be >= 1 and <= number of unique bundles.

    Returns
    -------
    edges_out : ndarray, shape (N, 10)
                Copy of edges_net with NETWORK_COL updated to the new labels
                (zero-indexed, network 0 = largest by edge count).
    nr_net    : int
                Actual number of networks produced (may be less than
                nr_networks if there are fewer bundles than requested).
    '''
    if linkage_matrix is None or linkage_matrix.shape[0] == 0:
        # No linkage available (e.g. only 1 bundle survived) — nothing to recut
        print("recut_networks: no linkage matrix available, returning unchanged.")
        return edges_net.copy(), int(np.max(edges_net[:, NETWORK_COL])) + 1

    nr_bundles = int(linkage_matrix.shape[0]) + 1   # B-1 merges → B leaves
    nr_net = max(1, min(nr_networks, nr_bundles))

    labels = fcluster(linkage_matrix, nr_net, criterion='maxclust') - 1  # zero-indexed

    # Map bundle label → new network label
    bundle_labels = edges_net[:, BUNDLE_COL].astype(int)
    network_labels = labels[bundle_labels].astype(np.float64)

    # Re-index so network 0 is the largest (by edge count), for consistency
    unique, counts = np.unique(network_labels.astype(int), return_counts=True)
    survivors = sorted(zip(unique, counts), key=lambda x: -x[1])
    remap = {old: new for new, (old, _) in enumerate(survivors)}
    network_labels = np.array([remap[l] for l in network_labels.astype(int)],
                               dtype=np.float64)

    edges_out = edges_net.copy()
    edges_out[:, NETWORK_COL] = network_labels
    print(f"recut_networks: {nr_net} network(s) from {nr_bundles} bundle(s)")
    return edges_out, nr_net


def process_edge_data_v2(input_csv, progress_callback=None, neighbor_dist=1.0,
                         top_n=None, tstat_threshold=None, min_network_size=2,
                         min_cluster_voxels=3, nr_networks=5, strict_bundles=False):
    '''
    V2 pipeline:
      0. (optional) tstat pre-filter
      1. Isolation filter
      2. Connected-component bundling  ← tight, density-robust
      3. Prune isolated endpoints within bundles (iterative)
      4. Prune connections touching small endpoint-voxel clusters
      5. hc2 hierarchical clustering of bundles → networks

    Parameters
    ----------
    input_csv            : str
    progress_callback    : callable(int 0-100) or None
    neighbor_dist        : float  (default 1 → 26-connected voxels)
    top_n                : int or None   – keep top-N by tstat
    tstat_threshold      : float or None – keep tstat >= T
    min_network_size     : int  – minimum edges per network (default 2)
    min_cluster_voxels   : int  – minimum endpoint-cluster size (default 3);
                                  use 1 to skip this pruning step
    nr_networks          : int  – number of FCNs for hc2 (default 5)
    strict_bundles       : bool – if True, use strict bundling (shared endpoint
                                  + neighbouring free endpoint) instead of the
                                  standard CC bundler.  Produces many smaller
                                  bundles on dense datasets (default False).

    Returns
    -------
    result : dict
        edges_net, linkage_matrix, kept_mask, tstat_mask, isolation_mask,
        size_mask, prune_mask, ep_cluster_mask, nr_networks_out
    '''

    # ------------------------------------------------------------------ load
    edges_ijk = pd.read_csv(input_csv)
    edges = edges_ijk.to_numpy()
    print(f"process_edge_data_v2: loaded {edges.shape[0]} edge(s) from '{input_csv}'")
    if progress_callback:
        progress_callback(5)

    # ------------------------------------------------- 0. tstat pre-filter
    tstat_mask = None
    if top_n is not None or tstat_threshold is not None:
        edges, tstat_mask = filter_top_tstat(edges, top_n=top_n,
                                             tstat_threshold=tstat_threshold)
        if edges.shape[0] == 0:
            raise ValueError("tstat filter removed all edges.")
    if progress_callback:
        progress_callback(10)

    # ------------------------------------------------- 1. isolation filter
    edges_filtered, isolation_mask = filter_isolated_edges(
        edges, neighbor_dist=neighbor_dist
    )
    if edges_filtered.shape[0] == 0:
        raise ValueError("All edges removed by isolation filter.")
    if progress_callback:
        progress_callback(20)

    # ----------------------------------------- 2. bundling
    if strict_bundles:
        edges_bundled, nr_bundles = assign_bundle_labels_strict(
            edges_filtered, neighbor_dist=neighbor_dist
        )
    else:
        edges_bundled, nr_bundles = assign_bundle_labels_cc(
            edges_filtered, neighbor_dist=neighbor_dist
        )
    if progress_callback:
        progress_callback(40)

    # ----------------------------------------- 3. prune isolated endpoints
    # Pruning operates on BUNDLE_COL labels (same column index as always)
    edges_bundled, prune_mask, _ = prune_intra_network_isolated(
        edges_bundled, neighbor_dist=neighbor_dist
    )
    edges_bundled, size_mask, _ = filter_small_networks(
        edges_bundled, min_size=min_network_size
    )
    if progress_callback:
        progress_callback(55)

    # ----------------------------------------- 4. endpoint-cluster pruning
    edges_bundled, ep_cluster_mask, _ = prune_small_endpoint_clusters(
        edges_bundled, min_cluster_voxels=min_cluster_voxels,
        neighbor_dist=neighbor_dist
    )
    edges_bundled, post_ep_size_mask, nr_bundles_final = filter_small_networks(
        edges_bundled, min_size=min_network_size
    )
    if progress_callback:
        progress_callback(65)

    print(f"  → {nr_bundles_final} bundle(s) remain after pruning")

    # ----------------------------------------- 5. hc2: bundles → networks
    if nr_bundles_final <= 1:
        # Only one bundle remaining — skip hc2, assign everything to network 0
        print("  → only 1 bundle after pruning, skipping hc2 (all edges → network 0)")
        edges_net = edges_bundled.copy()
        edges_net[:, NETWORK_COL] = 0
        linkage_matrix = np.empty((0, 4), dtype=np.float64)
        nr_net = 1
    else:
        bdist = h2_dist(edges_bundled)
        # Cap nr_networks to the number of surviving bundles
        nr_net = min(nr_networks, nr_bundles_final)
        if nr_net < 1:
            nr_net = 1
        result_hc2 = hc2(edges_bundled, bdist, 'average', nr_net)
        edges_net      = result_hc2['edges_net']
        linkage_matrix = result_hc2['linkage_matrix']

    # hc2 appends the network label as a new column beyond NETWORK_COL (col 9),
    # because the bundler already placed a stale copy of the bundle label in col 9.
    # Normalise: move the true network label (last col) into NETWORK_COL and
    # strip the extra column so the array is always exactly 10 columns wide.
    if edges_net.shape[1] > NETWORK_COL + 1:
        edges_net[:, NETWORK_COL] = edges_net[:, -1]
        edges_net = edges_net[:, :NETWORK_COL + 1]

    if progress_callback:
        progress_callback(90)

    # ---- propagate all masks back to original row indices ----
    # Chain: tstat → isolation → prune → size → ep_cluster → post_ep_size
    post_isolation_kept = np.zeros(isolation_mask.shape[0], dtype=bool)
    after_prune   = np.where(isolation_mask)[0][prune_mask]
    after_size    = after_prune[size_mask]
    after_ep      = after_size[ep_cluster_mask]
    after_epsize  = after_ep[post_ep_size_mask]
    post_isolation_kept[after_epsize] = True

    if tstat_mask is not None:
        kept_mask = tstat_mask.copy()
        kept_mask[tstat_mask] = post_isolation_kept
    else:
        kept_mask = post_isolation_kept

    if progress_callback:
        progress_callback(95)

    result = {
        "edges_net":        edges_net,
        "linkage_matrix":   linkage_matrix,
        "kept_mask":        kept_mask,
        "tstat_mask":       tstat_mask,
        "isolation_mask":   isolation_mask,
        "size_mask":        size_mask,
        "prune_mask":       prune_mask,
        "ep_cluster_mask":  ep_cluster_mask,
        "nr_networks_out":  nr_net,
    }

    save_result_v2(input_csv, result)
    if progress_callback:
        progress_callback(100)

    return result
