# imports

# general
import numpy as np
import os

# data loading
import pandas as pd

# clustering libraries
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, cdist

# standard library
from collections import defaultdict

# -------------------
# for gui-integration

def euc_dist(c1, c2):
    '''
    Given 2 points, find eucledian distance between them.
    '''
    
    dx = pow( abs(c1[0] - c2[0]) , 2)
    dy = pow( abs(c1[1] - c2[1]) , 2)
    dz = pow( abs(c1[2] - c2[2]) , 2)
    
    return( np.sqrt(dx+dy+dz) )  


def edge_dist(edge_a, edge_b, flag, directional=False):
    '''
    Given 2 edges, find distance between them.
    '''

    edge_a_ep1 = edge_a[0:3]
    edge_a_ep2 = edge_a[3:6]
    edge_b_ep1 = edge_b[0:3]
    edge_b_ep2 = edge_b[3:6]

    if directional:
        d1 = euc_dist(edge_a_ep1, edge_b_ep1)
        d2 = euc_dist(edge_a_ep1, edge_b_ep2)
        d3 = euc_dist(edge_a_ep2, edge_b_ep1)
        d4 = euc_dist(edge_a_ep2, edge_b_ep2)
        dists = (d1, d2, d3, d4)
    else:
        d1 = euc_dist(edge_a_ep1, edge_b_ep1)
        d4 = euc_dist(edge_a_ep2, edge_b_ep2)
        dists = (d1, d4)
         
    if flag == 'min':
        outp = np.min(dists)
    elif flag == 'max':
        outp = np.max(dists)
    elif flag == 'mean':
        outp = np.mean(dists)
    else:
        print("edge_dist error: flag must be either 'min', 'max', or 'mean'")
        return( -1 )
    
    return outp


def bundle_dist(bundle_a, bundle_b, chunk_size=2000):
    '''
    Given 2 bundles, find the minimum edge distance between them.

    Uses chunked vectorised cdist to avoid allocating a full Nba×Nbb matrix.
    chunk_size controls the number of rows of bundle_a processed at once.
    '''

    ep1_a = bundle_a[:, 0:3].astype(np.float64)
    ep2_a = bundle_a[:, 3:6].astype(np.float64)
    ep1_b = bundle_b[:, 0:3].astype(np.float64)
    ep2_b = bundle_b[:, 3:6].astype(np.float64)

    global_min = np.inf
    for start in range(0, len(ep1_a), chunk_size):
        end = start + chunk_size
        d1 = cdist(ep1_a[start:end], ep1_b, metric='euclidean')  # (chunk, Nbb)
        d4 = cdist(ep2_a[start:end], ep2_b, metric='euclidean')  # (chunk, Nbb)
        chunk_min = np.minimum(d1, d4).min()
        if chunk_min < global_min:
            global_min = chunk_min
            if global_min == 0.0:
                return 0.0  # can't get lower

    return global_min


# ---------
# analytics
# ---------

# analytics for first hclust: calculate max-min metric
def maxmin(edges, cluster_nr):
    
    edges_in_cl = edges[edges[:,6]==cluster_nr, :]
    min_dists = h1_dist(edges_in_cl, 'min')
    max_min = np.max(min_dists)
    return(max_min)

# ----------------------------
# distance matrix calculations
# ----------------------------

def h1_dist(edges, flag):
    '''
    Given a set of edges, calculate the pairwise distance vector (condensed form)
    for the first hierarchical clustering.

    Returns a 1-D condensed distance array (upper triangle, row-major order)
    compatible with scipy.cluster.hierarchy.linkage and scipy.spatial.distance.squareform.
    '''

    # Endpoints: ep1 = columns 0:3, ep2 = columns 3:6
    ep1 = edges[:, 0:3].astype(np.float64)
    ep2 = edges[:, 3:6].astype(np.float64)

    # Pairwise Euclidean distances between all ep1s and all ep2s
    d_ep1 = cdist(ep1, ep1, metric='euclidean')  # dist(ep1_a, ep1_b)
    d_ep2 = cdist(ep2, ep2, metric='euclidean')  # dist(ep2_a, ep2_b)

    if flag == 'max':
        dist_matrix = np.maximum(d_ep1, d_ep2)
    elif flag == 'min':
        dist_matrix = np.minimum(d_ep1, d_ep2)
    elif flag == 'mean':
        dist_matrix = (d_ep1 + d_ep2) / 2.0
    else:
        raise ValueError(f"h1_dist: flag must be 'min', 'max', or 'mean', got '{flag}'")

    # Return condensed upper-triangle vector for use with scipy linkage
    N = edges.shape[0]
    i_idx, j_idx = np.triu_indices(N, k=1)
    return dist_matrix[i_idx, j_idx]


def h2_dist(edges):
    '''
    Given a set of edges (with bundle labels in col BUNDLE_COL),
    calculate the distance matrix for the second hierarchical clustering.
    '''

    nr_bundles = int(np.max(edges[:, BUNDLE_COL])) + 1

    bdist = np.zeros([nr_bundles, nr_bundles])
    for i in range(nr_bundles):
        for j in range(i, nr_bundles):
            if i == j:
                bdist[i, j] = 0.
            else:
                bundle_i = edges[edges[:, BUNDLE_COL] == i, :]
                bundle_j = edges[edges[:, BUNDLE_COL] == j, :]
                bdist[i, j] = bundle_dist(bundle_i, bundle_j)
                bdist[j, i] = bdist[i, j]

    return bdist


# ---------------------------------
# Hierarchical clustering functions
# ---------------------------------

# ---------------------------------------
# edge-bundling: Hierarhical clustering 1
# ---------------------------------------
def hc1(edges, condensed_dist, flag, nr_clusters, max_exact=50_000):
    '''
    Perform hierarchical clustering on the edges based on the condensed distance vector.

    For datasets with N <= max_exact edges the full scipy linkage is used.
    For larger datasets a random subsample of max_exact edges is clustered exactly,
    then remaining edges are assigned to the nearest cluster centroid (label propagation).

    Parameters
    ----------
    edges          : ndarray, shape (N, >=7)
    condensed_dist : 1-D condensed distance array from h1_dist()
    flag           : linkage method – 'complete' or 'average'
    nr_clusters    : number of clusters (bundles)
    max_exact      : maximum N for exact linkage; larger sets use approximate mode
    '''

    N = edges.shape[0]
    ncl = nr_clusters

    if N <= max_exact:
        # ---- Exact path: full hierarchical clustering on condensed distances ----
        if condensed_dist is None:
            condensed_dist = h1_dist(edges, 'max')
        if flag not in ('complete', 'average'):
            print(f"hc1: unknown flag '{flag}', must be 'complete' or 'average'")
            return -1
        Z = linkage(condensed_dist, method=flag)
        labels = fcluster(Z, ncl, criterion='maxclust') - 1  # zero-indexed
    else:
        # ---- Approximate path for large N: subsample → linkage → propagate ----
        print(f"hc1: N={N} exceeds max_exact={max_exact}. "
              f"Using subsampled hierarchical clustering with label propagation.")
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(N, size=max_exact, replace=False)
        sub_edges = edges[sub_idx, :]

        # Build condensed distances for the subsample only
        sub_condensed = h1_dist(sub_edges, 'max')
        if flag not in ('complete', 'average'):
            flag = 'average'
        Z_sub = linkage(sub_condensed, method=flag)
        sub_labels = fcluster(Z_sub, ncl, criterion='maxclust') - 1  # zero-indexed

        # Compute cluster centroids in 6-D endpoint space from the subsample
        ep_sub = sub_edges[:, 0:6].astype(np.float64)
        centroids = np.array([
            ep_sub[sub_labels == c].mean(axis=0) if np.any(sub_labels == c)
            else ep_sub[0]
            for c in range(ncl)
        ])

        # Assign all edges to nearest centroid
        ep_all = edges[:, 0:6].astype(np.float64)
        dists_to_centroids = cdist(ep_all, centroids, metric='euclidean')
        labels = np.argmin(dists_to_centroids, axis=1)

    # Append bundle labels as a new column (col 8); preserve all original columns
    edges_out = np.c_[edges, labels.astype(np.float64)]
    return edges_out


# -----------------------------------------------------------------------
# functional connectivity network construction: hierarchical clustering 2
# -----------------------------------------------------------------------
def hc2(edges, dist, flag, nr_networks):
    '''
    Perform hierarchical clustering on the edge bundles based on the distance matrix.
    Bundle labels are in the last column of edges (appended by hc1).
    Network labels are appended as a further new column.
    '''

    if flag == 'single' or flag == 'average' or flag == 'complete':
        condensed_dist = squareform(dist)
        Z = linkage(condensed_dist, method=flag)
        labels = fcluster(Z, nr_networks, criterion='maxclust')
        labels -= 1  # zero-indexed

        # Map each edge's bundle label → network label (vectorised, no loop needed)
        network_labels = labels[edges[:, BUNDLE_COL].astype(int)]

        # Append network labels as a new column (col NETWORK_COL)
        edges_net = np.c_[edges, network_labels.astype(np.float64)]
    else:
        print("Unknown flag, must be either 'single', 'average', or 'complete'")
        return -1

    return {
        "edges_net": edges_net,
        "linkage_matrix": Z
    }
 
def estimate_nr_bundles(edges, neighbor_dist=1.0):
    '''
    Automatically estimate the number of bundles using the definition:

        "A connection belongs to a bundle if an endpoint of the connection
        is neighboring any connection already in the bundle."

    Two edges are considered neighbors if the Chebyshev (chess-king) distance
    between any pair of their endpoints is <= neighbor_dist (default 1 voxel,
    i.e. direct 26-connectivity in voxel space).

    The bundling is therefore the set of *connected components* of the graph
    where each node is an edge and two nodes are linked when they share an
    endpoint neighborhood.  The number of components is the estimated number
    of bundles.

    Parameters
    ----------
    edges         : ndarray, shape (N, >=6)
                    Columns 0:3 = ep1 (i,j,k), columns 3:6 = ep2 (i,j,k).
    neighbor_dist : float
                    Two endpoints are "neighboring" when their Chebyshev
                    distance is <= this value (default 1 → 26-connected
                    voxel neighbors).

    Returns
    -------
    nr_bundles : int
        Number of connected components = estimated number of bundles.
    labels     : ndarray of int, shape (N,)
        Zero-indexed component label for every edge.
    '''

    N = edges.shape[0]

    # --- Union-Find with path compression and union by rank ---
    parent = np.arange(N)
    rank   = np.zeros(N, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path halving
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

    # Build a voxel-coordinate → [edge indices] lookup for fast neighbor search.
    # Each edge contributes two entries (one per endpoint).
    ep1 = np.round(edges[:, 0:3]).astype(np.int32)
    ep2 = np.round(edges[:, 3:6]).astype(np.int32)

    voxel_map = defaultdict(list)
    for idx in range(N):
        voxel_map[tuple(ep1[idx])].append(idx)
        voxel_map[tuple(ep2[idx])].append(idx)

    # 26-connected neighbourhood offsets (all combinations of -1,0,+1 in 3-D,
    # excluding the zero vector only when neighbor_dist == 1 so that an edge
    # is not unioned with itself via the voxel_map lookup – the find() guard
    # handles that anyway, but skipping self saves work).
    d = int(np.ceil(neighbor_dist))
    offsets = [
        (di, dj, dk)
        for di in range(-d, d + 1)
        for dj in range(-d, d + 1)
        for dk in range(-d, d + 1)
    ]

    def _union_voxel_neighbors(coord, edge_idx):
        '''Union edge_idx with every edge that has an endpoint in a neighboring voxel.'''
        ci, cj, ck = coord
        for di, dj, dk in offsets:
            nbr = (ci + di, cj + dj, ck + dk)
            for other_idx in voxel_map.get(nbr, ()):
                union(edge_idx, other_idx)

    for idx in range(N):
        _union_voxel_neighbors(tuple(ep1[idx]), idx)
        _union_voxel_neighbors(tuple(ep2[idx]), idx)

    # Assign compact zero-indexed labels
    root_to_label = {}
    labels = np.empty(N, dtype=np.int32)
    for idx in range(N):
        root = find(idx)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels[idx] = root_to_label[root]

    nr_bundles = len(root_to_label)
    return nr_bundles, labels


# --------------------
# processing functions    
# --------------------

def process_edge_data(input_csv, nr_bundles=None, nr_networks=5, progress_callback=None,
                      max_exact=50_000, neighbor_dist=1.0):
    '''
    Process edge data from a CSV file, 

    Perform the entire pipeline: Dual hierarchical clustering, 

    Returns FCNs

    Parameters
    ----------
    nr_bundles  : int or None
        Number of edge bundles for hc1.  When *None* (default) the number of
        bundles is estimated automatically from the data using the definition:
        "a connection belongs to a bundle if an endpoint of the connection is
        neighboring any connection already in the bundle."  The estimated value
        is the number of connected components under that neighbourhood rule.
    neighbor_dist : float
        Chebyshev distance threshold (in voxels) used when estimating the
        number of bundles automatically (default 1 → 26-connected voxel
        neighbours).  Ignored when nr_bundles is supplied explicitly.
    max_exact : int
        Edges count threshold below which exact hierarchical clustering is used.
        Above this threshold a subsampled approximation is used instead to avoid
        memory exhaustion on large datasets (default 50 000).
    '''

    edges_ijk = pd.read_csv(input_csv) #converts to string
    edges = edges_ijk.to_numpy() #converts to arrays
    if progress_callback:
        progress_callback(10)

    # ------------------------------------------------------------------
    # Estimate number of bundles if not explicitly provided
    # ------------------------------------------------------------------
    if nr_bundles is None:
        print("nr_bundles not specified – estimating from data "
              f"(neighbor_dist={neighbor_dist} voxel(s))…")
        nr_bundles, _ = estimate_nr_bundles(edges, neighbor_dist=neighbor_dist)
        print(f"  → estimated nr_bundles = {nr_bundles}")

    N = edges.shape[0]
    if N <= max_exact:
        # Exact path: compute full condensed distance vector
        edist = h1_dist(edges, "max")  # condensed distance vector for edges
    else:
        # Large-dataset path: hc1 will handle subsampling internally;
        # pass None so hc1 knows to build its own distances from the subsample.
        edist = None
    edges = hc1(edges, edist, 'complete', nr_bundles, max_exact=max_exact)
    if progress_callback:
        progress_callback(50)

    bdist = h2_dist(edges) #second distance matrix for sets of edges 
    result = hc2(edges, bdist, 'average', nr_networks) #second hierarchical clustering
    if progress_callback:
        progress_callback(90)

    # Auto-save processed result so the GUI can reload instantly next time
    save_result(input_csv, result)

    if progress_callback:
        progress_callback(100)

    return result


# -------------------------
# cache save / load helpers
# -------------------------

# Column indices for the two label columns appended by the pipeline.
# Input CSVs have 8 columns (i1,j1,k1,i2,j2,k2,pvalue,tstat); the pipeline
# appends bundle (col 8) then network (col 9) without touching the originals.
BUNDLE_COL  = 8
NETWORK_COL = 9

# Column names written to / expected in the processed cache CSV
_CACHE_COLUMNS = ['i1', 'j1', 'k1', 'i2', 'j2', 'k2', 'pvalue', 'tstat',
                  'bundle', 'network']

def get_cache_paths(input_csv):
    '''
    Return the paths of the two cache files that accompany *input_csv*:
      - <stem>_processed.csv   – the edge array with bundle/network columns appended
      - <stem>_linkage.npy     – the scipy linkage matrix (Z) as a binary numpy file
    '''
    base = os.path.splitext(input_csv)[0]
    return base + '_processed.csv', base + '_linkage.npy'


def cache_exists(input_csv):
    '''Return True if both cache files exist for *input_csv*.'''
    csv_path, npy_path = get_cache_paths(input_csv)
    return os.path.isfile(csv_path) and os.path.isfile(npy_path)


def save_result(input_csv, result):
    '''
    Persist a pipeline result dict (keys: edges_net, linkage_matrix) to disk
    so that subsequent loads can skip the expensive clustering steps.

    Files written
    -------------
    <stem>_processed.csv  – edges_net array as CSV with named columns
    <stem>_linkage.npy    – linkage matrix as a binary .npy file
    '''
    csv_path, npy_path = get_cache_paths(input_csv)

    edges_net = result['edges_net']
    n_cols = edges_net.shape[1]
    # Use the standard column list; pad / truncate if the array has unexpected width
    cols = _CACHE_COLUMNS[:n_cols]
    if n_cols > len(_CACHE_COLUMNS):
        cols += [f'col{i}' for i in range(len(_CACHE_COLUMNS), n_cols)]

    pd.DataFrame(edges_net, columns=cols).to_csv(csv_path, index=False)
    np.save(npy_path, result['linkage_matrix'])


def load_cached_result(input_csv):
    '''
    Load a previously saved pipeline result from the cache files.

    Returns the same dict shape as process_edge_data:
      { "edges_net": ndarray, "linkage_matrix": ndarray }
    '''
    csv_path, npy_path = get_cache_paths(input_csv)
    edges_net = pd.read_csv(csv_path).to_numpy()
    linkage_matrix = np.load(npy_path)
    return {"edges_net": edges_net, "linkage_matrix": linkage_matrix}
