import numpy as np

# ------------------------------------------------------
# low-level functions to parse upper triangular matrices
# ------------------------------------------------------

from itertools import chain
from scipy.spatial.distance import squareform
def iter_data(fileobj):
    for line in fileobj:
        yield from line.split()

def read_triangular_array(path):
    with open(path) as fileobj:
        first = fileobj.readline().split()
        n = len(first)
        count = int(n*(n+1)/2)
        data = chain(first, iter_data(fileobj))
        return np.fromiter(data, float, count=count)


# ------------------------------------------------------------------------
# support functions to switch between upper triangular and linear indexing
# ------------------------------------------------------------------------

def ij2k(i, j, n):
    k = int( (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1 )
    return k

def k2ij(k, n):
    i = int( n - 2 - int(np.sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5) )
    j = int( k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2 )
    return i, j

# ----------------------------------------------------------------------------
# functions to parse blocks in either upper triangular or linear (rectangular)
# form into the full upper triangular cc matrix
# ----------------------------------------------------------------------------

def parse_triag(filename, n, blk_x, blk_y, buffer, N, mod=0):
    bn00 = filename
    b00 = read_triangular_array(bn00)

    range_n = n
    if mod != 0:
        range_n = mod
        print("performing asymmetrical parsing of final bit of diagonal: ", range_n)

    for i in range(range_n):
        for j in range(i+1,range_n):
            k = ij2k(i, j, range_n)
            Fi, Fj = blk_x*n+i, blk_y*n+j
            Fk = ij2k(Fi, Fj, N)
            buffer[Fk] = b00[k]

def parse_odiag(filename, n, blk_x, blk_y, buffer, N, rect=(0,0)):
    bn10 = filename
    b10 = np.loadtxt(bn10)
    b10 = b10.flatten()

    range_x = n
    range_y = n
    if rect[0] != 0:
        range_x = rect[0]
        range_y = rect[1]
        print("performing asymmetrical parsing of rectangular matrix: ", range_x, range_y)

    for i in range(range_x):
        for j in range(range_y):
            Fi, Fj = blk_x*n+i, blk_y*n+j
            if Fi > Fj:
                Fi, Fj = blk_y*n+j, blk_x*n+i

            Fk = ij2k(Fi, Fj, N)
            buffer[Fk] = b10[i*range_y+j]
        

import glob
def parse_inParts(prefix, nr_cuts, std_block_size, start_indices, end_indices, buffer, N):
    '''
    parse all files with prefix as individual blocks in the full cc matrix

    args:
    * nr_cuts           = number of subdivsions
    * std_block_size    = standard block size, as opposed to the "left-over" block size
    * start_indices     = starting indices of the subdivsions
    * end_indices       = end indices of the subdivisions
    * buffer            = full (upper-triangular) array in 1D
    * N                 = size of the full cc-matrix (gV)

    '''
    files = sorted(glob.glob(prefix+'*'))
    print(files)

    nr_files = len(files)
    nr_runs = int( (nr_cuts*(nr_cuts+1))/2 )
    if ( nr_runs != nr_files ):
        print("nr. files is not equals to nr. runs! sanity check failed, exiting.")
        return

    print("number of cuts:", nr_cuts)
    print("number of runs:", nr_runs)
    c = 0
    for i in range(nr_cuts):
        for j in range(i, nr_cuts):
            print("processing block:" , str(i), str(j))

            block_size_x = end_indices[i] - start_indices[i] + 1
            block_size_y = end_indices[j] - start_indices[j] + 1
            print("size: ", str(block_size_x), str(block_size_y))

            if i == j:
                block_size = block_size_x
                if block_size != std_block_size:
                    parse_triag(files[c], std_block_size, i, j, buffer, N, mod=block_size)
                else:
                    parse_triag(files[c], std_block_size, i, j, buffer, N)
                c += 1
            else:
                if block_size_x != block_size_y:
                    parse_odiag(files[c], block_size, i,  j, buffer, N, rect=(block_size_x, block_size_y))
                else:
                    parse_odiag(files[c], block_size, i,  j, buffer, N)
                c += 1


# ---------------
# debug functions
# ---------------

def peak_block(data, block_size, blk_x, blk_y, peak_size):
    '''
    support function for debugging
    
    peaks at the first indices of a block in the full cc matrix
    '''
    idx_x = block_size*blk_x
    idx_y = block_size*blk_y
    print(data[idx_x:(idx_x+peak_size), idx_y:(idx_y+peak_size)])

def compare_the_two(d1, d2):
    '''
    compared two 2d (numpy) arrays d1 and d2 to see if they are "the same" (within error)
    
    returns True if they are, False otherwise (and prints out error messages are
    debug information)

    assumes matrices are symmetric upper triangular with "trivial" diagonal, so
    only checks upper triangular elements.
    '''
    if d1.shape != d2.shape:
        print("the shapes of the 2 matrices are not even the same, are you even trying?")
        return False
    else:
        for i in range(d1.shape[0]):
            for j in range(i+1, d1.shape[1]):
                if abs(d1[i, j] - d2[i, j]) > 1e-3:
                    print("Mismatch found at index (i,j) = ", str(i), ", ", str(j))
                    print("gt[", str(i), ",", str(j), "]=", str(d1[i,j]))
                    print("buffer_sq[", str(i), ",", str(j), "]=", str(d2[i,j]))
                    print("with a difference of: ", str(abs(d1[i,j]-d2[i,j])))
                    return False
    return True