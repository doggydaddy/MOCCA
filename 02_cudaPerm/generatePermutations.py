#!/usr/bin/python

# imports
import numpy as np
import argparse
import random
import sys
import math
from time import perf_counter

parser = argparse.ArgumentParser(
                    prog='generatePermutations.py',
                    description=
                    '''generates indices for N unique permutations
                    of group A of size nA and group B of nB indices.'''
                    )
parser.add_argument('-nPerm', '--numberPermutations', 
                    type=int, 
                    help='number of permutations (excluding the original grouping row prepended as row 0)', 
                    default=5000)
parser.add_argument('-nA', '--numberGroupA', 
                    type=int, 
                    help='number of indices in group A')
parser.add_argument('-nB', '--numberGroupB', 
                    type=int, 
                    help='number of indices in group B')
parser.add_argument('-o', '--outputfile',
                    help='output filepath')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

nA = int(args.numberGroupA)
nB = int(args.numberGroupB)
nrp = int(args.numberPermutations)
outputfile = args.outputfile

print("creating", str(nrp), "permutations")
print(str(nA), "in one group")
print(str(nB), "in the other group")

def genMfromN(nA, nB, onehot=False):
    '''
    generate M (50) random numbers between [0,N-1] without replacement.
    
    can output indices of one group (let's say the first group), sorted, 
    or the one-hot indices (not sorted, for obvious reasons).
    '''

    N = nA + nB
    M = nA
    # random.sample is without replacement and correctly samples [0, N-1].
    rnd_indices = sorted(random.sample(range(N), M))

    if onehot:
        onehot_output = np.zeros(N, dtype=np.uint16)
        for k in range(M):
            onehot_output[int(rnd_indices[k])] = 1
        return onehot_output
    else:
        return rnd_indices

def genPermutations(nA, nB, nperm):
    '''
    generate nperm permutations with 
    M=nA indices amongst N=nA+nB total indices.
    
    outputs the indices only, 
    and not one-hot (to save space)

    note: There are a lot better ways to do this using standard python
    libraries, but this implementation is sufficiently fast and robust that can
    be directly translated to C/C++ libraries.
    '''

    M = nA
    N = nA + nB
    max_unique = combination(N, M)
    # The original grouping (0..nA-1) occupies one slot, so at most
    # max_unique-1 distinct permutations remain for the random draws.
    if nperm > max_unique - 1:
        raise ValueError(
            f"Requested {nperm} unique permutations, but only {max_unique - 1} exist "
            f"for nA={nA}, nB={nB} (one slot is reserved for the original grouping row)."
        )

    # Row 0 is always the original grouping: indices 0, 1, ..., nA-1.
    # It must be excluded from the random permutations.
    original_grouping = tuple(range(nA))

    output = np.empty((nperm, M), dtype=np.uint16)
    print(output.shape)

    seen = set()
    seen.add(original_grouping)  # reserve row 0 so it is never drawn again
    p = 0
    attempts = 0
    progress_step = max(1, nperm // 20)

    while p < nperm:
        a_perm = tuple(genMfromN(nA, nB))
        attempts += 1

        if a_perm in seen:
            continue

        seen.add(a_perm)
        output[p, :] = a_perm
        p += 1

        if p % progress_step == 0 or p == nperm:
            print(f"  generated {p}/{nperm} unique permutations (attempts={attempts})")

    return original_grouping, output

def combination(n, k):
    '''
    returns n choose k (combinatorics)
    
    computes the combination by formula, 
    which is probably will not translate well to C/C++
    '''
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n-k)))

def choose(n, k):
    '''
    returns n choose k (combinatorics)

    computes the combination by recursion.
    
    this is probably the fastest and most efficient method 
    to do this that can be directly translated to C/C++
    '''
    if k == 0:
        return 1
    else:
        return int((n*choose(n-1, k-1))/k)

print("generating", str(nrp), "permutations")
start = perf_counter()
original_grouping, generated_permutations = genPermutations(nA, nB, nrp)
end = perf_counter()
print("took", str(end-start), "seconds")

print("saving ...")
# Row 0: original grouping (observed statistic for the CUDA kernel).
# Rows 1..nrp: the nrp random permutations.
original_row = np.array(original_grouping, dtype=np.uint16).reshape(1, -1)
all_rows = np.vstack([original_row, generated_permutations])
np.savetxt(outputfile, all_rows, fmt='% 4d')
print(f"saved {nrp + 1} rows ({nrp} permutations + 1 original grouping row) to {outputfile}")
print("all done!")