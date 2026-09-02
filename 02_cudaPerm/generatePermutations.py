#!/usr/bin/python
"""Generate one master permutation file with a disjoint calibration/inference split.

Row 0 is always the observed grouping (``range(nA)``: the first ``nA`` filelist
entries are group A).  Rows 1.. are unique random label permutations,
partitioned into a calibration-only range and an inference-only range that
never overlap -- see ``manuscript/ANALYSIS_DECISIONS.md`` (2026-09-02,
"disjoint calibration and inference permutations")::

    row 0             observed assignment
    rows 1..1000      calibration set only   (1,000 null permutations)
    rows 1001..11000  inference set only    (10,000 null permutations)

Every generated row is unique and differs from row 0, so the two null subsets
are guaranteed disjoint by construction as well as by row range.  A
``.partition.json`` sidecar records the seed, the row ranges, the checksum of
the file just written, and the resulting FWER denominator.

``t = mean(group A) - mean(group B)``, so a positive sign means group A
greater and negative means group B greater -- confirm which physical group is
"A" in your filelist before interpreting bundle sign, since getting this
backwards is a correctness bug, not a cosmetic one.
"""

# imports
import numpy as np
import argparse
import json
import random
import sys
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from permutation_rows import (
    DEFAULT_CALIBRATION_PERMUTATIONS,
    DEFAULT_INFERENCE_PERMUTATIONS,
    DEFAULT_REPRESENTATION,
    REPRESENTATIONS,
    RowPartition,
    add_partition_arguments,
    partition_from_args,
    sha256_file,
    validate_permutation_file,
)


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

def genFullIndexPermutations(n, nperm, generator):
    '''
    generate nperm unique full reorderings of 0..n-1.

    Freedman--Lane permutes every participant's residual vector, so a row must
    be a complete permutation rather than the membership of group A. Row 0 is
    the identity (the observed, unpermuted assignment) and is excluded from the
    random draws, exactly as the group-membership generator excludes the
    observed grouping.
    '''
    identity = tuple(range(n))
    seen = {identity}
    output = np.empty((nperm, n), dtype=np.uint16)
    print(output.shape)

    p = 0
    attempts = 0
    progress_step = max(1, nperm // 20)
    while p < nperm:
        candidate = tuple(int(value) for value in generator.permutation(n))
        attempts += 1
        if candidate in seen:
            continue
        seen.add(candidate)
        output[p, :] = candidate
        p += 1
        if p % progress_step == 0 or p == nperm:
            print(f"  generated {p}/{nperm} unique permutations (attempts={attempts})")
    return identity, output


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='generatePermutations.py',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-nPerm', '--numberPermutations',
                        type=int,
                        default=None,
                        help='total null permutations to generate, excluding the '
                             'observed row 0. Defaults to '
                             '--calibration-permutations + --inference-permutations '
                             f'({DEFAULT_CALIBRATION_PERMUTATIONS} + '
                             f'{DEFAULT_INFERENCE_PERMUTATIONS}); if given '
                             'explicitly it must equal that sum.')
    parser.add_argument('-nA', '--numberGroupA',
                        type=int, required=True,
                        help='number of indices in group A')
    parser.add_argument('-nB', '--numberGroupB',
                        type=int, required=True,
                        help='number of indices in group B')
    parser.add_argument('-o', '--outputfile', required=True,
                        help='output filepath')
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='random seed for reproducible permutation generation')
    parser.add_argument('--representation',
                        choices=REPRESENTATIONS,
                        default=DEFAULT_REPRESENTATION,
                        help='group-a writes the sorted group A indices per row, '
                             'which is what the Welch CUDA backend consumes '
                             '(default). full-index writes a complete '
                             'participant reordering per row, which is what the '
                             'covariate-adjusted Freedman-Lane model requires; '
                             'group-membership rows cannot be reused for it.')
    add_partition_arguments(parser, stage="generate", include_file_checks=False)
    return parser.parse_args(args=argv if argv is not None else
                             (None if sys.argv[1:] else ['--help']))


def main(argv=None):
    args = parse_args(argv)

    partition = partition_from_args(args)
    nA = int(args.numberGroupA)
    nB = int(args.numberGroupB)
    outputfile = Path(args.outputfile)

    # The partition is the source of truth for how many nulls to generate.
    # -nPerm is kept for backward compatibility but may not contradict it:
    # a file whose length disagrees with its declared row ranges is exactly
    # the off-by-one class of error this design exists to rule out.
    nrp = partition.null_permutations_total
    if args.numberPermutations is not None and args.numberPermutations != nrp:
        raise ValueError(
            f"-nPerm {args.numberPermutations} contradicts the row partition, "
            f"which needs {nrp} null rows "
            f"({partition.calibration_count} calibration + "
            f"{partition.inference_count} inference). Set "
            "--calibration-permutations/--inference-permutations instead of -nPerm."
        )
    if partition.required_rows != nrp + 1:
        raise ValueError(
            f"The partition spans {partition.required_rows} rows but only "
            f"{nrp + 1} would be written; calibration and inference ranges must "
            "be contiguous with row 0 and with each other. Expected "
            f"--inference-start-row {partition.calibration_stop}."
        )

    if args.seed is not None:
        random.seed(args.seed)
        print("random seed", args.seed)
    generator = np.random.default_rng(args.seed)

    print(f"partition: {partition.describe()}")
    print(f"representation: {args.representation}")
    print("creating", str(nrp), "permutations")
    print(str(nA), "in one group")
    print(str(nB), "in the other group")

    print("generating", str(nrp), "permutations")
    start = perf_counter()
    if args.representation == "full-index":
        original_grouping, generated_permutations = genFullIndexPermutations(
            nA + nB, nrp, generator
        )
    else:
        original_grouping, generated_permutations = genPermutations(nA, nB, nrp)
    end = perf_counter()
    print("took", str(end-start), "seconds")

    print("saving ...")
    # Row 0: original grouping (observed statistic for the CUDA kernel).
    # Rows 1..nrp: the nrp random permutations.
    original_row = np.array(original_grouping, dtype=np.uint16).reshape(1, -1)
    all_rows = np.vstack([original_row, generated_permutations])
    outputfile.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(outputfile, all_rows, fmt='% 4d')
    print(f"saved {nrp + 1} rows ({nrp} permutations + 1 original grouping row) to {outputfile}")

    # Validate what was actually written, rather than trusting the generator.
    report = validate_permutation_file(
        outputfile, partition,
        representation=args.representation, n_subjects=nA + nB,
    )
    sidecar = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "02_cudaPerm/generatePermutations.py",
        "seed": args.seed,
        "n_group_a": nA,
        "n_group_b": nB,
        "command_line": sys.argv,
        **report,
    }
    sidecar_path = outputfile.with_suffix(outputfile.suffix + ".partition.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"validated: {report['unique_permutation_rows']} unique rows of "
          f"{report['permutation_file_rows']}")
    print(f"calibration rows {partition.calibration_start}.."
          f"{partition.calibration_stop - 1} are excluded from FWER; "
          f"minimum attainable p_FWER = 1/{partition.fwer_denominator}")
    print(f"partition sidecar: {sidecar_path}")
    print("all done!")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
