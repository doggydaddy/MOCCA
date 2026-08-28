#!/usr/bin/env bash
# ============================================================
# run_permutationTest.sh
#
# Launches permutationTest_cuda in a detached tmux session with
# timestamped log output.
#
# Usage:
#   run_permutationTest.sh \
#       -i <filelist>       \
#       -p <permutations>   \
#       -o <output_file>    \
#       [-s <session_name>] \
#       [--two-tailed]      \
#       [-b]                \
#       [--fwer]
#
#   -i  path to subject filelist (one .ccmat path per line)
#   -p  path to permutations file (one permutation per line)
#   -o  path to output p-value file (t-stat file derived automatically)
#   -s  tmux session name (default: permTest)
#   --two-tailed  enable two-tailed test
#   -b  write output in binary format instead of text
#   --fwer  enable FWER max-statistic correction (two-pass)
# ============================================================

BINARY=/mnt/islay/MOCCA/02_cudaPerm/archives/edgewise_fwer_and_supercritical_grid_2026_08_28/build/permutationTest_cuda_fwer

# ---- defaults ------------------------------------------------
SESSION="permTest"
TWO_TAILED_FLAG=""
BINARY_FLAG=""
FWER_FLAG=""

# ---- argument parsing ----------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i) FILELIST="$2";   shift 2 ;;
        -p) PERMFILE="$2";   shift 2 ;;
        -o) OUTFILE="$2";    shift 2 ;;
        -s) SESSION="$2";    shift 2 ;;
        --two-tailed) TWO_TAILED_FLAG="--two-tailed"; shift ;;
        -b) BINARY_FLAG="-b"; shift ;;
        --fwer) FWER_FLAG="--fwer"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- validate required args ----------------------------------
if [[ -z "$FILELIST" || -z "$PERMFILE" || -z "$OUTFILE" ]]; then
    echo "Usage: $0 -i <filelist> -p <permutations> -o <output_file> [-s session] [--two-tailed] [-b] [--fwer]"
    exit 1
fi

[[ -f "$FILELIST" ]] || { echo "ERROR: filelist not found: $FILELIST"; exit 1; }
[[ -f "$PERMFILE" ]] || { echo "ERROR: permutations file not found: $PERMFILE"; exit 1; }
[[ -x "$BINARY"  ]] || { echo "ERROR: binary not found/executable: $BINARY"; exit 1; }

OUTDIR=$(dirname "$OUTFILE")
mkdir -p "$OUTDIR"

# ---- log file -----------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${OUTDIR}/permutationTest_run_${TIMESTAMP}.log"

# ---- build the inner command --------------------------------
CMD="${BINARY} ${FILELIST} ${PERMFILE} ${OUTFILE}"
[[ -n "$TWO_TAILED_FLAG" ]] && CMD="${CMD} ${TWO_TAILED_FLAG}"
[[ -n "$BINARY_FLAG"     ]] && CMD="${CMD} ${BINARY_FLAG}"
[[ -n "$FWER_FLAG"       ]] && CMD="${CMD} ${FWER_FLAG}"

# ---- write a wrapper script to avoid tmux quoting issues ----
WRAPPER=$(mktemp /tmp/run_permTest_XXXXXX.sh)
cat > "${WRAPPER}" << WEOF
#!/bin/bash
echo '========================================================'
echo '  permutationTest_cuda_fwer'
echo "  Started : \$(date)"
echo '  Filelist: ${FILELIST}'
echo '  Perms   : ${PERMFILE}'
echo '  Output  : ${OUTFILE}'
echo '  Flags   : ${TWO_TAILED_FLAG} ${BINARY_FLAG} ${FWER_FLAG}'
echo '  Log     : ${LOGFILE}'
echo '========================================================'
${CMD}
echo '========================================================'
echo "  FINISHED: \$(date)"
echo '========================================================'
WEOF
chmod +x "${WRAPPER}"

# ---- launch tmux session ------------------------------------
echo "Launching tmux session '${SESSION}' ..."
echo "  Filelist: ${FILELIST}"
echo "  Perms   : ${PERMFILE}"
echo "  Output  : ${OUTFILE}"
[[ -n "$TWO_TAILED_FLAG" ]] && echo "  Test    : two-tailed"
[[ -n "$BINARY_FLAG"     ]] && echo "  Format  : binary (-b)"
[[ -n "$FWER_FLAG"       ]] && echo "  FWER    : enabled"
echo "  Log     : ${LOGFILE}"

tmux new-session -d -s "${SESSION}" \
    "bash -c '${WRAPPER} 2>&1 | tee ${LOGFILE}; rm -f ${WRAPPER}'"

echo "Session launched."
echo "  Monitor: tmux attach -t ${SESSION}"
echo "  Log    : tail -f ${LOGFILE}"
