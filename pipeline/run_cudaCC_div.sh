#!/bin/bash
# run_cudaCC_div.sh
#
# Runs cudaCC_div on every .nii file in a given input directory, using a
# specified brain mask, and writes .ccmat files to an output directory.
#
# Launches in a new tmux session so it can be detached and monitored.
# All output is also tee'd to a timestamped log file.
#
# Usage:
#   ./run_cudaCC_div.sh -i INPUT_DIR -m MASK -o OUTPUT_DIR [-s SESSION_NAME] [-b]
#
# Options:
#   -i  Input directory containing 4D NIfTI files
#   -m  Brain mask NIfTI file
#   -o  Output directory for .ccmat files
#   -s  tmux session name (default: cudaCC_div)
#   -b  Write binary .ccmat format (faster, ~6x smaller than text)
#
# Example:
#   ./run_cudaCC_div.sh \
#       -i /mnt/highlands/data/MOCCA_UCLA/subset_resampled_func_images_3mm \
#       -m /mnt/islay/MOCCA/templates/mask3mm_intersection.nii \
#       -o /mnt/storage/MOCCA_UCLA/ccmat_3mm_LTLEvsRTLE_run1 \
#       -b
#
# To monitor progress once launched:
#   tmux attach -t cudaCC_div
#
# To watch the log file live:
#   tail -f <logfile>

set -euo pipefail

BINARY="/mnt/islay/MOCCA/01_cudaCC/build/cudaCC_div"

# ── argument parsing ──────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,1\}//'
    exit 1
}

INPUT_DIR=""
MASK=""
OUTPUT_DIR=""
SESSION="cudaCC_div"
BINARY_FLAG=""

while getopts ":i:m:o:s:b" opt; do
    case ${opt} in
        i) INPUT_DIR="${OPTARG}" ;;
        m) MASK="${OPTARG}" ;;
        o) OUTPUT_DIR="${OPTARG}" ;;
        s) SESSION="${OPTARG}" ;;
        b) BINARY_FLAG="-b" ;;
        *) usage ;;
    esac
done

if [[ -z "${INPUT_DIR}" || -z "${MASK}" || -z "${OUTPUT_DIR}" ]]; then
    echo "ERROR: -i, -m, and -o are all required."
    usage
fi

# ── resolve to absolute paths ─────────────────────────────────────────────────
INPUT_DIR="$(realpath "${INPUT_DIR}")"
MASK="$(realpath "${MASK}")"
# OUTPUT_DIR may not exist yet — don't use realpath
OUTPUT_DIR="$(readlink -m "${OUTPUT_DIR}")"

# ── validate ──────────────────────────────────────────────────────────────────
if [[ ! -f "${BINARY}" ]]; then
    echo "ERROR: cudaCC_div binary not found: ${BINARY}" >&2; exit 1
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "ERROR: input directory not found: ${INPUT_DIR}" >&2; exit 1
fi
if [[ ! -f "${MASK}" ]]; then
    echo "ERROR: mask file not found: ${MASK}" >&2; exit 1
fi

# ── check for existing tmux session ──────────────────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "ERROR: tmux session '${SESSION}' already exists."
    echo "       Attach with: tmux attach -t ${SESSION}"
    echo "       Or kill with: tmux kill-session -t ${SESSION}"
    exit 1
fi

# ── prepare output dir and log file ──────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/cudaCC_div_run_${TIMESTAMP}.log"

# ── build the inner script that tmux will run ─────────────────────────────────
# Written to a temp file so tmux can source it cleanly
INNER_SCRIPT="$(mktemp /tmp/cudaCC_div_inner_XXXXXX.sh)"
chmod +x "${INNER_SCRIPT}"

cat > "${INNER_SCRIPT}" << INNEREOF
#!/bin/bash
set -euo pipefail

BINARY="${BINARY}"
INPUT_DIR="${INPUT_DIR}"
MASK="${MASK}"
OUTPUT_DIR="${OUTPUT_DIR}"
LOG_FILE="${LOG_FILE}"
BINARY_FLAG="${BINARY_FLAG}"

# All stdout/stderr goes to both terminal and log file
exec > >(tee -a "\${LOG_FILE}") 2>&1

echo "========================================================"
echo "  cudaCC_div batch run"
echo "  Start  : \$(date '+%Y-%m-%d %H:%M:%S')"
echo "  Input  : \${INPUT_DIR}"
echo "  Mask   : \${MASK}"
echo "  Output : \${OUTPUT_DIR}"
echo "  Format : \$([ -n "\${BINARY_FLAG}" ] && echo 'binary (-b)' || echo 'text')"
echo "  Log    : \${LOG_FILE}"
echo "========================================================"
echo ""

# Count inputs
NII_FILES=( "\${INPUT_DIR}"/*.nii )
N_TOTAL=\${#NII_FILES[@]}
echo "Found \${N_TOTAL} .nii file(s) to process."
echo ""

n_ok=0
n_skip=0
n_fail=0

for nii in "\${NII_FILES[@]}"; do
    base="\$(basename "\${nii}" .nii)"
    # output file: lowercase basename + .ccmat  (e.g. S104_1 -> s104_1.ccmat)
    ccmat="\${OUTPUT_DIR}/\${base,,}.ccmat"

    if [[ -f "\${ccmat}" ]]; then
        echo "[\$(date '+%H:%M:%S')]  [SKIP]  \${base}.nii  (output already exists)"
        (( n_skip++ )) || true
        continue
    fi

    echo "------------------------------------------------------------"
    echo "[\$(date '+%H:%M:%S')]  [START]  \${base}.nii  (\$((n_ok+n_skip+n_fail+1))/\${N_TOTAL})"
    echo "  input : \${nii}"
    echo "  mask  : \${MASK}"
    echo "  output: \${ccmat}"
    echo ""

    t_start=\$(date +%s)

    if "\${BINARY}" "\${nii}" "\${MASK}" "\${ccmat}" \${BINARY_FLAG}; then
        t_end=\$(date +%s)
        elapsed=\$(( t_end - t_start ))
        echo ""
        echo "[\$(date '+%H:%M:%S')]  [DONE]   \${base}.nii  (elapsed: \$(printf '%02d:%02d:%02d' \$((elapsed/3600)) \$(((elapsed%3600)/60)) \$((elapsed%60))))"
        (( n_ok++ )) || true
    else
        t_end=\$(date +%s)
        elapsed=\$(( t_end - t_start ))
        echo ""
        echo "[\$(date '+%H:%M:%S')]  [FAIL]   \${base}.nii  (exit code \$?) — continuing"
        (( n_fail++ )) || true
    fi
    echo ""
done

echo "========================================================"
echo "  FINISHED : \$(date '+%Y-%m-%d %H:%M:%S')"
echo "  Processed : \${n_ok}"
echo "  Skipped   : \${n_skip}  (already existed)"
echo "  Failed    : \${n_fail}"
echo "  Log file  : \${LOG_FILE}"
echo "========================================================"

# clean up inner script
rm -f "${INNER_SCRIPT}"
INNEREOF

# ── launch in tmux ────────────────────────────────────────────────────────────
echo "Launching tmux session '${SESSION}' ..."
echo "  Input  : ${INPUT_DIR}"
echo "  Mask   : ${MASK}"
echo "  Output : ${OUTPUT_DIR}"
echo "  Format : $([ -n "${BINARY_FLAG}" ] && echo 'binary (-b)' || echo 'text')"
echo "  Log    : ${LOG_FILE}"
echo ""
echo "Attach with : tmux attach -t ${SESSION}"
echo "Monitor log : tail -f ${LOG_FILE}"
echo ""

tmux new-session -d -s "${SESSION}" "bash ${INNER_SCRIPT}"

echo "Session launched. To follow progress:"
echo "  tail -f ${LOG_FILE}"
