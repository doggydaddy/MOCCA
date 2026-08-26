#!/usr/bin/env bash
set -euo pipefail

project_root=/mnt/islay/MOCCA
input_file_list="$project_root/data/share_with_KI/filelist_LTLEvsRTLE_runAll_ccmat3mm.txt"
job_root=/mnt/storage/MOCCA_UCLA/permout_3mm_LTLEvsRTLE_subjectMean_100k_fwer
subject_matrix_dir="$job_root/subject_mean_ccmat"
subject_file_list="$job_root/filelist_LTLEvsRTLE_subjectMean_ccmat3mm.txt"
permutations="$job_root/permutations100k_LTLEvsRTLE_subjectMean_seed20260824.txt"
output="$job_root/LTLEvsRTLE_subjectMean_100k_twoTailed_fwer.permout"

mkdir -p "$job_root"

echo "[$(date --iso-8601=seconds)] Step 1/3: participant-level run averaging"
python3 "$project_root/02_cudaPerm/average_ccmat_runs.py" \
  --file-list "$input_file_list" \
  --group-a-runs 84 \
  --output-dir "$subject_matrix_dir" \
  --output-file-list "$subject_file_list"

echo "[$(date --iso-8601=seconds)] Step 2/3: reproducible 100k permutations"
if [[ ! -s "$permutations" ]] || [[ $(wc -l < "$permutations") -ne 100001 ]]; then
  python3 "$project_root/02_cudaPerm/generatePermutations.py" \
    --numberPermutations 100000 \
    --numberGroupA 24 \
    --numberGroupB 13 \
    --seed 20260824 \
    --outputfile "$permutations"
else
  echo "Existing permutation file has 100001 rows; reusing it"
fi

echo "[$(date --iso-8601=seconds)] Step 3/3: two-sided max-statistic FWER"
"$project_root/02_cudaPerm/build/permutationTest_cuda_fwer" \
  "$subject_file_list" \
  "$permutations" \
  "$output" \
  --two-tailed \
  --fwer \
  -b

echo "[$(date --iso-8601=seconds)] Complete: $output"
