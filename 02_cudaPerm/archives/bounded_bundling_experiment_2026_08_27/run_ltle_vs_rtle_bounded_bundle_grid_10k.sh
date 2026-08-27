#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
job_root="/mnt/storage/MOCCA_UCLA/bundle_fwer_LTLEvsRTLE_10k_pgrid_bounded_cpp"
filelist="/mnt/storage/MOCCA_UCLA/permout_3mm_LTLEvsRTLE_subjectMean_100k_fwer/filelist_LTLEvsRTLE_subjectMean_ccmat3mm.txt"
permutations="/mnt/storage/MOCCA_UCLA/permout_3mm_LTLEvsRTLE_subjectMean_100k_fwer/permutations100k_LTLEvsRTLE_subjectMean_seed20260824.txt"

mkdir -p "$job_root"
cd "$project_root"

exec /usr/bin/time -v \
  "$project_root/.venv/bin/python" \
  "$project_root/02_cudaPerm/run_bundle_fwer.py" \
  "$filelist" "$permutations" "$job_root" \
  --mask "$project_root/templates/mask3mm.dump" \
  --cluster-forming-p-grid \
    0.001 0.0005 0.0002 0.0001 0.00005 0.00002 0.00001 \
  --null-permutations 10000 \
  --statistic mass \
  --neighbor-dist 1 \
  --min-size 10 \
  --min-cluster-voxels 6 \
  --batch-size 10001 \
  --capacity 20000000 \
  --bundle-engine cpp \
  --bundle-method bounded \
  --bundle-threads 16 \
  --resume
