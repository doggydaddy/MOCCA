#!/usr/bin/env bash
set -euo pipefail

RESAMPLED_DIR="/mnt/highlands/data/MOCCA_UCLA/resampled_func_images_2mm"
LIST_FILE="/tmp/resampled_files.txt"
OUT_FILE="/tmp/resampled_qc.tsv"

if ! command -v fslstats >/dev/null 2>&1; then
  echo "ERROR: fslstats not found in PATH"
  exit 1
fi

if [[ ! -d "$RESAMPLED_DIR" ]]; then
  echo "ERROR: missing directory: $RESAMPLED_DIR"
  exit 1
fi

find "$RESAMPLED_DIR" -maxdepth 1 -type f -name '*.nii' | sort > "$LIST_FILE"

printf "file\tmin\tmax\tmean\tstd\n" > "$OUT_FILE"

n=0
while IFS= read -r f; do
  stats=$(fslstats "$f" -R -m -s)
  printf "%s\t%s\n" "$f" "$stats" >> "$OUT_FILE"
  n=$((n+1))
  if (( n % 25 == 0 )); then
    echo "processed=$n"
  fi
done < "$LIST_FILE"

echo "files_listed=$(wc -l < "$LIST_FILE")"
echo "rows_written=$(($(wc -l < "$OUT_FILE")-1))"

awk 'NR>1{if($2<min||NR==2)min=$2; if($3>max||NR==2)max=$3; neg+=($2<0); c++} END{print "files=" c " global_min=" min " global_max=" max " files_with_negative_min=" neg}' "$OUT_FILE"
awk 'NR>1{if($5<ms||NR==2){ms=$5;f=$1}} END{print "min_std_file=" f " min_std=" ms}' "$OUT_FILE"
awk 'NR>1{if($5>ms||NR==2){ms=$5;f=$1}} END{print "max_std_file=" f " max_std=" ms}' "$OUT_FILE"

echo "qc_file=$OUT_FILE"
