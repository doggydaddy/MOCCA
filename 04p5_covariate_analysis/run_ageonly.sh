#!/bin/bash
# Pre-declared confirmatory run: whole-brain AGE-ONLY covariate adjustment.
# Fisher-z, HC2 Freedman-Lane, mass statistic, p_CF=5e-6 (carried over, not
# recalibrated), max-statistic FWER, same 1000/10000 disjoint partition.
set -euo pipefail
cd /mnt/islay/MOCCA
BASE=/mnt/storage/MOCCA_UCLA
ADJ=$BASE/adjusted_controlsVSpatients
ROOT=$BASE/ageonly_controlsVSpatients

echo "=== Freedman-Lane plan (age-only design) ==="; date -Is
.venv/bin/python 02_cudaPerm/freedman_lane.py \
  --design $ROOT/design/design.npz \
  --permutations $ADJ/permutations_fullindex.txt \
  --output-dir $ROOT/tables \
  --calibration-permutations 1000 --calibration-start-row 1 \
  --inference-permutations 10000 --inference-start-row 1001

echo "=== inference: mass, p_CF=5e-6, FWER ==="; date -Is
.venv/bin/python 02_cudaPerm/run_bundle_fwer.py \
  $BASE/fisherz_3mm_controlsVSpatients/participants.txt \
  $ADJ/permutations_fullindex.txt "$ROOT/inference_10k_p5e-6" \
  --cluster-forming-p 5e-6 --statistic mass --fwer \
  --freedman-lane-plan $ROOT/tables/freedman_lane_plan.flp \
  --calibration-permutations 1000 --calibration-start-row 1 \
  --inference-permutations 10000 --inference-start-row 1001 \
  --neighbor-dist 1.0 --min-size 10 --min-cluster-voxels 6 \
  --bundle-engine cpp --bundle-threads 16 \
  --capacity 20000000 --batch-size 2500

echo "=== precision check ==="; date -Is
.venv/bin/python 02_cudaPerm/bundle_fwer_precision.py \
  "$ROOT/inference_10k_p5e-6" || true
echo "=== DONE ==="; date -Is
