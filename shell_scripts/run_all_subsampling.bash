#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
# This assumes processing_scripts/subsample_dataset_and_remap_paras.py exists
# and follows the IRCoT/AdaptiveRAG calling convention:
#
#   python processing_scripts/subsample_dataset_and_remap_paras.py DATASET SPLIT NUM_SAMPLES
#
# Main AdaptiveRAG classifier construction uses 500 samples per dataset.

SUBSAMPLE_SCRIPT="processing_scripts.subsample_dataset_and_remap_paras"
NUM_SAMPLES="${NUM_SAMPLES:-10}"

DATASETS=(
  "hotpotqa"
  "2wikimultihopqa"
  "musique"
)

for dataset in "${DATASETS[@]}"; do

  echo
  echo "============================================================"
  echo "Subsampling dataset=${dataset}, split=dev_diff_size, n=${NUM_SAMPLES}"
  echo "============================================================"
  python -m "$SUBSAMPLE_SCRIPT" "$dataset" "dev_diff_size" "$NUM_SAMPLES"

done

DATASETS=(
  "nq"
  "trivia"
  "squad"
)

# Missing split files are skipped safely.
SPLITS=(
  "test"
  "dev_diff_size"
)

echo "Running subsampling with NUM_SAMPLES=${NUM_SAMPLES}"

for dataset in "${DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do

    echo
    echo "============================================================"
    echo "Subsampling dataset=${dataset}, split=${split}, n=${NUM_SAMPLES}"
    echo "============================================================"
    python -m "$SUBSAMPLE_SCRIPT" "$dataset" "$split" "$NUM_SAMPLES"
  done
done

echo
echo "All available subsampling jobs completed successfully."