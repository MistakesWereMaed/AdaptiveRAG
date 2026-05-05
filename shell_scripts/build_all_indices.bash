#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
#
# Expected layout:
#   retriever_server/build_index.py
#   processed_data/{dataset}/...
#
# Expected Elasticsearch:
#   Elasticsearch should already be running before this script is launched.
#
# Usage:
#   bash build_all_indices.sh
#
# Optional overrides:
#   INDEX_SCRIPT=retriever_server/build_index.py bash build_all_indices.sh
#   DATASETS="hotpotqa 2wikimultihopqa musique wiki" bash build_all_indices.sh

INDEX_SCRIPT="${INDEX_SCRIPT:-retriever_server.build_index}"

DEFAULT_DATASETS=(
  "hotpotqa"
  "2wikimultihopqa"
  "musique"
  "wiki"
)

if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASET_LIST=(${DATASETS})
else
  DATASET_LIST=("${DEFAULT_DATASETS[@]}")
fi

echo "Building Elasticsearch indices..."
echo "Index script: $INDEX_SCRIPT"
echo "Datasets: ${DATASET_LIST[*]}"
echo

for dataset in "${DATASET_LIST[@]}"; do
  echo "============================================================"
  echo "Building index: ${dataset}"
  echo "============================================================"

  python -m "$INDEX_SCRIPT" "$dataset"

  echo
  echo "Finished index: ${dataset}"
  echo
done

echo "All requested indices built successfully."
