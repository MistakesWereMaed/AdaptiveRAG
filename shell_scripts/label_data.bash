#!/usr/bin/env bash
set -euo pipefail

python -m labeling.preprocess_silver \
  --model flan_t5_xl \
  --processed-root processed_data \
  --predictions-root predictions

python -m labeling.preprocess_binary \
  --processed-root processed_data

python -m labeling.concat_binary_silver \
  --model flan_t5_xl \