#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="classifier/data/flan_t5_xl"
MODEL_NAMES=(
  #t5-small
  #t5-base
  #t5-large
  google/flan-t5-large
)

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export SKIP_GENERATION=1

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  bash shell_scripts/run_adaptive_router_pipeline.bash \
  classifier/checkpoints/${MODEL_NAME}/hf_best \
  flan_t5_xl \
  validation \
  8010
done 