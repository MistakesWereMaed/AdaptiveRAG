#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="classifier/data/flan_t5_xl"
MODEL_NAMES=(
  t5-small
  t5-base
  t5-large
  google/flan-t5-large
)

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  OUT_DIR="classifier/checkpoints/${MODEL_NAME}"

  python -m classifier.scripts.train_t5_router \
  --train-file "${MODEL_DIR}/train.json" \
  --validation-file "${MODEL_DIR}/validation.json" \
  --model-name-or-path ${MODEL_NAME} \
  --output-dir "${OUT_DIR}" \
  --batch-size 8 \
  --eval-batch-size 16 \
  --accumulate-grad-batches 8 \
  --max-epochs 5 \
  --learning-rate 3e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.06 \
  --precision 32-true \
  --wandb-project adaptive-rag-router \
  --wandb-run-name ${MODEL_NAME}
done
