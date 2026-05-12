#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="classifier/data/flan_t5_xl"
MODEL_NAMES=(
  google/flan-t5-small
  google/flan-t5-base
  google/flan-t5-large
  microsoft/deberta-v3-large
)

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  OUT_DIR="classifier/checkpoints/${MODEL_NAME}"

  if [[ ${MODEL_NAME} == "microsoft/deberta-v3-large" ]]; then
    python -m classifier.scripts.train_deberta_router \
    --train-file "${MODEL_DIR}/train.json" \
    --validation-file "${MODEL_DIR}/validation.json" \
    --model-name-or-path microsoft/deberta-v3-large \
    --output-dir "${OUT_DIR}" \
    --batch-size 8 \
    --eval-batch-size 16 \
    --accumulate-grad-batches 8 \
    --max-epochs 5 \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.06 \
    --precision 16-mixed \
    --monitor val/macro_f1 \
    --wandb-project adaptive-rag-router \
    --wandb-run-name deberta-v3-large-router
  else
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
  fi
done
