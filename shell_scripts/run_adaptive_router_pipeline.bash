#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash router/run_adaptive_router_pipeline.sh \
#     classifier/checkpoints/t5_large_router_flan_t5_xl/hf_best \
#     flan_t5_xl \
#     validation \
#     8010 \
#     t5_large
#
# If arg 5 is omitted, the script derives a name from the checkpoint path.

ROUTER_MODEL="${1:-}"
GEN_MODEL_TOKEN="${2:-flan_t5_xl}"
SPLIT="${3:-validation}"
LLM_PORT_NUM="${4:-8010}"
ROUTER_RUN_NAME="${5:-}"

if [[ -z "$ROUTER_MODEL" ]]; then
  echo "Usage: bash run_adaptive_router_pipeline.sh ROUTER_MODEL GEN_MODEL_TOKEN SPLIT LLM_PORT_NUM [ROUTER_RUN_NAME]" >&2
  exit 1
fi

if [[ -z "$ROUTER_RUN_NAME" ]]; then
  # If checkpoint ends in hf_best, use parent directory name.
  if [[ "$(basename "$ROUTER_MODEL")" == "hf_best" ]]; then
    ROUTER_RUN_NAME="$(basename "$(dirname "$ROUTER_MODEL")")"
  else
    ROUTER_RUN_NAME="$(basename "$ROUTER_MODEL")"
  fi
fi

# Make folder-safe.
ROUTER_RUN_NAME="$(echo "$ROUTER_RUN_NAME" | sed 's/[^A-Za-z0-9._-]/_/g')"

GEN_MODEL_ARG="${GEN_MODEL_TOKEN//_/-}"

DATASETS=("musique" "2wikimultihopqa" "hotpotqa" "nq" "trivia" "squad")
SYSTEMS=("nor_qa" "oner_qa" "ircot_qa")

ROUTER_PRED_ROOT="router/router_predictions/${ROUTER_RUN_NAME}/${GEN_MODEL_TOKEN}"
ROUTED_PRED_ROOT="router/predictions/${ROUTER_RUN_NAME}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

echo "Router model: ${ROUTER_MODEL}"
echo "Router run name: ${ROUTER_RUN_NAME}"
echo "Generation model token: ${GEN_MODEL_TOKEN}"
echo "Generation model arg: ${GEN_MODEL_ARG}"
echo "Split: ${SPLIT}"
echo "Router prediction root: ${ROUTER_PRED_ROOT}"
echo "Routed prediction root: ${ROUTED_PRED_ROOT}"

python -m router.predict_router_labels \
  --router-model "${ROUTER_MODEL}" \
  --processed-root processed_data \
  --split "${SPLIT}" \
  --out-root "${ROUTER_PRED_ROOT}"

if [[ "${SKIP_GENERATION:-0}" != "1" ]]; then
  for dataset in "${DATASETS[@]}"; do
    for system in "${SYSTEMS[@]}"; do
      if [[ "${SPLIT}" == "validation" || "${SPLIT}" == "test" ]]; then
        bash run_retrieval_test.sh "${system}" "${GEN_MODEL_ARG}" "${dataset}" "${LLM_PORT_NUM}"
      else
        bash run_retrieval_dev.sh "${system}" "${GEN_MODEL_ARG}" "${dataset}" "${LLM_PORT_NUM}"
      fi
    done
  done
else
  echo "SKIP_GENERATION=1: not running run_retrieval_* scripts."
fi

python -m router.build_routed_predictions \
  --model "${GEN_MODEL_TOKEN}" \
  --split "${SPLIT}" \
  --router-predictions "${ROUTER_PRED_ROOT}/${SPLIT}/router_predictions.json" \
  --predictions-root predictions \
  --out-root "${ROUTED_PRED_ROOT}"

python -m router.evaluate_routed_predictions \
  --model "${GEN_MODEL_TOKEN}" \
  --split "${SPLIT}" \
  --processed-root processed_data \
  --prediction-root "${ROUTED_PRED_ROOT}" \
  --original-predictions-root predictions \
  --llm-port-num "${LLM_PORT_NUM}"