#!/usr/bin/env bash
set -euo pipefail

# End-to-end AdaptiveRAG router evaluation.
#
# Steps:
#   1. Predict A/B/C route labels with the trained T5 router.
#   2. Optionally run repo generation scripts for all strategies.
#   3. Build routed predictions by selecting from nor_qa/oner_qa/ircot_qa outputs.
#   4. Evaluate routed predictions.
#
# Usage:
#   bash classifier/scripts/run_adaptive_router_pipeline.sh \
#     classifier/checkpoints/t5_large_router_flan_t5_xl/hf_best \
#     flan_t5_xl \
#     validation \
#     8010
#
# Skip generation if all strategy prediction files already exist:
#   SKIP_GENERATION=1 bash classifier/scripts/run_adaptive_router_pipeline.sh ...

ROUTER_MODEL="${1:-}"
GEN_MODEL_TOKEN="${2:-flan_t5_xl}"
SPLIT="${3:-validation}"
LLM_PORT_NUM="${4:-8010}"

if [[ -z "$ROUTER_MODEL" ]]; then
  echo "Usage: bash run_adaptive_router_pipeline.sh ROUTER_MODEL GEN_MODEL_TOKEN SPLIT LLM_PORT_NUM" >&2
  exit 1
fi

GEN_MODEL_ARG="${GEN_MODEL_TOKEN//_/-}"
DATASETS=("musique" "2wikimultihopqa" "hotpotqa" "nq" "trivia" "squad")
SYSTEMS=("nor_qa" "oner_qa" "ircot_qa")

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

echo "Router model: ${ROUTER_MODEL}"
echo "Generation model token: ${GEN_MODEL_TOKEN}"
echo "Generation model arg: ${GEN_MODEL_ARG}"
echo "Split: ${SPLIT}"

python -m router.predict_router_labels \
  --router-model "${ROUTER_MODEL}" \
  --processed-root processed_data \
  --split "${SPLIT}" \
  --out-root "router/router_predictions/${GEN_MODEL_TOKEN}"

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
  --router-predictions "router/router_predictions/${GEN_MODEL_TOKEN}/${SPLIT}/router_predictions.json" \
  --predictions-root predictions \
  --out-root predictions/adaptive_rag

python -m router.evaluate_routed_predictions \
  --model "${GEN_MODEL_TOKEN}" \
  --split "${SPLIT}" \
  --processed-root processed_data \
  --prediction-root predictions/adaptive_rag \
  --original-predictions-root predictions \
  --llm-port-num 8010 