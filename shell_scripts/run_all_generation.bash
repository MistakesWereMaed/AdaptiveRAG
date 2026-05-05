#!/usr/bin/env bash
set -euo pipefail

# Unified AdaptiveRAG/IRCoT generation runner.
#
# Starts the local LLM server for the requested model, runs all three retrieval
# strategies on dev and test for all six datasets, then stops the LLM server.
#
# Usage:
#   bash run_all_generation.bash flan-t5-xl
#
# Optional:
#   bash run_all_generation.bash flan-t5-xxl 8010
#   DATASETS="nq squad trivia" bash run_all_generation.bash flan-t5-xl
#   SYSTEMS="nor_qa oner_qa ircot_qa" bash run_all_generation.bash flan-t5-xl
#   FORCE=1 bash run_all_generation.bash flan-t5-xl
#
# Notes:
#   - Run from the repository root.
#   - Elasticsearch/retriever server must already be running separately.
#   - This script intentionally does not wrap run_retrieval_dev.sh/test.sh.
# It directly performs their runner.py calls for both splits.

MODEL="${1:-}"
LLM_PORT_NUM="${2:-8010}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: bash run_all_generation.bash MODEL [LLM_PORT_NUM]" >&2
  echo "Example: bash run_all_generation.bash flan-t5-xl 8010" >&2
  exit 1
fi

VALID_MODELS=("flan-t5-xxl" "flan-t5-xl" "gpt")
VALID_SYSTEMS=("ircot_qa" "oner_qa" "nor_qa")
VALID_DATASETS=("hotpotqa" "2wikimultihopqa" "musique" "nq" "trivia" "squad")

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

if ! contains "$MODEL" "${VALID_MODELS[@]}"; then
  echo "ERROR: Invalid MODEL='$MODEL'. Valid models: ${VALID_MODELS[*]}" >&2
  exit 1
fi

if [[ -n "${SYSTEMS:-}" ]]; then
  # shellcheck disable=SC2206
  SYSTEM_LIST=(${SYSTEMS})
else
  SYSTEM_LIST=("${VALID_SYSTEMS[@]}")
fi

if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASET_LIST=(${DATASETS})
else
  DATASET_LIST=("${VALID_DATASETS[@]}")
fi

for system in "${SYSTEM_LIST[@]}"; do
  if ! contains "$system" "${VALID_SYSTEMS[@]}"; then
    echo "ERROR: Invalid system '$system'. Valid systems: ${VALID_SYSTEMS[*]}" >&2
    exit 1
  fi
done

for dataset in "${DATASET_LIST[@]}"; do
  if ! contains "$dataset" "${VALID_DATASETS[@]}"; then
    echo "ERROR: Invalid dataset '$dataset'. Valid datasets: ${VALID_DATASETS[*]}" >&2
    exit 1
  fi
done

if [[ ! -f "runner.py" ]]; then
  echo "ERROR: runner.py not found. Run this script from the repository root." >&2
  exit 1
fi

if [[ ! -f "llm_server/serve.py" ]]; then
  echo "ERROR: llm_server/serve.py not found. Run this script from the repository root." >&2
  exit 1
fi

if [[ "$MODEL" == "gpt" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: MODEL=gpt requires OPENAI_API_KEY to be set." >&2
    exit 1
  fi
  START_LOCAL_LLM=0
else
  START_LOCAL_LLM=1
fi

LLM_PID=""

cleanup() {
  local exit_code=$?
  if [[ -n "${LLM_PID:-}" ]]; then
    echo
    echo "Stopping LLM server with PID ${LLM_PID}..."
    kill "$LLM_PID" 2>/dev/null || true
    wait "$LLM_PID" 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local port="$1"
  local max_wait_seconds="${2:-300}"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
    then
  return 0
    fi

    local now
    now="$(date +%s)"
    if (( now - start_ts >= max_wait_seconds )); then
  echo "ERROR: Timed out waiting for port ${port}." >&2
  return 1
    fi

    sleep 5
  done
}

run_dev() {
  local system="$1"
  local dataset="$2"

  echo
  echo "============================================================"
  echo "DEV | system=${system} | model=${MODEL} | dataset=${dataset}"
  echo "============================================================"

  python runner.py "$system" "$MODEL" "$dataset" write --prompt_set 1 --llm_port_num "$LLM_PORT_NUM"

  if [[ "${FORCE:-0}" == "1" ]]; then
    python runner.py "$system" "$MODEL" "$dataset" predict   --prompt_set 1   --sample_size 10   --llm_port_num "$LLM_PORT_NUM"   --force
  else
    python runner.py "$system" "$MODEL" "$dataset" predict   --prompt_set 1   --sample_size 10   --llm_port_num "$LLM_PORT_NUM"
  fi

  python runner.py "$system" "$MODEL" "$dataset" evaluate --prompt_set 1 --sample_size 10 --llm_port_num "$LLM_PORT_NUM"

  python runner.py "$system" "$MODEL" "$dataset" summarize --prompt_set 1 --sample_size 10 --llm_port_num "$LLM_PORT_NUM"
}

run_test() {
  local system="$1"
  local dataset="$2"

  echo
  echo "============================================================"
  echo "TEST | system=${system} | model=${MODEL} | dataset=${dataset}"
  echo "============================================================"

  python runner.py "$system" "$MODEL" "$dataset" write --prompt_set 1 --llm_port_num "$LLM_PORT_NUM"

  if [[ "${FORCE:-0}" == "1" ]]; then
    python runner.py "$system" "$MODEL" "$dataset" predict   --prompt_set 1   --eval_test   --official   --llm_port_num "$LLM_PORT_NUM"   --force
  else
    python runner.py "$system" "$MODEL" "$dataset" predict   --prompt_set 1   --eval_test   --official   --llm_port_num "$LLM_PORT_NUM"
  fi

  python runner.py "$system" "$MODEL" "$dataset" evaluate --prompt_set 1 --eval_test --official --llm_port_num "$LLM_PORT_NUM"
}

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH=.
export WANDB_DISABLED=true

echo "Model: ${MODEL}"
echo "LLM port: ${LLM_PORT_NUM}"
echo "Systems: ${SYSTEM_LIST[*]}"
echo "Datasets: ${DATASET_LIST[*]}"

if [[ "$START_LOCAL_LLM" == "1" ]]; then
  echo
  echo "Starting local LLM server..."
  MODEL_NAME="$MODEL" uvicorn serve:app --port "$LLM_PORT_NUM" --app-dir llm_server &

  LLM_PID="$!"
  echo "LLM server PID: ${LLM_PID}"
  echo "Waiting for LLM server on port ${LLM_PORT_NUM}..."
  wait_for_port "$LLM_PORT_NUM" 600
  echo "LLM server is reachable."
else
  echo
  echo "MODEL=gpt selected; skipping local LLM server startup."
fi

for dataset in "${DATASET_LIST[@]}"; do
  for system in "${SYSTEM_LIST[@]}"; do
    run_dev "$system" "$dataset"
    run_test "$system" "$dataset"
  done
done

echo
echo "All generation and evaluation jobs completed successfully."
