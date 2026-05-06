#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="processing_scripts"

PROCESSORS=(
  "process_hotpotqa"
  "process_2wikimultihopqa"
  "process_musique"
  "process_nq"
  "process_trivia"
  "process_squad"
)

echo "Running dataset processing scripts..."

for script in "${PROCESSORS[@]}"; do
  path="${SCRIPT_DIR}.${script}"

  echo
  echo "============================================================"
  echo "Running $path"
  echo "============================================================"
  python -m "$path"
done

echo
echo "All processing scripts completed successfully."
