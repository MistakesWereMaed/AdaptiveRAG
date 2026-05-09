#!/usr/bin/env bash
set -euo pipefail

export SKIP_GENERATION=1 

bash shell_scripts/run_adaptive_router_pipeline.bash \
  classifier/checkpoints/t5-small/hf_best \
  flan_t5_xl \
  validation \
  8010

bash shell_scripts/run_adaptive_router_pipeline.bash \
  classifier/checkpoints/t5-base/hf_best \
  flan_t5_xl \
  validation \
  8010

bash shell_scripts/run_adaptive_router_pipeline.bash \
  classifier/checkpoints/t5-large/hf_best \
  flan_t5_xl \
  validation \
  8010