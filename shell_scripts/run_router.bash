#!/usr/bin/env bash
set -euo pipefail

SKIP_GENERATION=1 bash router/run_adaptive_router_pipeline.bash \
  classifier/checkpoints/t5-large/hf_best \
  flan_t5_xl \
  validation \
  8010