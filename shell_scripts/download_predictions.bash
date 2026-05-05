#!/usr/bin/env bash
set -euo pipefail

wget -O ./predictions/predictions.tar.gz https://github.com/starsuzi/Adaptive-RAG/raw/refs/heads/main/predictions.tar.gz
tar -xzf ./predictions/predictions.tar.gz