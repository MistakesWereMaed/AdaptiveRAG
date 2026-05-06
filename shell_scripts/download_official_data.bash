#!/usr/bin/env bash
set -euo pipefail

# Download processed data, needed to match with downloaded predictions for labeling.
mkdir -p processed_data
wget -O ./processed_data/processed_data.tar.gz https://github.com/starsuzi/Adaptive-RAG/raw/refs/heads/main/processed_data.tar.gz
tar -xzf ./processed_data/processed_data.tar.gz

# Download predictions.
mkdir -p predictions
wget -O ./predictions/predictions.tar.gz https://github.com/starsuzi/Adaptive-RAG/raw/refs/heads/main/predictions.tar.gz
tar -xzf ./predictions/predictions.tar.gz