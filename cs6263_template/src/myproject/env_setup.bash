#!/usr/bin/env bash

set -e  # exit on error

ENV_NAME="adaptive-rag"
PYTHON_VERSION="3.10"
export UV_CACHE_DIR=$CONDA_PREFIX/.uv_cache

echo "======================================"
echo "Creating conda environment: $ENV_NAME"
echo "======================================"

# Create environment
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "======================================"
echo "Installing uv"
echo "======================================"

# Install uv (inside env)
pip install --upgrade pip
pip install uv
uv pip install cmake

echo "======================================"
echo "Installing vLLM via uv"
echo "======================================"

# Install vLLM FIRST (important for dependency resolution)
uv pip install vllm --torch-backend=cu121 --no-build-isolation

echo "======================================"
echo "Installing project requirements"
echo "======================================"

# Install your requirements
if [ -f ../../requirements.txt ]; then
    uv pip install -r ../../requirements.txt
else
    echo "WARNING: requirements.txt not found, skipping."
fi

echo "======================================"
echo "Environment setup complete"
echo "======================================"

echo ""
echo "To activate the environment later:"
echo "conda activate $ENV_NAME"