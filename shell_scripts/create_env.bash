#!/usr/bin/env bash

set -e  # exit on error

ENV_NAME="adaptive-rag"
PYTHON_VERSION="3.10"
export PYTHONNOUSERSITE=1

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
pip install uv

echo "======================================"
echo "Installing project requirements"
echo "======================================"

uv pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install your requirements
if [ -f requirements.txt ]; then
    uv pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found, skipping."
fi

uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

echo "======================================"
echo "Environment setup complete"
echo "======================================"

echo ""
echo "To activate the environment later:"
echo "conda activate $ENV_NAME"