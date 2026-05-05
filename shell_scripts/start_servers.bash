#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONPATH=.

cd elasticsearch-7.10.2/

echo "Starting Elasticsearch server..."
./bin/elasticsearch &

cd ..
sleep 15

echo "Starting retriever server..."
uvicorn serve:app --port 8000 --app-dir retriever_server &