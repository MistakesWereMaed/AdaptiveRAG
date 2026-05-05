#!/usr/bin/env bash
set -euo pipefail

pkill -f uvicorn
pkill -f elasticsearch