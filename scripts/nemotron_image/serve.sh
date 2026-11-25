#!/usr/bin/env bash
# Serves NVIDIA Nemotron-Labs-Image-30B-A3B for `-i nemotron-image`.
# See README.md for environment setup.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_DIR="${MODEL_DIR:-$HOME/Nemotron-Labs-Image-30B-A3B}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8001}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
VENV_DIR="${VENV_DIR:-$HOME/.venv-nemotron-image}"
export PATH="$VENV_DIR/bin:/usr/local/cuda/bin:$PATH"
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/server.py"
