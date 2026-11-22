#!/usr/bin/env bash
# Serves NVIDIA Nemotron 3.5 Super VL (120B total / 12B active, NVFP4) for
# `-m nvidia-nemotron-3.5-super`. See README.md for setup.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-$HOME/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}

cd "${VLLM_DIR:-$HOME/nemotron-llm}"
export PATH=$PWD/.venv/bin:/usr/local/cuda/bin:$PATH

exec .venv/bin/vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --reasoning-parser nemotron_v3
