#!/usr/bin/env bash
# Serves NVIDIA Nemotron 3.5 Bolt (74B total / 9B active, 24 GB IQ2) for
# `-m nvidia-nemotron-3.5-bolt`. See README.md for setup.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-$HOME/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.94}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
REASONING_PARSER_PLUGIN=${REASONING_PARSER_PLUGIN:-$MODEL_DIR/ultra_v3_reasoning_parser.py}

cd "${VLLM_DIR:-$HOME/nemotron-llm}"
export PATH=$PWD/.venv/bin:/usr/local/cuda/bin:$PATH
export VLLM_USE_V2_MODEL_RUNNER=1

exec .venv/bin/vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --trust-remote-code \
  --load-format auto \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8 \
  --linear-backend b12x \
  --moe-backend b12x \
  --attention-backend FLASHINFER \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype bfloat16 \
  --mamba-cache-dtype bfloat16 \
  --mamba-cache-mode align \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --reasoning-parser-plugin "$REASONING_PARSER_PLUGIN" \
  --reasoning-parser ultra_v3
