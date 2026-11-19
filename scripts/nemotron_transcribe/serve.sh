#!/usr/bin/env bash
# Serves NVIDIA Nemotron 3.5 Transcribe for `-t nvidia-nemotron-3.5-transcribe`.
# These are the model card's settings for one H100 80GB. See README.md for setup.
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-$HOME/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B}
cd "${NEMO_SPEECH_DIR:-$HOME/nemo-speech}"
# FlashInfer compiles kernels on first start and needs ninja (in the venv) and nvcc.
export PATH=$PWD/.venv/bin:/usr/local/cuda/bin:$PATH
export VLLM_PLUGINS=nemo_speechlm
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
export VLLM_USE_V2_MODEL_RUNNER=0
exec .venv/bin/vllm serve "$MODEL_DIR" \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --host 127.0.0.1 --port 8000 \
  --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.94 \
  --max-model-len 65536 --max-num-batched-tokens 4096 --max-num-seqs 192 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --use-replayssm --mamba-backend flashinfer --mamba-cache-mode none \
  --mm-processor-cache-gb 0 --no-async-scheduling \
  --limit-mm-per-prompt "{\"audio\":{\"count\":6,\"length\":120000000}}" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":8}" \
  --reasoning-parser nemotron_v3 \
  --trust-request-mm-kwargs \
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
