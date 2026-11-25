# Running Nemotron-Labs-Image-30B-A3B

`-i nemotron-image` uses [NVIDIA Nemotron-Labs-Image-30B-A3B](https://huggingface.co/nvidia/Nemotron-Labs-Image-30B-A3B), an open-weight autoregressive text-to-image model. NVIDIA doesn't host it, so you run it yourself on your own GPU(s) behind a local HTTP server (`server.py`, using vLLM + the bundled L24SQ pixel decoder). `youtube-to-docs` sends the infographic prompt to that server.

## What you need

Linux (or WSL2) with NVIDIA GPU(s) supporting bfloat16 (compute capability 8.0+) and at least 80 GB of total GPU memory across 1, 2, or 4 GPUs:

| Setup | Total GPU memory | Tensor parallel (`TENSOR_PARALLEL_SIZE`) |
| :--- | :--- | :--- |
| 1x H100 / A100 80GB | 80 GB | `1` |
| 2x A100 40GB | 80 GB | `2` |
| 4x L4 24GB | 96 GB | `4` |
| 4x A100 40GB / 4x H100 80GB | 160–320 GB | `4` |

`server.py` automatically picks the largest supported tensor-parallel size (`1`, `2`, or `4`) for the visible GPUs. `youtube-to-docs` checks the GPU(s) before it starts, unless `NEMOTRON_IMAGE_URL` points at a server on another machine.

## Set up the server

These steps were tested on a Google Cloud `a2-highgpu-2g` (2x A100 40GB) with the `common-cu129-ubuntu-2404-nvidia-580` Deep Learning VM image and a 300 GB SSD boot disk. Generating one 1248×832 infographic takes about 4.5 minutes on 2x A100 40GB (faster on H100/4x GPUs).

The model weights are ~122 GB on disk (32 safetensors shards plus the L24SQ tokenizer/decoder), so use a disk with at least 250–300 GB free.

```bash
# Python headers and C++ compiler for Triton/vLLM kernels.
sudo apt-get update && sudo apt-get install -y python3.12-dev g++
curl -LsSf https://astral.sh/uv/install.sh | sh

# Download the model weights (~122 GB).
uvx --from "huggingface_hub[hf_xet]" hf download \
  nvidia/Nemotron-Labs-Image-30B-A3B \
  --local-dir ~/Nemotron-Labs-Image-30B-A3B

# Create a Python 3.12 environment with the model card's reference packages.
uv venv --python 3.12 ~/.venv-nemotron-image
uv pip install --python ~/.venv-nemotron-image/bin/python \
  "torch==2.11.0" \
  "vllm==0.21.0" \
  "transformers==5.23.0" \
  "Pillow==12.3.0" \
  "safetensors==0.8.0" \
  "einops>=0.8,<1" \
  "huggingface_hub>=1,<3"
```

Start the server:

```bash
nohup ./serve.sh > ~/nemotron-image-serve.log 2>&1 &
curl -s http://127.0.0.1:8001/v1/models   # Answers once it's ready.
```

You can override `MODEL_DIR`, `HOST`, `PORT` (default `8001`), `TENSOR_PARALLEL_SIZE`, `GPU_MEMORY_UTILIZATION` (default `0.70`), and `MAX_MODEL_LEN` (default `8192`) via environment variables.

## Generate infographics

On the same machine:

```bash
uvx --with "youtube-to-docs[all]" youtube-to-docs atmGAHYpf_c -m -i nemotron-image
```

From another machine, point `NEMOTRON_IMAGE_URL` at the server, for example through an SSH tunnel:

```bash
gcloud compute ssh my-gpu-vm -- -N -L 8001:127.0.0.1:8001 &
export NEMOTRON_IMAGE_URL=http://127.0.0.1:8001/v1
youtube-to-docs atmGAHYpf_c -m -i nemotron-image
```

The server has no authentication, so don't expose port 8001 to the internet.
