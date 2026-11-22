# Running NVIDIA Nemotron LLMs (Self-Hosted vLLM)

`-m nvidia-nemotron-3.5-lightning` (30B total / 3B active) and `-m nvidia-nemotron-3.5-super` (120B total / 12B active) run NVIDIA's open hybrid Mamba-2 + Transformer MoE models on your own GPU(s) behind a vLLM server, with no API rate limits or per-token charges.

If no compatible local GPU is detected (and `NEMOTRON_LLM_URL` is not set), `youtube-to-docs` automatically falls back to NVIDIA's free rate-limited hosted API at `build.nvidia.com` (`NVIDIA_API_KEY`).

## What you need

Linux or WSL2 with NVIDIA GPU(s):

### `nvidia-nemotron-3.5-lightning` (`serve.sh`)

| Checkpoint | GPU memory | Examples (GCP / local) |
| :--- | :--- | :--- |
| NVFP4 (`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`) | 24 GB (compute capability 8.9+) | GCP `g2-standard-12` (L4 24GB, 48 GB RAM), `g4` (RTX PRO 6000), RTX 4090 / 5090, L40S |
| BF16 (`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`) | 80 GB (compute capability 8.0+) | GCP `a2-ultragpu-1g` (A100 80GB), `a3-highgpu-1g` (H100), H200, B200 |

### `nvidia-nemotron-3.5-super` (`serve_super.sh`)

| Checkpoint | GPU memory | Examples (GCP / local) |
| :--- | :--- | :--- |
| NVFP4 (`NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`) | 96 GB Blackwell (compute capability 10.0+) | GCP `g4` (1× RTX PRO 6000 96GB), `a4` (1× B200) |
| FP8 (`NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8`) | 130 GB total (compute capability 8.9+) | 1× H200 141GB, 2× H100 80GB (`TENSOR_PARALLEL_SIZE=2`) |
| BF16 (`NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`) | 260 GB total (compute capability 8.0+) | 4× H100 80GB, 2× H200, 2× B200 |

`serve.sh` runs the Lightning NVFP4 checkpoint by default (using vLLM's Marlin MoE backend so it works on Ada/Hopper GPUs like the L4 as well as Blackwell). `serve_super.sh` runs the Super VL NVFP4 checkpoint by default (set `TENSOR_PARALLEL_SIZE=2` and override `MODEL_DIR`/`SERVED_MODEL_NAME` for 2× H100 FP8).

## Set up the server

Tested on a Google Cloud `g2-standard-12` (1× NVIDIA L4 24 GB, 48 GB RAM) with the `common-cu129-ubuntu-2404-nvidia-580` Deep Learning VM image and a 150 GB disk:

```bash
# A C++ compiler and Python headers for FlashInfer / Triton kernels.
sudo apt-get update && sudo apt-get install -y g++ python3.12-dev
curl -LsSf https://astral.sh/uv/install.sh | sh

# Download the NVFP4 model weights (~20 GB).
uvx --from "huggingface_hub[hf_xet]" hf download \
  nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --local-dir ~/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

# Create a venv with vLLM (v0.27.1+ required for Nemotron 3.5).
mkdir -p ~/nemotron-llm && cd ~/nemotron-llm
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "vllm>=0.27.1" flashinfer-python
```

Start the server:

```bash
nohup ./serve.sh > ~/nemotron-llm.log 2>&1 &
curl -s localhost:8000/v1/models   # Answers once it's ready.
```

To serve the full-precision BF16 checkpoint on an 80 GB GPU instead, download `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` and override `MODEL_DIR` and `SERVED_MODEL_NAME` when running `serve.sh`.

## Run youtube-to-docs

On the same machine (auto-detects the GPU and uses the local vLLM server):

```bash
uvx --with "youtube-to-docs[all]" youtube-to-docs B0x2I_doX9o -m nvidia-nemotron-3.5-lightning
```

From another machine, point `NEMOTRON_LLM_URL` at the server, for example through an SSH tunnel:

```bash
gcloud compute ssh my-gpu-vm -- -N -L 8000:127.0.0.1:8000 &
export NEMOTRON_LLM_URL=http://127.0.0.1:8000/v1
uvx --with "youtube-to-docs[all]" youtube-to-docs B0x2I_doX9o -m nvidia-nemotron-3.5-lightning
```

If you are also running `scripts/nemotron_transcribe/serve.sh` on port `8000`, set `PORT=8001` for one of the servers and point `NEMOTRON_LLM_URL=http://127.0.0.1:8001/v1`.

The server has no authentication, so don't expose its port to the internet.
