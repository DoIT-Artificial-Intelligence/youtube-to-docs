# Running Nemotron 3.5 Transcribe

`-t nvidia-nemotron-3.5-transcribe` uses [NVIDIA Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B), an open speech-to-text model. NVIDIA doesn't host it, so you run it yourself on a GPU with a vLLM server. youtube-to-docs sends the audio to that server.

It gives word timestamps and `[Speaker N]` labels, in 40 languages. A 9-minute recording takes about 6 seconds on an H100.

## What you need

One NVIDIA GPU, on Linux or WSL2:

| Checkpoint | GPU memory | Examples |
| :--- | :--- | :--- |
| BF16 | 80 GB | A100 80GB, H100, H200, B200 |
| FP8 | 48 GB | L40S, RTX PRO 6000 |
| NVFP4 | 32 GB (Blackwell) | RTX 5090 |

`serve.sh` runs the BF16 checkpoint. youtube-to-docs checks the GPU before it starts, unless `NEMOTRON_TRANSCRIBE_URL` points at a server on another machine.

## Set up the server

These steps were tested on a Google Cloud `a3-highgpu-1g` (1x H100 80GB) with the `common-cu129-ubuntu-2404-nvidia-580` Deep Learning VM image and a 300 GB disk. Setup takes about 10 minutes.

```bash
# A C++ compiler for FlashInfer's kernels, and ffmpeg for youtube-to-docs.
sudo apt-get update && sudo apt-get install -y g++ ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh

# The model weights (63 GB).
uvx --from "huggingface_hub[hf_xet]" hf download \
  nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --local-dir ~/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B

# NeMo Speech with vLLM.
git clone --depth 1 https://github.com/NVIDIA-NeMo/Speech.git ~/nemo-speech
cd ~/nemo-speech
uv venv --python 3.13 .venv
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --extra asr --extra vllm --locked --no-install-project
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python --no-deps av==19.0.2 opencv-python-headless==5.0.0.93
```

Start the server. The first start takes about 7 minutes while it compiles kernels.

```bash
nohup ./serve.sh > ~/serve.log 2>&1 &
curl -s localhost:8000/v1/models   # Answers once it's ready.
```

## Transcribe

On the same machine:

```bash
uvx --with "youtube-to-docs[all]" youtube-to-docs B0x2I_doX9o -t nvidia-nemotron-3.5-transcribe
```

From another machine, point `NEMOTRON_TRANSCRIBE_URL` at the server, for example through an SSH tunnel:

```bash
gcloud compute ssh my-gpu-vm -- -N -L 8000:127.0.0.1:8000 &
export NEMOTRON_TRANSCRIBE_URL=http://127.0.0.1:8000/v1
```

The server has no authentication, so don't expose port 8000 to the internet.

Recordings over 45 minutes are sent in 45-minute chunks. Speaker numbers restart in each chunk, so the same person can get two numbers in a long recording.
