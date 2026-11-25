"""
NVIDIA Nemotron-Labs-Image-30B-A3B (`-i nemotron-image`).

NVIDIA publishes the model as open weights
(https://huggingface.co/nvidia/Nemotron-Labs-Image-30B-A3B) and doesn't host
it at build.nvidia.com, so it runs on the user's own GPU(s) behind a local
vLLM + L24SQ HTTP server (see scripts/nemotron_image/README.md). This module
checks that the machine has enough GPU memory to run it and generates
infographics through the server.
"""

import base64
import os
import sys
from typing import List, Optional, Tuple

import requests

from youtube_to_docs.nemotron_transcribe import (
    GpuInfo,
    NemotronSetupError,
    detect_gpus,
)

NEMOTRON_IMAGE_MODEL = "nemotron-image"
NEMOTRON_IMAGE_HF_ID = "nvidia/Nemotron-Labs-Image-30B-A3B"

# The image server's OpenAI-compatible API. Set NEMOTRON_IMAGE_URL to use a
# server on another machine; otherwise this machine runs it and must have GPUs.
NEMOTRON_IMAGE_URL_ENV = "NEMOTRON_IMAGE_URL"
NEMOTRON_IMAGE_DEFAULT_URL = "http://127.0.0.1:8001/v1"

# Landscape resolution from the model card's recommended set (multiple of 16,
# fits within the 4,352 visual-token budget).
NEMOTRON_IMAGE_DEFAULT_SIZE = "1248x832"

# Seconds to wait for one image. Generation of ~4,100 visual tokens takes about
# 4.5 minutes on 2x A100 40GB (faster on H100); the ceiling exists so a stalled
# server fails instead of hanging.
NEMOTRON_IMAGE_TIMEOUT_SECONDS = (30, 15 * 60)

# Minimum compute capability for bfloat16 (Ampere 8.0+) and total GPU memory
# in MiB across tensor-parallel GPUs. Weights in bfloat16 are ~65 GB plus
# ~1.3 GB for the L24SQ pixel decoder and KV cache.
# nvidia-smi reports slightly under marketing size (e.g. 81,559 MiB for an
# 80 GB H100, 40,960 MiB for an A100 40GB, 23,034 MiB for an L4 24GB).
NEMOTRON_IMAGE_MIN_COMPUTE_CAP = 8.0
NEMOTRON_IMAGE_MIN_TOTAL_MIB = 76_000


def _requirements_text() -> str:
    return (
        f"{NEMOTRON_IMAGE_MODEL} runs on your own NVIDIA GPU(s) in bfloat16. "
        f"It needs compute capability {NEMOTRON_IMAGE_MIN_COMPUTE_CAP}+ and at "
        "least 80 GB total GPU memory across 1, 2, or 4 GPUs "
        "(e.g. 1x H100/A100 80GB, 2x A100 40GB, or 4x L4 24GB)."
    )


def _tensor_parallel_count(num_gpus: int) -> int:
    """Largest supported tensor-parallel size (1, 2, or 4) that fits num_gpus."""
    for tp in (4, 2, 1):
        if num_gpus >= tp:
            return tp
    return 0


def check_nemotron_image_hardware(
    gpus: Optional[List[GpuInfo]] = None, platform: str = sys.platform
) -> Tuple[bool, str]:
    """
    Checks whether this machine can run Nemotron-Labs-Image-30B-A3B.

    Returns (ok, message). The message names the usable GPU(s) or explains
    what's missing.
    """
    if not platform.startswith("linux"):
        return False, (
            f"{NEMOTRON_IMAGE_MODEL} needs Linux (or WSL2 on Windows); "
            f"this machine runs {platform}.\n{_requirements_text()}"
        )

    if gpus is None:
        gpus = detect_gpus()
    if gpus is None:
        return False, (
            "No NVIDIA driver found (nvidia-smi isn't installed).\n"
            f"{_requirements_text()}"
        )
    if not gpus:
        return False, f"nvidia-smi found no GPUs.\n{_requirements_text()}"

    capable = [
        gpu for gpu in gpus if gpu.compute_capability >= NEMOTRON_IMAGE_MIN_COMPUTE_CAP
    ]
    if not capable:
        found = ", ".join(
            f"{gpu.name} ({gpu.memory_mib / 1024:.0f} GB, compute capability "
            f"{gpu.compute_capability})"
            for gpu in gpus
        )
        return False, (
            f"Found {found}, which does not support bfloat16.\n{_requirements_text()}"
        )

    # Sort largest-memory GPUs first and take the largest supported TP slice.
    capable.sort(key=lambda g: -g.memory_mib)
    tp = _tensor_parallel_count(len(capable))
    selected = capable[:tp]
    # Tensor parallelism shards evenly, so effective capacity is tp * min_mem.
    effective_mib = tp * min(gpu.memory_mib for gpu in selected)
    if effective_mib < NEMOTRON_IMAGE_MIN_TOTAL_MIB:
        found = ", ".join(
            f"{gpu.name} ({gpu.memory_mib / 1024:.0f} GB, compute capability "
            f"{gpu.compute_capability})"
            for gpu in gpus
        )
        return False, f"Found {found}, which is too small.\n{_requirements_text()}"

    total_gb = effective_mib / 1024
    gpu_summary = (
        f"{tp}x {selected[0].name}"
        if all(g.name == selected[0].name for g in selected)
        else ", ".join(g.name for g in selected)
    )
    return True, (
        f"{gpu_summary} ({total_gb:.0f} GB usable) can run "
        f"{NEMOTRON_IMAGE_HF_ID} in BF16 (TP={tp})."
    )


def nemotron_image_server_url() -> str:
    """The image server's base URL, without a trailing slash."""
    url = os.environ.get(NEMOTRON_IMAGE_URL_ENV) or NEMOTRON_IMAGE_DEFAULT_URL
    return url.rstrip("/")


def uses_local_image_server() -> bool:
    """True when the server runs on this machine, so its GPUs must be checked."""
    return not os.environ.get(NEMOTRON_IMAGE_URL_ENV)


def _server_hint() -> str:
    return (
        "Start the Nemotron image server (see scripts/nemotron_image/README.md), or "
        f"set {NEMOTRON_IMAGE_URL_ENV} to a server on another machine."
    )


def check_nemotron_image_server() -> Tuple[bool, str]:
    """Checks that the Nemotron image server is up and serving the model."""
    url = nemotron_image_server_url()
    try:
        response = requests.get(f"{url}/models", timeout=10)
        response.raise_for_status()
        served = [model.get("id") for model in response.json().get("data", [])]
    except Exception as e:
        return False, f"No Nemotron image server at {url} ({e}).\n{_server_hint()}"
    if NEMOTRON_IMAGE_HF_ID not in served:
        return False, (
            f"The server at {url} serves {', '.join(map(str, served)) or 'nothing'}, "
            f"not {NEMOTRON_IMAGE_HF_ID}.\n{_server_hint()}"
        )
    return True, f"Using the Nemotron image server at {url}."


def generate_nemotron_image(
    prompt: str,
    size: str = NEMOTRON_IMAGE_DEFAULT_SIZE,
) -> Tuple[Optional[bytes], int, int]:
    """
    Generates an image using Nemotron-Labs-Image-30B-A3B via the local or
    remote OpenAI-compatible image server.

    Returns (image_bytes, input_tokens, output_tokens).
    """
    url = nemotron_image_server_url()
    body = {
        "model": NEMOTRON_IMAGE_HF_ID,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    try:
        response = requests.post(
            f"{url}/images/generations",
            json=body,
            timeout=NEMOTRON_IMAGE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.HTTPError as e:
        detail = e.response.text[:500] if e.response is not None else ""
        print(f"Error: Nemotron image server failed: {e} {detail}")
        return None, 0, 0
    except Exception as e:
        print(f"Error: Nemotron image server failed: {e}")
        return None, 0, 0

    data = payload.get("data") or []
    b64_json = data[0].get("b64_json") if data else None
    if not b64_json:
        print(f"No image data found in response from {NEMOTRON_IMAGE_MODEL}")
        return None, 0, 0

    usage = payload.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return base64.b64decode(b64_json), input_tokens, output_tokens


__all__ = [
    "NEMOTRON_IMAGE_DEFAULT_SIZE",
    "NEMOTRON_IMAGE_DEFAULT_URL",
    "NEMOTRON_IMAGE_HF_ID",
    "NEMOTRON_IMAGE_MODEL",
    "NEMOTRON_IMAGE_URL_ENV",
    "NemotronSetupError",
    "check_nemotron_image_hardware",
    "check_nemotron_image_server",
    "generate_nemotron_image",
    "nemotron_image_server_url",
    "uses_local_image_server",
]
