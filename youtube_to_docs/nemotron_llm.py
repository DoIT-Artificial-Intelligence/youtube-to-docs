"""
NVIDIA Nemotron LLMs (e.g. `-m nvidia-nemotron-3.5-lightning`).

Runs NVIDIA's open Nemotron checkpoints on your own GPU behind a vLLM server
(see scripts/nemotron_llm/README.md) when a compatible local GPU is detected or
`NEMOTRON_LLM_URL` is set. When no compatible GPU is detected locally, it falls
back to NVIDIA's hosted API catalog at build.nvidia.com (`NVIDIA_API_KEY`).
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from youtube_to_docs.nemotron_transcribe import (
    GpuInfo,
    NemotronSetupError,
    detect_gpus,
)
from youtube_to_docs.providers import BaseProvider, LLMProvider

# The vLLM server's OpenAI-compatible API. Set NEMOTRON_LLM_URL to use a server
# on another machine; otherwise a local GPU is checked and used if compatible.
NEMOTRON_LLM_URL_ENV = "NEMOTRON_LLM_URL"
NEMOTRON_LLM_DEFAULT_URL = "http://127.0.0.1:8000/v1"

# (connect, read) timeout in seconds. Long transcripts with reasoning can take
# several minutes; the limit exists so a stalled server fails instead of hanging.
NEMOTRON_LLM_TIMEOUT_SECONDS = (30, 15 * 60)

NEMOTRON_LIGHTNING_MODEL = "nvidia-nemotron-3.5-lightning"
NEMOTRON_BOLT_MODEL = "nvidia-nemotron-3.5-bolt"
NEMOTRON_SUPER_MODEL = "nvidia-nemotron-3.5-super"


@dataclass(frozen=True)
class NemotronLlmSpec:
    cli_name: str
    hosted_model_name: Optional[str]
    default_hf_id: str
    hf_ids: Tuple[str, ...]
    # (precision, min_compute_cap, min_memory_mib, size_label, example_gpus)
    requirements: Tuple[Tuple[str, float, int, str, str], ...]
    readme_path: str


NEMOTRON_LLM_SPECS: Dict[str, NemotronLlmSpec] = {
    NEMOTRON_LIGHTNING_MODEL: NemotronLlmSpec(
        cli_name=NEMOTRON_LIGHTNING_MODEL,
        hosted_model_name="nvidia-nemotron-3.5-lightning-30b-a3b",
        default_hf_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        hf_ids=(
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B",
        ),
        # nvidia-smi reports a little under the marketing size (a 24 GB L4 shows
        # 23034 MiB, an 80 GB H100 shows 81559 MiB), so the limits leave room.
        # NVFP4 runs natively on Blackwell (CC 10.0+) and via vLLM's Marlin MoE
        # backend on Ada/Hopper (CC 8.9+).
        requirements=(
            ("BF16", 8.0, 76_000, "80 GB", "A100 80GB, H100, H200, B200"),
            (
                "NVFP4",
                8.9,
                22_000,
                "24 GB",
                "L4, RTX 4090, L40S, RTX 5090, RTX PRO 6000",
            ),
        ),
        readme_path="scripts/nemotron_llm/README.md",
    ),
    NEMOTRON_BOLT_MODEL: NemotronLlmSpec(
        cli_name=NEMOTRON_BOLT_MODEL,
        hosted_model_name=None,
        default_hf_id="nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
        hf_ids=(
            "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
            "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16",
            "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B",
        ),
        # Compressed from Super VL (120B-A12B) to 74B-A9B using native 2-bit IQ2
        # quantization to fit in 24 GB VRAM (batch size 1, 128K context). In
        # vLLM, the block-quantized IQ2 checkpoint requires `b12x` on SM12x
        # (compute capability 12.0+, e.g. RTX 5090, RTX PRO 6000 / GCP g4, DGX
        # Spark). BF16 requires ~160 GB total VRAM (e.g. 2x H100 80GB or 1x B200).
        requirements=(
            (
                "BF16",
                8.0,
                155_000,
                "160 GB total",
                "2x H100 80GB, 2x A100 80GB, 1x B200",
            ),
            (
                "IQ2",
                12.0,
                22_000,
                "24 GB Blackwell SM12x",
                "RTX 5090, RTX PRO 6000 (GCP g4), DGX Spark",
            ),
        ),
        readme_path="scripts/nemotron_llm/README.md",
    ),
    NEMOTRON_SUPER_MODEL: NemotronLlmSpec(
        cli_name=NEMOTRON_SUPER_MODEL,
        hosted_model_name="nvidia-nemotron-3.5-super-vl-120b-a12b",
        default_hf_id="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
        hf_ids=(
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8",
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B",
        ),
        # All 120B weights must fit in VRAM (single GPU or tensor-parallel across
        # multiple GPUs). The vision encoder stays in BF16 even for quantized
        # checkpoints, so NVFP4 needs ~80 GB weights (~96 GB GPU like RTX PRO 6000
        # or B200), FP8 needs ~130 GB (1x H200 or 2x H100), BF16 needs ~260 GB.
        requirements=(
            ("BF16", 8.0, 260_000, "260 GB total", "4x H100 80GB, 2x H200, 2x B200"),
            (
                "FP8",
                8.9,
                130_000,
                "130 GB total",
                "1x H200 141GB, 2x H100 80GB, 1x B200",
            ),
            ("NVFP4", 10.0, 90_000, "96 GB Blackwell", "1x RTX PRO 6000 96GB, 1x B200"),
        ),
        readme_path="scripts/nemotron_llm/README.md",
    ),
}

# Maps (server_url, cli_name) -> served model ID discovered from /models.
_served_model_cache: Dict[Tuple[str, str], str] = {}


def is_nemotron_llm_model(model_name: str) -> bool:
    """True when `model_name` is a registered Nemotron LLM with vLLM support."""
    return model_name in NEMOTRON_LLM_SPECS


def get_nemotron_llm_spec(model_name: str) -> NemotronLlmSpec:
    """Returns the spec for a Nemotron LLM or raises ValueError."""
    if model_name not in NEMOTRON_LLM_SPECS:
        raise ValueError(f"Unknown Nemotron LLM: {model_name}")
    return NEMOTRON_LLM_SPECS[model_name]


def best_llm_precision(gpu: GpuInfo, spec: NemotronLlmSpec) -> Optional[str]:
    """The highest-precision checkpoint the GPU can run for `spec`, or None."""
    for precision, min_cc, min_mib, _, _ in spec.requirements:
        if gpu.compute_capability >= min_cc and gpu.memory_mib >= min_mib:
            return precision
    return None


def _requirements_text(spec: NemotronLlmSpec) -> str:
    lines = [
        f"- {precision}: {size}, compute capability {min_cc}+ (e.g. {examples})"
        for precision, min_cc, _, size, examples in spec.requirements
    ]
    return (
        f"{spec.cli_name} runs on your own NVIDIA GPU when available. "
        "It needs one GPU with:\n" + "\n".join(lines)
    )


def check_nemotron_llm_hardware(
    model_name: str = NEMOTRON_LIGHTNING_MODEL,
    gpus: Optional[List[GpuInfo]] = None,
    platform: str = sys.platform,
) -> Tuple[bool, str]:
    """
    Checks whether this machine can run the given Nemotron LLM locally.

    Returns (ok, message). The message names the GPU and checkpoint to use,
    or explains why local GPU execution isn't available.
    """
    spec = get_nemotron_llm_spec(model_name)
    if not platform.startswith("linux"):
        return False, (
            f"{spec.cli_name} needs Linux (or WSL2 on Windows) to run locally; "
            f"this machine runs {platform}.\n{_requirements_text(spec)}"
        )

    if gpus is None:
        gpus = detect_gpus()
    if gpus is None:
        return False, (
            "No NVIDIA driver found (nvidia-smi isn't installed).\n"
            f"{_requirements_text(spec)}"
        )
    if not gpus:
        return False, f"nvidia-smi found no GPUs.\n{_requirements_text(spec)}"

    # Prefer the highest precision: first check if any single GPU can run it,
    # then check aggregate VRAM across multiple compatible GPUs (tensor parallel).
    for precision, min_cc, min_mib, _, _ in spec.requirements:
        single = [
            gpu
            for gpu in gpus
            if gpu.compute_capability >= min_cc and gpu.memory_mib >= min_mib
        ]
        if single:
            gpu = max(single, key=lambda g: g.memory_mib)
            return True, (
                f"{gpu.name} ({gpu.memory_mib / 1024:.0f} GB) can run "
                f"{spec.cli_name} in {precision}."
            )
        compatible = [gpu for gpu in gpus if gpu.compute_capability >= min_cc]
        total_mib = sum(gpu.memory_mib for gpu in compatible)
        if len(compatible) > 1 and total_mib >= min_mib:
            names = ", ".join(dict.fromkeys(gpu.name for gpu in compatible))
            return True, (
                f"{len(compatible)}x {names} ({total_mib / 1024:.0f} GB total) "
                f"can run {spec.cli_name} in {precision}."
            )

    found = ", ".join(
        f"{gpu.name} ({gpu.memory_mib / 1024:.0f} GB, compute capability "
        f"{gpu.compute_capability})"
        for gpu in gpus
    )
    return False, (f"Found {found}, which is too small.\n{_requirements_text(spec)}")


def nemotron_llm_server_url() -> str:
    """The vLLM server's base URL, without a trailing slash."""
    url = os.environ.get(NEMOTRON_LLM_URL_ENV) or NEMOTRON_LLM_DEFAULT_URL
    return url.rstrip("/")


def uses_local_llm_server() -> bool:
    """True when no remote NEMOTRON_LLM_URL is configured."""
    return not os.environ.get(NEMOTRON_LLM_URL_ENV)


def use_self_hosted_nemotron_llm(model_name: str = NEMOTRON_LIGHTNING_MODEL) -> bool:
    """True when requests should go to a vLLM server instead of build.nvidia.com."""
    if not uses_local_llm_server():
        return True
    spec = get_nemotron_llm_spec(model_name)
    if spec.hosted_model_name is None:
        return True
    hardware_ok, _ = check_nemotron_llm_hardware(model_name)
    return hardware_ok


def _server_hint(spec: NemotronLlmSpec) -> str:
    return (
        f"Start the vLLM server (see {spec.readme_path}), or "
        f"set {NEMOTRON_LLM_URL_ENV} to a server on another machine."
    )


def _match_served_model(served: List[str], spec: NemotronLlmSpec) -> Optional[str]:
    """Finds a served model ID matching one of the spec's HF IDs."""
    for hf_id in spec.hf_ids:
        if hf_id in served:
            return hf_id
    return None


def check_nemotron_llm_server(
    model_name: str = NEMOTRON_LIGHTNING_MODEL,
) -> Tuple[bool, str]:
    """Checks that the vLLM server is up and serving `model_name`."""
    spec = get_nemotron_llm_spec(model_name)
    url = nemotron_llm_server_url()
    try:
        response = requests.get(f"{url}/models", timeout=10)
        response.raise_for_status()
        served = [
            str(model.get("id"))
            for model in response.json().get("data", [])
            if model.get("id")
        ]
    except Exception as e:
        return False, f"No Nemotron LLM server at {url} ({e}).\n{_server_hint(spec)}"

    matched = _match_served_model(served, spec)
    if matched is None:
        return False, (
            f"The server at {url} serves {', '.join(served) or 'nothing'}, "
            f"not {spec.default_hf_id}.\n{_server_hint(spec)}"
        )
    _served_model_cache[(url, model_name)] = matched
    return True, f"Using the Nemotron LLM server at {url} ({matched})."


def _resolve_served_model(model_name: str) -> str:
    """Returns the served model ID on the vLLM server, falling back to default."""
    spec = get_nemotron_llm_spec(model_name)
    url = nemotron_llm_server_url()
    cached = _served_model_cache.get((url, model_name))
    if cached:
        return cached
    try:
        response = requests.get(f"{url}/models", timeout=10)
        response.raise_for_status()
        served = [
            str(model.get("id"))
            for model in response.json().get("data", [])
            if model.get("id")
        ]
        matched = _match_served_model(served, spec)
        if matched:
            _served_model_cache[(url, model_name)] = matched
            return matched
    except Exception:
        pass
    return spec.default_hf_id


def generate_nemotron_llm(model_name: str, prompt: str) -> Tuple[str, int, int]:
    """Calls a self-hosted Nemotron vLLM server for chat completions."""
    url = nemotron_llm_server_url()
    served_model = _resolve_served_model(model_name)
    try:
        response = requests.post(
            f"{url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": served_model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=NEMOTRON_LLM_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return (
                f"Error: Nemotron LLM server error {response.status_code}: "
                f"{response.text}",
                0,
                0,
            )
        data = response.json()
        # Reasoning models put their thinking in `reasoning_content`; only
        # the final answer is kept.
        response_text = ((data.get("choices") or [{}])[0].get("message") or {}).get(
            "content"
        ) or ""
        usage = data.get("usage") or {}
        return (
            response_text,
            usage.get("prompt_tokens") or 0,
            usage.get("completion_tokens") or 0,
        )
    except Exception as e:
        return f"Error: Nemotron LLM server failed: {e}", 0, 0


class NemotronLlmProvider(BaseProvider, LLMProvider):
    def generate_content(self, prompt: str, **kwargs) -> Tuple[str, int, int]:
        if use_self_hosted_nemotron_llm(self.model_name):
            return generate_nemotron_llm(self.model_name, prompt)
        from youtube_to_docs.llms import NvidiaProvider

        spec = get_nemotron_llm_spec(self.model_name)
        if spec.hosted_model_name is None:
            raise NemotronSetupError(
                f"{self.model_name} is self-hosted only (not available on "
                f"build.nvidia.com). {_server_hint(spec)}"
            )
        return NvidiaProvider(spec.hosted_model_name).generate_content(prompt, **kwargs)


__all__ = [
    "NEMOTRON_BOLT_MODEL",
    "NEMOTRON_LIGHTNING_MODEL",
    "NEMOTRON_LLM_DEFAULT_URL",
    "NEMOTRON_LLM_SPECS",
    "NEMOTRON_LLM_URL_ENV",
    "NEMOTRON_SUPER_MODEL",
    "NemotronLlmProvider",
    "NemotronLlmSpec",
    "NemotronSetupError",
    "best_llm_precision",
    "check_nemotron_llm_hardware",
    "check_nemotron_llm_server",
    "generate_nemotron_llm",
    "get_nemotron_llm_spec",
    "is_nemotron_llm_model",
    "nemotron_llm_server_url",
    "use_self_hosted_nemotron_llm",
    "uses_local_llm_server",
]
