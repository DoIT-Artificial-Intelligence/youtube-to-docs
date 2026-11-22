"""
NVIDIA Nemotron 3.5 Transcribe (`-t nvidia-nemotron-3.5-transcribe`).

NVIDIA publishes the model as open weights
(https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) but
doesn't host it at build.nvidia.com, so it runs on the user's own GPU behind
a vLLM server (see scripts/nemotron_transcribe/README.md). This module checks
that the machine has a GPU that can run it and transcribes through the server.
"""

import base64
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from rich import print as rprint

from youtube_to_docs.providers import BaseProvider, STTProvider

NEMOTRON_TRANSCRIBE_MODEL = "nvidia-nemotron-3.5-transcribe"
NEMOTRON_TRANSCRIBE_HF_ID = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"

# The vLLM server's OpenAI-compatible API. Set NEMOTRON_TRANSCRIBE_URL to use a
# server on another machine; otherwise this machine runs it and must have a GPU.
NEMOTRON_TRANSCRIBE_URL_ENV = "NEMOTRON_TRANSCRIBE_URL"
NEMOTRON_TRANSCRIBE_DEFAULT_URL = "http://127.0.0.1:8000/v1"

# One request holds up to an hour of audio in a 65,536-token context. An hour
# of audio is about 45k prompt tokens plus about 14k transcript tokens, so
# longer audio is sent in 45-minute chunks to leave room.
NEMOTRON_CHUNK_SECONDS = 45 * 60

# Seconds to wait for one chunk. A 9-minute recording takes about 6 seconds on
# an H100; the limit exists so a stalled server fails instead of hanging.
NEMOTRON_TIMEOUT_SECONDS = (30, 30 * 60)

# Clean (non-verbatim) speaker-attributed transcription prompt based on the
# model card's wording, matching MAI-Transcribe-2 and Gemini Transcribe style
# (no fillers/repetitions, digits for numbers) while keeping `<spk:N>` tags and
# per-word CTC timestamps.
NEMOTRON_PROMPT = (
    "Transcribe this audio with punctuation and capitalization. Omit fillers "
    "and accidental repetitions; keep the remaining spoken wording unchanged. "
    "Write numbers as digits and use standard abbreviations. Put speaker tags "
    "<spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
)

_SPEAKER_TAG = re.compile(r"\s*<spk:\d+>\s*")

# What one GPU needs for each checkpoint, from the model card: (precision,
# minimum compute capability, minimum memory in MiB, memory as sold, example
# GPUs).
# nvidia-smi reports a little under the marketing size (an 80 GB H100 shows
# 81559 MiB, a 48 GB L40S 46068 MiB, a 32 GB RTX 5090 32607 MiB), so the
# limits leave some room.
NEMOTRON_REQUIREMENTS: Tuple[Tuple[str, float, int, str, str], ...] = (
    ("BF16", 8.0, 76_000, "80 GB", "A100 80GB, H100, H200, B200"),
    ("FP8", 8.9, 44_000, "48 GB", "L40S, H100, RTX PRO 6000"),
    ("NVFP4", 10.0, 30_000, "32 GB", "RTX 5090, RTX PRO 6000"),
)


class NemotronSetupError(RuntimeError):
    """This machine or its Nemotron server can't run the transcription."""


@dataclass
class GpuInfo:
    name: str
    memory_mib: int
    compute_capability: float


def detect_gpus() -> Optional[List[GpuInfo]]:
    """
    Lists the NVIDIA GPUs that nvidia-smi can see.

    Returns None when nvidia-smi isn't installed (no NVIDIA driver), or an
    empty list when it finds no GPUs.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, memory, compute_cap = parts
        try:
            gpus.append(GpuInfo(name, int(float(memory)), float(compute_cap)))
        except ValueError:
            continue
    return gpus


def best_precision(gpu: GpuInfo) -> Optional[str]:
    """The highest-precision checkpoint the GPU can run, or None."""
    for precision, min_cc, min_mib, _, _ in NEMOTRON_REQUIREMENTS:
        if gpu.compute_capability >= min_cc and gpu.memory_mib >= min_mib:
            return precision
    return None


def _requirements_text() -> str:
    lines = [
        f"- {precision}: {size}, compute capability {min_cc}+ (e.g. {examples})"
        for precision, min_cc, _, size, examples in NEMOTRON_REQUIREMENTS
    ]
    return (
        f"{NEMOTRON_TRANSCRIBE_MODEL} runs on your own NVIDIA GPU. "
        "It needs one GPU with:\n" + "\n".join(lines)
    )


def check_nemotron_hardware(
    gpus: Optional[List[GpuInfo]] = None, platform: str = sys.platform
) -> Tuple[bool, str]:
    """
    Checks whether this machine can run Nemotron 3.5 Transcribe.

    Returns (ok, message). The message names the GPU and checkpoint to use,
    or explains what's missing.
    """
    if not platform.startswith("linux"):
        return False, (
            f"{NEMOTRON_TRANSCRIBE_MODEL} needs Linux (or WSL2 on Windows); "
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

    usable = [(gpu, best_precision(gpu)) for gpu in gpus]
    usable = [(gpu, precision) for gpu, precision in usable if precision]
    if not usable:
        found = ", ".join(
            f"{gpu.name} ({gpu.memory_mib / 1024:.0f} GB, compute capability "
            f"{gpu.compute_capability})"
            for gpu in gpus
        )
        return False, f"Found {found}, which is too small.\n{_requirements_text()}"

    # Prefer the GPU that runs the most precise checkpoint, then the most memory.
    order = [precision for precision, *_ in NEMOTRON_REQUIREMENTS]
    gpu, precision = min(
        usable, key=lambda item: (order.index(item[1]), -item[0].memory_mib)
    )
    return True, (
        f"{gpu.name} ({gpu.memory_mib / 1024:.0f} GB) can run "
        f"{NEMOTRON_TRANSCRIBE_HF_ID} in {precision}."
    )


def nemotron_server_url() -> str:
    """The vLLM server's base URL, without a trailing slash."""
    url = os.environ.get(NEMOTRON_TRANSCRIBE_URL_ENV) or (
        NEMOTRON_TRANSCRIBE_DEFAULT_URL
    )
    return url.rstrip("/")


def uses_local_server() -> bool:
    """True when the server runs on this machine, so its GPU must be checked."""
    return not os.environ.get(NEMOTRON_TRANSCRIBE_URL_ENV)


def _server_hint() -> str:
    return (
        f"Start the vLLM server (see scripts/nemotron_transcribe/README.md), or "
        f"set {NEMOTRON_TRANSCRIBE_URL_ENV} to a server on another machine."
    )


def check_nemotron_server() -> Tuple[bool, str]:
    """Checks that the vLLM server is up and serving the model."""
    url = nemotron_server_url()
    try:
        response = requests.get(f"{url}/models", timeout=10)
        response.raise_for_status()
        served = [model.get("id") for model in response.json().get("data", [])]
    except Exception as e:
        return False, f"No Nemotron server at {url} ({e}).\n{_server_hint()}"
    if NEMOTRON_TRANSCRIBE_HF_ID not in served:
        return False, (
            f"The server at {url} serves {', '.join(map(str, served)) or 'nothing'}, "
            f"not {NEMOTRON_TRANSCRIBE_HF_ID}.\n{_server_hint()}"
        )
    return True, f"Using the Nemotron server at {url}."


def _nemotron_entries(
    response: Dict[str, Any],
    offset_seconds: float,
    srt_counter: int,
    speaker_numbers: Dict[str, int],
    speaker_scope: str,
) -> Tuple[str, List[str], int]:
    """
    Turns one server response into text and SRT entries.

    A `[Speaker N]` label starts each speaker's turn. Speaker numbers are only
    consistent within one request, so `speaker_scope` keeps chunks apart.
    Returns (text, srt_entries, next_srt_counter).
    """
    from youtube_to_docs.llms import _words_to_srt_entries

    choices = response.get("choices") or [{}]
    content = (choices[0].get("message") or {}).get("content") or ""
    text = _SPEAKER_TAG.sub(" ", content).strip()

    words = (response.get("ctc_timestamps") or {}).get("words") or []
    srt_entries: List[str] = []
    turn: List[Tuple[str, float, float]] = []
    turn_speaker: Optional[str] = None

    def flush() -> None:
        nonlocal srt_counter, turn
        if not turn:
            return
        label = None
        if turn_speaker is not None:
            key = f"{speaker_scope}{turn_speaker}"
            speaker_numbers.setdefault(key, len(speaker_numbers) + 1)
            label = f"[Speaker {speaker_numbers[key]}]"
        entries, srt_counter = _words_to_srt_entries(turn, srt_counter, label)
        srt_entries.extend(entries)
        turn = []

    for word in words:
        if not word.get("word"):
            continue
        speaker = word.get("speaker")
        speaker = None if speaker is None else str(speaker)
        if speaker != turn_speaker:
            flush()
            turn_speaker = speaker
        turn.append(
            (
                word["word"],
                float(word["start"]) + offset_seconds,
                float(word["end"]) + offset_seconds,
            )
        )
    flush()
    return text, srt_entries, srt_counter


def _transcribe_chunk(audio_path: str) -> Dict[str, Any]:
    """Sends one FLAC file to the server and returns its JSON response."""
    with open(audio_path, "rb") as f:
        audio = base64.b64encode(f.read()).decode()
    body = {
        "model": NEMOTRON_TRANSCRIBE_HF_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": NEMOTRON_PROMPT},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/flac;base64,{audio}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        # The speaker tags are special tokens.
        "skip_special_tokens": False,
        "mm_processor_kwargs": {"capture_ctc_timestamps": True},
    }
    response = requests.post(
        f"{nemotron_server_url()}/chat/completions",
        json=body,
        timeout=NEMOTRON_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def transcribe_nemotron(
    audio_path: str, url: str, duration_seconds: Optional[float] = None
) -> Tuple[str, str, int, int]:
    """
    Transcribes audio with Nemotron 3.5 Transcribe on a vLLM server, with word
    timestamps and speaker labels. The language is detected from the audio.

    Returns (transcript_text, srt_content, input_tokens, output_tokens).
    """
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
    except ImportError:
        pass  # Fall back to an ffmpeg already on the PATH.

    chunk_count = max(1, math.ceil((duration_seconds or 0) / NEMOTRON_CHUNK_SECONDS))
    if chunk_count > 1:
        rprint(
            f"[cyan]Audio is long ({duration_seconds:.0f}s). Transcribing in "
            f"{chunk_count} chunks...[/cyan]"
        )

    text_parts: List[str] = []
    srt_entries: List[str] = []
    srt_counter = 1
    speaker_numbers: Dict[str, int] = {}
    input_tokens = output_tokens = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(chunk_count):
            offset = i * NEMOTRON_CHUNK_SECONDS
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.flac")
            # 16 kHz mono FLAC is what the model expects and is lossless.
            command = ["ffmpeg", "-y", "-loglevel", "error"]
            if chunk_count > 1:
                command += ["-ss", str(offset), "-t", str(NEMOTRON_CHUNK_SECONDS)]
            command += ["-i", audio_path, "-vn", "-ac", "1", "-ar", "16000"]
            command += ["-c:a", "flac", chunk_path]
            try:
                subprocess.run(command, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                return f"Error: converting audio with ffmpeg failed: {e}", "", 0, 0

            if chunk_count > 1:
                rprint(f"  Transcribing chunk {i + 1}/{chunk_count}...")
            try:
                response = _transcribe_chunk(chunk_path)
            except requests.HTTPError as e:
                detail = e.response.text[:500] if e.response is not None else ""
                return f"Error: Nemotron server failed: {e} {detail}", "", 0, 0
            except Exception as e:
                return f"Error: Nemotron server failed: {e}", "", 0, 0

            usage = response.get("usage") or {}
            input_tokens += usage.get("prompt_tokens") or 0
            output_tokens += usage.get("completion_tokens") or 0
            text, entries, srt_counter = _nemotron_entries(
                response, offset, srt_counter, speaker_numbers, f"chunk{i}:"
            )
            if text:
                text_parts.append(text)
            srt_entries.extend(entries)

    transcript_text = " ".join(text_parts)
    if not transcript_text:
        return (
            f"Error: {NEMOTRON_TRANSCRIBE_MODEL} returned no transcript for {url}",
            "",
            input_tokens,
            output_tokens,
        )
    return transcript_text, "\n".join(srt_entries), input_tokens, output_tokens


class NemotronTranscribeProvider(BaseProvider, STTProvider):
    def transcribe(
        self,
        audio_path: str,
        url: str,
        language: str = "en",
        duration_seconds: Optional[float] = None,
        **kwargs,
    ) -> Tuple[str, str, int, int]:
        return transcribe_nemotron(audio_path, url, duration_seconds)
