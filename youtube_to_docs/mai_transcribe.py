"""
Microsoft MAI-Transcribe-2 (`-t mai-transcribe-2`).

MAI-Transcribe is Microsoft AI's speech-to-text model, served by the Azure
Speech fast transcription API ("enhanced mode") on a Foundry resource. Like
`gemini-3.5-transcribe`, it's asked for word timestamps and speaker
diarization, so the SRT carries real offsets and `[Speaker N]` labels.

Needs AZURE_TRANSCRIPTION_ENDPOINT and AZURE_TRANSCRIPTION_API_KEY.
"""

import json
import math
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import requests
from rich import print as rprint

from youtube_to_docs.providers import BaseProvider, STTProvider

MAI_TRANSCRIBE_PREFIX = "mai-transcribe"
MAI_TRANSCRIBE_ENDPOINT_ENV = "AZURE_TRANSCRIPTION_ENDPOINT"
MAI_TRANSCRIBE_API_KEY_ENV = "AZURE_TRANSCRIPTION_API_KEY"
MAI_TRANSCRIBE_API_VERSION = "2025-10-15"

# One request takes up to 2 hours and 250 MB, and less with diarization (the
# docs don't say how much less). 30 minutes of 16 kHz mono FLAC is about 60 MB.
MAI_TRANSCRIBE_CHUNK_SECONDS = 30 * 60

# Seconds to connect, and to wait for one chunk's transcript.
MAI_TRANSCRIBE_TIMEOUT_SECONDS = (30, 15 * 60)


def is_mai_transcribe_model(model_name: str) -> bool:
    """True for Microsoft MAI-Transcribe models (e.g. `mai-transcribe-2`)."""
    return model_name.startswith(MAI_TRANSCRIBE_PREFIX)


def _mai_entries(
    response: Dict[str, Any],
    offset_seconds: float,
    srt_counter: int,
    speaker_numbers: Dict[str, int],
    speaker_scope: str,
) -> Tuple[List[str], List[str], int]:
    """
    Turns one API response into text parts and SRT entries.

    Each phrase is one speaker's turn. A `[Speaker N]` label is added when the
    speaker changes. Speaker numbers are only consistent within one request,
    so `speaker_scope` keeps chunks apart.
    Returns (text_parts, srt_entries, next_srt_counter).
    """
    from youtube_to_docs.llms import _words_to_srt_entries

    text_parts: List[str] = []
    srt_entries: List[str] = []
    previous_speaker: Optional[str] = None

    for phrase in response.get("phrases") or []:
        text = (phrase.get("text") or "").strip()
        if text:
            text_parts.append(text)

        speaker = phrase.get("speaker")
        speaker = None if speaker is None else str(speaker)
        label = None
        if speaker is not None and speaker != previous_speaker:
            key = f"{speaker_scope}{speaker}"
            speaker_numbers.setdefault(key, len(speaker_numbers) + 1)
            label = f"[Speaker {speaker_numbers[key]}]"
        previous_speaker = speaker

        words = [
            (
                word["text"],
                word.get("offsetMilliseconds", 0) / 1000 + offset_seconds,
                (
                    word.get("offsetMilliseconds", 0)
                    + word.get("durationMilliseconds", 0)
                )
                / 1000
                + offset_seconds,
            )
            for word in phrase.get("words") or []
            if word.get("text")
        ]
        if not words and text:
            # No word timings: time the whole phrase as one entry.
            start = phrase.get("offsetMilliseconds", 0) / 1000 + offset_seconds
            end = start + phrase.get("durationMilliseconds", 0) / 1000
            words = [(text, start, end)]
        entries, srt_counter = _words_to_srt_entries(words, srt_counter, label)
        srt_entries.extend(entries)

    return text_parts, srt_entries, srt_counter


def _transcribe_chunk(
    model_name: str, audio_path: str, language: Optional[str]
) -> Dict[str, Any]:
    """Sends one FLAC file to the fast transcription API and returns its JSON."""
    endpoint = os.environ[MAI_TRANSCRIBE_ENDPOINT_ENV].rstrip("/")
    definition: Dict[str, Any] = {
        "diarization": {"enabled": True},
        # The default style is verbatim (fillers kept), like the other STT models.
        "enhancedMode": {
            "enabled": True,
            "model": model_name,
            "modelOptions": {"timestamps": "word"},
        },
    }
    if language:
        definition["locales"] = [language]
    with open(audio_path, "rb") as f:
        response = requests.post(
            f"{endpoint}/speechtotext/transcriptions:transcribe",
            params={"api-version": MAI_TRANSCRIBE_API_VERSION},
            headers={
                "Ocp-Apim-Subscription-Key": os.environ[MAI_TRANSCRIBE_API_KEY_ENV]
            },
            files={
                "audio": (os.path.basename(audio_path), f, "audio/flac"),
                "definition": (None, json.dumps(definition), "application/json"),
            },
            timeout=MAI_TRANSCRIBE_TIMEOUT_SECONDS,
        )
    response.raise_for_status()
    return response.json()


def transcribe_mai(
    model_name: str,
    audio_path: str,
    url: str,
    language: Optional[str] = "en",
    duration_seconds: Optional[float] = None,
) -> Tuple[str, str, int, int]:
    """
    Transcribes audio with MAI-Transcribe, with word timestamps and speaker
    labels.

    The service bills per minute, so the token counts are per-minute
    pseudo-tokens (see `minute_pseudo_tokens`).
    Returns (transcript_text, srt_content, input_tokens, output_tokens).
    """
    from youtube_to_docs.llms import minute_pseudo_tokens

    if not (
        os.environ.get(MAI_TRANSCRIBE_ENDPOINT_ENV)
        and os.environ.get(MAI_TRANSCRIBE_API_KEY_ENV)
    ):
        # A fixed message, so nothing read near the key reaches the transcript.
        return (
            "Error: set AZURE_TRANSCRIPTION_ENDPOINT and "
            "AZURE_TRANSCRIPTION_API_KEY to use MAI-Transcribe",
            "",
            0,
            0,
        )

    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
    except ImportError:
        pass  # Fall back to an ffmpeg already on the PATH.

    chunk_count = max(
        1, math.ceil((duration_seconds or 0) / MAI_TRANSCRIBE_CHUNK_SECONDS)
    )
    if chunk_count > 1:
        rprint(
            f"[cyan]Audio is long ({duration_seconds:.0f}s). Transcribing in "
            f"{chunk_count} chunks...[/cyan]"
        )

    text_parts: List[str] = []
    srt_entries: List[str] = []
    srt_counter = 1
    speaker_numbers: Dict[str, int] = {}
    billed_seconds = 0.0
    print(f"Starting transcription with model: {model_name}...")
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(chunk_count):
            offset = i * MAI_TRANSCRIBE_CHUNK_SECONDS
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.flac")
            # The API takes WAV, MP3 or FLAC (not M4A); 16 kHz mono FLAC is
            # lossless and small.
            command = ["ffmpeg", "-y", "-loglevel", "error"]
            if chunk_count > 1:
                command += ["-ss", str(offset), "-t", str(MAI_TRANSCRIBE_CHUNK_SECONDS)]
            command += ["-i", audio_path, "-vn", "-ac", "1", "-ar", "16000"]
            command += ["-c:a", "flac", chunk_path]
            try:
                subprocess.run(command, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                return f"Error: converting audio with ffmpeg failed: {e}", "", 0, 0

            if chunk_count > 1:
                rprint(f"  Transcribing chunk {i + 1}/{chunk_count}...")
            try:
                response = _transcribe_chunk(model_name, chunk_path, language)
            except requests.HTTPError as e:
                detail = e.response.text[:500] if e.response is not None else ""
                return f"Error: {model_name} failed: {e} {detail}", "", 0, 0
            except Exception as e:
                return f"Error: {model_name} failed: {e}", "", 0, 0

            billed_seconds += (response.get("durationMilliseconds") or 0) / 1000
            parts, entries, srt_counter = _mai_entries(
                response, offset, srt_counter, speaker_numbers, f"chunk{i}:"
            )
            text_parts.extend(parts)
            srt_entries.extend(entries)

    tokens = minute_pseudo_tokens(billed_seconds or duration_seconds)
    transcript_text = " ".join(text_parts)
    if not transcript_text:
        return (
            f"Error: {model_name} returned no transcript for {url}",
            "",
            tokens,
            tokens,
        )
    return transcript_text, "\n".join(srt_entries), tokens, tokens


class MaiTranscribeProvider(BaseProvider, STTProvider):
    def transcribe(
        self,
        audio_path: str,
        url: str,
        language: str = "en",
        duration_seconds: Optional[float] = None,
        **kwargs,
    ) -> Tuple[str, str, int, int]:
        return transcribe_mai(
            self.model_name, audio_path, url, language, duration_seconds
        )
