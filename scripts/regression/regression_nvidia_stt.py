"""
Run with:
uv run --extra all python scripts/regression/regression_nvidia_stt.py
"""

import os
import sys

from regression_core import (
    clear_artifacts,
    run_regression,
    verify_output,
)


def main():
    print("=== YouTube-to-Docs Regression: NVIDIA Parakeet STT Test ===")

    if "NVIDIA_API_KEY" not in os.environ:
        print("SKIPPED: NVIDIA_API_KEY not set.")
        sys.exit(0)

    clear_artifacts()

    model = "gemini-3.5-flash-lite"
    transcript_model = "nvidia-parakeet-tdt-0.6b-v2"
    infographic_model = None
    tts_model = None

    print(f"\nUsing model: {model}")
    print(f"Using STT: {transcript_model}")

    run_regression(
        model,
        transcript_model,
        infographic_model,
        tts_model,
        no_youtube_summary=False,
    )

    verify_output(
        model,
        transcript_model,
        infographic_model,
        tts_model,
        no_youtube_summary=False,
    )

    print("\n=== SUCCESS: NVIDIA Parakeet STT Regression Passed ===")


if __name__ == "__main__":
    main()
