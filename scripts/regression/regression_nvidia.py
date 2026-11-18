"""
Run with:
uv run python scripts/regression/regression_nvidia.py
"""

import os
import sys

from regression_core import (
    clear_artifacts,
    run_regression,
    verify_output,
)

# Open models from the NVIDIA API catalog (build.nvidia.com): free, rate limited.
NVIDIA_MODELS = [
    "nvidia-nemotron-3-ultra-550b-a55b",
    "nvidia-gemma-4-31b-it",
]


def main():
    print("=== YouTube-to-Docs Regression: NVIDIA API Catalog Models Test ===")

    if "NVIDIA_API_KEY" not in os.environ:
        print("SKIPPED: NVIDIA_API_KEY not set.")
        sys.exit(0)

    for model in NVIDIA_MODELS:
        print(f"\n--- Testing model: {model} ---")

        clear_artifacts()

        run_regression(
            model=model,
            transcript_model=None,
            infographic_model=None,
            tts_model=None,
            no_youtube_summary=False,
        )

        verify_output(
            model=model,
            transcript_model=None,
            infographic_model=None,
            tts_model=None,
            no_youtube_summary=False,
        )

    print("\n=== SUCCESS: NVIDIA Regression Passed ===")


if __name__ == "__main__":
    main()
