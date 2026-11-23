"""
Regression test for --corrected-captions (default).

Runs bare `-cc`: transcribes with mai-transcribe-2 (word timestamps
and diarization), extracts on-screen context from the video, and corrects
the STT SRT with gemini-4-carbon against the YouTube captions.

Requires:
    GEMINI_API_KEY
    AZURE_TRANSCRIPTION_ENDPOINT
    AZURE_TRANSCRIPTION_API_KEY

Run with:
    uv run --extra all python scripts/regression/regression_cc_default.py
"""

import os
import sys

from regression_core import (
    ARTIFACTS_DIR,
    clear_artifacts,
    run_regression,
    verify_corrected_captions,
)


def main() -> None:
    print("=== Regression: --corrected-captions (default) ===\n")

    missing = [
        var
        for var in (
            "GEMINI_API_KEY",
            "AZURE_TRANSCRIPTION_ENDPOINT",
            "AZURE_TRANSCRIPTION_API_KEY",
        )
        if var not in os.environ
    ]
    if missing:
        print(f"ERROR: {', '.join(missing)} required.")
        sys.exit(1)

    clear_artifacts()

    out = os.path.join(ARTIFACTS_DIR, "regression_cc_default.csv")
    run_regression(
        model=None,
        transcript_model=None,
        infographic_model=None,
        tts_model=None,
        no_youtube_summary=True,
        output_target=out,
        corrected_captions="default",
    )
    verify_corrected_captions(
        out, "Corrected Captions File gemini-4-carbon from mai-transcribe-2"
    )
    print("\n=== SUCCESS ===")


if __name__ == "__main__":
    main()
