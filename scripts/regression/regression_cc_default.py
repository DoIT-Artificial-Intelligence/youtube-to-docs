"""
Regression test for --corrected-captions (default).

Runs bare `-cc`: transcribes with gemini-3.5-transcribe (word timestamps
and diarization), extracts on-screen context from the video, and corrects
the STT SRT with gemini-4-argon against the YouTube captions.

Requires:
    GEMINI_API_KEY

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

    if "GEMINI_API_KEY" not in os.environ:
        print("ERROR: GEMINI_API_KEY is required.")
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
        out, "Corrected Captions File gemini-4-argon from gemini-3.5-transcribe"
    )
    print("\n=== SUCCESS ===")


if __name__ == "__main__":
    main()
