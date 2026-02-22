"""
Run with:
uv run --extra all python scripts/regression/regression_gcp_stt_long.py
"""

import regression_core
from regression_core import (
    clear_artifacts,
    run_regression,
    verify_output,
)

# Override video ID for long video test
regression_core.VIDEO_ID = "g0HRfR19R_w"


def main():
    print("=== YouTube-to-Docs Regression: GCP Chirp3 STT Long Video Test ===")

    # 1. Clear Artifacts
    clear_artifacts()

    # Run Regression with GCP Chirp3 STT
    model = "gemini-3-flash-preview"
    transcript_model = "gcp-chirp3"
    infographic_model = None
    tts_model = None

    print(f"\nUsing model: {model}")
    print(f"Using STT: {transcript_model}")

    run_regression(
        model,
        transcript_model,
        infographic_model,
        tts_model,
        no_youtube_summary=True,  # Skip YT summary to save time/tokens
        verbose=True,
    )

    # 3. Verify Output
    verify_output(
        model,
        transcript_model,
        infographic_model,
        tts_model,
        no_youtube_summary=True,
    )

    print("\n=== SUCCESS: GCP Chirp3 STT Long Video Regression Passed ===")


if __name__ == "__main__":
    main()
