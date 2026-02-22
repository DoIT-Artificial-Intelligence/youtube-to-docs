"""
Run with:
uv run --extra all python scripts/regression/regression_gcp_tts.py
"""

from regression_core import (
    clear_artifacts,
    run_regression,
    verify_output,
)


def main():
    print("=== YouTube-to-Docs Regression: GCP Chirp3 TTS Test ===")

    # 1. Clear Artifacts
    clear_artifacts()

    # Run Regression with Gemini Pro model and gcp-chirp3 TTS
    model = "gemini-3.1-pro-preview"
    transcript_model = "youtube"  # Use YouTube transcripts
    infographic_model = None  # No infographic for this test
    tts_model = "gcp-chirp3"

    print(f"\nUsing model: {model}")
    print(f"Using TTS: {tts_model}")

    run_regression(
        model,
        transcript_model,
        infographic_model,
        tts_model,
        no_youtube_summary=False,
    )

    # 3. Verify Output
    verify_output(
        model,
        transcript_model,
        infographic_model,
        tts_model,
        no_youtube_summary=False,
    )

    print("\n=== SUCCESS: GCP Chirp3 TTS Regression Passed ===")


if __name__ == "__main__":
    main()
