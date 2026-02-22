"""
Run with:
uv run --extra all python scripts/regression/regression_aws_polly.py
"""

from regression_core import (
    clear_artifacts,
    run_regression,
    verify_output,
)


def main():
    print("=== YouTube-to-Docs Regression: AWS Polly TTS Test ===")

    # 1. Clear Artifacts
    clear_artifacts()

    # Run Regression with AWS Bedrock Nova Lite model and aws-polly TTS
    # Using Nova Lite for speed/cost efficiency and to keep it all-AWS
    model = "bedrock-nova-2-lite-v1"
    transcript_model = "youtube"  # Use YouTube transcripts
    infographic_model = None  # No infographic for this test
    tts_model = "aws-polly"

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

    print("\n=== SUCCESS: AWS Polly TTS Regression Passed ===")


if __name__ == "__main__":
    main()
