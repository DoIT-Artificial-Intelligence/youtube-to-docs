"""
Run with:
uv run --extra all python scripts/regression/regression_aws_transcribe.py
"""

import os
import sys

from regression_core import (
    clear_artifacts,
    run_regression,
    verify_output,
)


def main():
    print("=== YouTube-to-Docs Regression: AWS Transcribe Test ===")

    # Check for S3 bucket
    if "YTD_S3_BUCKET_NAME" not in os.environ:
        print(
            "Error: YTD_S3_BUCKET_NAME environment variable is required for this "
            "regression test."
        )
        sys.exit(1)

    # 1. Clear Artifacts
    clear_artifacts()

    # Run Regression with AWS Transcribe
    model = "bedrock-nova-2-lite-v1"
    transcript_model = "aws-transcribe"
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

    # 3. Verify Output
    verify_output(
        model,
        transcript_model,
        infographic_model,
        tts_model,
        no_youtube_summary=False,
    )

    print("\n=== SUCCESS: AWS Transcribe Regression Passed ===")


if __name__ == "__main__":
    main()
