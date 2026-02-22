"""
Run with:
uv run --extra aws --extra gcp python scripts/regression/regression_aws_translate.py
"""

import sys

from regression_core import (
    check_cloud_stack,
    clear_artifacts,
    run_regression,
)


def main():
    print("=== YouTube-to-Docs Regression: AWS Translate (French) ===")

    # 1. Clear Artifacts
    clear_artifacts()

    # 2. Check Cloud Stack
    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Please check your environment variables.")
        sys.exit(1)

    # 3. Select Model
    selected_model = "bedrock-nova-2-lite-v1"
    print(f"\nUsing model: {selected_model}")

    # 4. Run Regression with AWS Translate (Simplified)
    transcript_model = "youtube"
    infographic_model = None
    tts_model = None

    run_regression(
        selected_model,
        transcript_model,
        infographic_model,
        tts_model,
        language="gemini-fr",
    )

    # run_regression already calls verify_output internally.
    # No need for manual call unless extra flags are needed.

    print("\n=== SUCCESS: AWS Translate Regression Passed ===")


if __name__ == "__main__":
    main()
