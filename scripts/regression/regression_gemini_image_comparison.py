"""
Regression test to compare all Gemini image models:
- gemini-3.1-flash-lite-image
- gemini-3.1-flash-image
- gemini-3-pro-image

Run with:
    uv run --extra gcp python scripts/regression/regression_gemini_image_comparison.py
"""

import os
import sys

from regression_core import (
    ARTIFACTS_DIR,
    check_cloud_stack,
    clear_artifacts,
    get_default_model,
    run_regression,
    verify_output,
)

IMAGE_MODELS = [
    "gemini-3.1-flash-lite-image",
    "gemini-3.1-flash-image",
    "gemini-3-pro-image",
]


def main():
    print("=== YouTube-to-Docs Regression: Gemini Image Comparison ===")

    # 1. Clear Artifacts
    clear_artifacts()

    # 2. Check Cloud Stack
    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Please check your environment variables.")
        sys.exit(1)

    # 3. Select Summary Model
    summary_model = get_default_model(available_models)
    print(f"\nUsing summary model: {summary_model}")

    for img_model in IMAGE_MODELS:
        print(f"\n--- Testing Infographic Model: {img_model} ---")
        output_target = os.path.join(ARTIFACTS_DIR, f"youtube-docs-{img_model}.csv")

        # 4. Run: YouTube transcript + Specific infographic model
        run_regression(
            model=summary_model,
            transcript_model=None,  # use YouTube transcript
            infographic_model=img_model,
            tts_model=None,
            output_target=output_target,
        )

        # 5. Verify
        verify_output(
            model=summary_model,
            transcript_model=None,
            infographic_model=img_model,
            tts_model=None,
            output_target=output_target,
        )

    print("\n=== SUCCESS: Gemini Image Comparison Regression Passed ===")


if __name__ == "__main__":
    main()
