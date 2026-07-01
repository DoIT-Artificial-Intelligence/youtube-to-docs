"""
Regression test for gemini-3.1-flash-lite-image model.

Run with:
    uv run --extra gcp python scripts/regression/regression_flash_lite_image.py
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

INFOGRAPHIC_MODEL = "gemini-3.1-flash-lite-image"


def main():
    print(f"=== YouTube-to-Docs Regression: {INFOGRAPHIC_MODEL} ===")

    # 1. Clear Artifacts
    clear_artifacts()

    # 2. Check Cloud Stack
    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Please check your environment variables.")
        sys.exit(1)

    # 3. Select Model
    selected_model = get_default_model(available_models)
    print(f"\nUsing summary model: {selected_model}")
    print(f"Using infographic model: {INFOGRAPHIC_MODEL}")

    output_target = os.path.join(ARTIFACTS_DIR, "youtube-docs.csv")

    # 4. Run: YouTube transcript + Flash-Lite infographic
    print(f"\n--- YouTube transcript, infographic with {INFOGRAPHIC_MODEL} ---")
    run_regression(
        model=selected_model,
        transcript_model=None,  # use YouTube transcript
        infographic_model=INFOGRAPHIC_MODEL,
        tts_model=None,
        output_target=output_target,
    )

    # 5. Verify
    verify_output(
        model=selected_model,
        transcript_model=None,
        infographic_model=INFOGRAPHIC_MODEL,
        tts_model=None,
        output_target=output_target,
    )

    print(f"\n=== SUCCESS: {INFOGRAPHIC_MODEL} Regression Passed ===")


if __name__ == "__main__":
    main()
