"""
Run with:
uv run --extra gcp python scripts/regression/regression_es_no_yt_summary.py
"""

import sys

from regression_core import (
    check_cloud_stack,
    clear_artifacts,
    get_default_model,
    run_regression,
    verify_output,
)


def main():
    print("=== YouTube-to-Docs Regression: Spanish No YouTube Summary ===")

    # 1. Clear Artifacts
    clear_artifacts()

    # 2. Check Cloud Stack
    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Please check your environment variables.")
        sys.exit(1)

    # 3. Select Model
    selected_model = get_default_model(available_models)
    print(f"\nUsing model: {selected_model}")

    # 4. Run Regression
    transcript_model = "gemini-3-flash-preview"
    infographic_model = "gemini-2.5-flash-image"
    tts_model = "gemini-2.5-flash-preview-tts-Kore"

    run_regression(
        selected_model,
        transcript_model,
        infographic_model,
        tts_model,
        language="es",
        no_youtube_summary=True,
    )

    # 5. Verify Output
    verify_output(
        selected_model,
        transcript_model,
        infographic_model,
        tts_model,
        language="es",
        no_youtube_summary=True,
    )

    print("\n=== SUCCESS: Spanish No YouTube Summary Regression Passed ===")


if __name__ == "__main__":
    main()
