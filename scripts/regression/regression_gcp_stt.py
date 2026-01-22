import os
import sys

import polars as pl
from regression_core import (
    clear_artifacts,
    run_regression,
)


def main():
    print("=== YouTube-to-Docs Regression: GCP STT (Chirp) ===")

    if "GOOGLE_CLOUD_PROJECT" not in os.environ:
        print("Skipping: GOOGLE_CLOUD_PROJECT not set")
        sys.exit(0)
    # YTD_GCS_BUCKET_NAME defaults to "youtube-to-docs" in code now, so we don't
    # strictly enforce it here
    # but strictly checking PROJECT_ID is good.
    if "GOOGLE_CLOUD_PROJECT" not in os.environ:
        print("Skipping: GOOGLE_CLOUD_PROJECT not set")
        sys.exit(0)
    # if "YTD_GCS_BUCKET_NAME" not in os.environ:
    #     print("Skipping: YTD_GCS_BUCKET_NAME not set")
    #     sys.exit(0)

    # 1. Clear Artifacts
    clear_artifacts()

    # 2. Config
    # We will use a standard model for summary, but test the GCP transcript
    selected_model = "gemini-3-flash-preview"
    transcript_model = "gcp-chirp3"

    # Check if Gemini is available (since we use it for summary)
    if "GEMINI_API_KEY" not in os.environ:
        print(
            "Warning: GEMINI_API_KEY not set. Summary might fail if default "
            "youtube transcript is not used, "
            "but here we use gcp transcript."
        )
        # Actually if we use gcp-chirp3, we have a transcript.
        # Summary needs a model.

    print(f"\nUsing Summary Model: {selected_model}")
    print(f"Using Transcript Model: {transcript_model}")

    # 3. Run Regression
    run_regression(
        model=selected_model,
        transcript_model=transcript_model,
        infographic_model=None,
        tts_model=None,
        language="en",
        no_youtube_summary=True,  # Simplify test
        verbose=True,
    )

    # 4. Verify Output (Custom for GCP Test)
    print("\n--- Verifying Output (Custom) ---")
    df = pl.read_csv("youtube-to-docs-test/youtube-docs.csv")

    expected_partial = [
        "Transcript File gcp-chirp3 generated",
        "gcp-chirp3 STT cost ($)",  # Now that we added it to prices
    ]

    missing = [c for c in expected_partial if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        sys.exit(1)

    print("Verification PASSED!")

    print("\n=== SUCCESS: GCP STT Regression Passed ===")


if __name__ == "__main__":
    main()
