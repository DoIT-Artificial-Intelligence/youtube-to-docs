"""
Regression test for --corrected-captions (GCP Chirp 3 source).

Transcribes with GCP Chirp 3 (`-t gcp-chirp3`) and corrects that SRT in
the same run with `-cc {model}-gcp-chirp3`, using the YouTube captions as
the second transcript. Chirp 3 has no diarization, so speaker changes
come from the YouTube `>>` marks, the video and the content.

Requires:
    GOOGLE_CLOUD_PROJECT  — GCP project ID
    GEMINI_API_KEY        — for the correction model
    YTD_GCS_BUCKET_NAME   — recommended for GCP STT temp storage

Run with:
    uv run --extra all python scripts/regression/regression_cc_stt.py

To force a specific correction model:
    CC_MODEL=gemini-3.5-flash-lite uv run --extra all python \
        scripts/regression/regression_cc_stt.py
"""

import os
import sys

from regression_core import (
    ARTIFACTS_DIR,
    check_cloud_stack,
    clear_artifacts,
    get_default_model,
    run_regression,
    verify_corrected_captions,
)


def main() -> None:
    print("=== Regression: --corrected-captions (GCP Chirp 3 source) ===\n")

    if "GOOGLE_CLOUD_PROJECT" not in os.environ:
        print("ERROR: GOOGLE_CLOUD_PROJECT is required for GCP Chirp 3 STT.")
        sys.exit(1)

    clear_artifacts()

    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Check your environment variables.")
        sys.exit(1)
    model = os.environ.get("CC_MODEL") or get_default_model(available_models)
    print(f"Correction model: {model}")

    out = os.path.join(ARTIFACTS_DIR, "regression_cc_stt.csv")
    run_regression(
        model=None,
        transcript_model="gcp-chirp3",
        infographic_model=None,
        tts_model=None,
        no_youtube_summary=True,
        output_target=out,
        corrected_captions=f"{model}-gcp-chirp3",
    )
    verify_corrected_captions(out, f"Corrected Captions File {model} from gcp-chirp3")
    print("\n=== SUCCESS ===")


if __name__ == "__main__":
    main()
