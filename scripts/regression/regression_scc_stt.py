"""
Regression test for --suggest-corrected-captions using a speech-to-text
generated SRT as the correction source.

Downloads audio for the test video, transcribes it with GCP Chirp3
(speech-to-text), then asks an LLM to suggest WCAG 2.1 Level AA compliant
corrections on the resulting SRT. Because no source suffix is passed to -scc,
the correction step automatically picks up the most recent AI-generated SRT
from the row (i.e. the gcp-chirp3 SRT).

Requires:
    GOOGLE_CLOUD_PROJECT  — GCP project ID
    GEMINI_API_KEY        — for the correction LLM
    YTD_GCS_BUCKET_NAME   — recommended for GCP STT temp storage

Run with:
    uv run --extra all python scripts/regression/regression_scc_stt.py

To force a specific correction model:
    SCC_MODEL=gemini-3-flash-preview uv run --extra all python \
        scripts/regression/regression_scc_stt.py
"""

import os
import re
import sys

import polars as pl
from regression_core import (
    ARTIFACTS_DIR,
    check_cloud_stack,
    clear_artifacts,
    get_default_model,
    run_regression,
)

STT_MODEL = "gcp-chirp3"
SRT_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}")


def verify(csv_path: str, correction_model: str) -> None:
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    df = pl.read_csv(csv_path)

    # Check the STT SRT column was populated
    stt_srt_col = f"SRT File {STT_MODEL}"
    if stt_srt_col not in df.columns:
        print(f"ERROR: STT SRT column not found: {stt_srt_col!r}")
        print(f"Available columns: {df.columns}")
        sys.exit(1)
    print(f"  STT SRT column present: {stt_srt_col!r}")

    # Check the correction output column
    expected_col = f"Suggested Corrected Captions File ({correction_model})"
    if expected_col not in df.columns:
        print(f"ERROR: Expected column not found: {expected_col!r}")
        print(f"Available columns: {df.columns}")
        sys.exit(1)
    print(f"  Correction column present: {expected_col!r}")

    for val in df[expected_col]:
        if val and isinstance(val, str):
            if not os.path.exists(val):
                print(f"  ERROR: File not found: {val}")
                sys.exit(1)
            content = open(val).read()
            if not SRT_TIMESTAMP_RE.search(content) and content.strip() != "NO_CHANGES":
                print(f"  ERROR: File does not look like a valid SRT: {val}")
                print(f"  Content preview: {content[:200]!r}")
                sys.exit(1)
            print(f"  OK  {val}")

    print("Verification PASSED.")


def main() -> None:
    print(
        f"=== Regression: --suggest-corrected-captions ({STT_MODEL} STT source) ===\n"
    )

    if "GOOGLE_CLOUD_PROJECT" not in os.environ:
        print("ERROR: GOOGLE_CLOUD_PROJECT is required for GCP Chirp3 STT.")
        sys.exit(1)

    clear_artifacts()

    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Check your environment variables.")
        sys.exit(1)

    correction_model = os.environ.get("SCC_MODEL") or get_default_model(
        available_models
    )
    print(f"STT model      : {STT_MODEL}")
    print(f"Correction model: {correction_model}")

    out = os.path.join(ARTIFACTS_DIR, "scc-stt.csv")

    # Run STT transcription and caption correction in one pass.
    # No source suffix on -scc — the correction step automatically uses
    # the most recent AI-generated SRT (i.e. the gcp-chirp3 SRT).
    run_regression(
        model=None,
        transcript_model=STT_MODEL,
        infographic_model=None,
        tts_model=None,
        no_youtube_summary=True,
        output_target=out,
        suggest_corrected_captions=correction_model,
    )

    verify(csv_path=out, correction_model=correction_model)

    print("\n=== SUCCESS ===")


if __name__ == "__main__":
    main()
