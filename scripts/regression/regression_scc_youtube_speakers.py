"""
Regression test for --suggest-corrected-captions with speaker extraction.

Fetches the YouTube transcript/SRT, runs speaker extraction with an LLM, then
asks the same LLM to suggest WCAG 2.1 Level AA compliant corrections. Because
speaker extraction has been run, the LLM should insert [Name] labels at speaker
changes.

Run with:
    uv run python scripts/regression/regression_scc_youtube_speakers.py

To force a specific model:
    SCC_MODEL=gemini-3-flash-preview uv run python \
        scripts/regression/regression_scc_youtube_speakers.py
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

SRT_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}")


def verify(csv_path: str, correction_model: str) -> None:
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    df = pl.read_csv(csv_path)
    expected_col = f"Suggested Corrected Captions File ({correction_model})"

    if expected_col not in df.columns:
        print(f"ERROR: Expected column not found: {expected_col!r}")
        print(f"Available columns: {df.columns}")
        sys.exit(1)

    print(f"  Column present: {expected_col!r}")

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

    # Check that speaker extraction ran
    speakers_col = next(
        (c for c in df.columns if c.startswith("Speakers ") and "File" not in c),
        None,
    )
    if speakers_col:
        print(f"  Speakers column present: {speakers_col!r}")
    else:
        print("  WARNING: No Speakers column found — speaker labels may be absent.")

    print("Verification PASSED.")


def main() -> None:
    print(
        "=== Regression: --suggest-corrected-captions "
        "(YouTube source + speaker extraction) ===\n"
    )

    clear_artifacts()

    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Check your environment variables.")
        sys.exit(1)

    correction_model = os.environ.get("SCC_MODEL") or get_default_model(
        available_models
    )
    print(f"Correction model: {correction_model}")

    out = os.path.join(ARTIFACTS_DIR, "scc-youtube-speakers.csv")

    run_regression(
        model=correction_model,
        transcript_model=None,
        infographic_model=None,
        tts_model=None,
        output_target=out,
        suggest_corrected_captions=f"{correction_model}-youtube",
    )

    verify(csv_path=out, correction_model=correction_model)

    print("\n=== SUCCESS ===")


if __name__ == "__main__":
    main()
