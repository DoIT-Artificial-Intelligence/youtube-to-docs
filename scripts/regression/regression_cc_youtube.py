"""
Regression test for --corrected-captions (YouTube source).

Corrects the YouTube SRT (no transcription) with `-cc {model}-youtube`.

Requires:
    GEMINI_API_KEY (or another supported model's credentials)

Run with:
    uv run --extra all python scripts/regression/regression_cc_youtube.py

To force a specific correction model:
    CC_MODEL=gemini-3.5-flash-lite uv run --extra all python \
        scripts/regression/regression_cc_youtube.py
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
    print("=== Regression: --corrected-captions (YouTube source) ===\n")

    clear_artifacts()

    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Check your environment variables.")
        sys.exit(1)
    model = os.environ.get("CC_MODEL") or get_default_model(available_models)
    print(f"Correction model: {model}")

    out = os.path.join(ARTIFACTS_DIR, "regression_cc_youtube.csv")
    run_regression(
        model=None,
        transcript_model=None,
        infographic_model=None,
        tts_model=None,
        no_youtube_summary=True,
        output_target=out,
        corrected_captions=f"{model}-youtube",
    )
    verify_corrected_captions(out, f"Corrected Captions File {model} from youtube")
    print("\n=== SUCCESS ===")


if __name__ == "__main__":
    main()
