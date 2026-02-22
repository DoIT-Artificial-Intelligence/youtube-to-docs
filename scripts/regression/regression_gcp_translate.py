"""
Regression test for the --translate gcp-translate feature.

Generates all LLM outputs in English using the YouTube transcript, then
translates the transcript and all outputs (summary, Q&A, tags, one-sentence
summary) using the Google Cloud Translation API.

Run with:
    uv run --extra all python scripts/regression/regression_gcp_translate.py

To test with a different language, set TRANSLATE_LANG:
    TRANSLATE_LANG=fr uv run --extra all python \
        scripts/regression/regression_gcp_translate.py
"""

import os

from regression_core import (
    ARTIFACTS_DIR,
    clear_artifacts,
    run_regression,
    verify_output,
)


def main():
    target_lang = os.environ.get("TRANSLATE_LANG", "es")
    translate_arg = f"gcp-translate-{target_lang}"

    print(
        f"=== YouTube-to-Docs Regression: --translate gcp-translate ({target_lang}) ==="
    )

    # 1. Clear Artifacts
    clear_artifacts()

    # 2. Use Gemini Flash for summarization
    selected_model = "gemini-3-flash-preview"
    print(f"\nUsing summarization model: {selected_model}")
    print(f"Translate arg: {translate_arg}")

    output_target = os.path.join(ARTIFACTS_DIR, "youtube-docs.csv")

    # 4. Run: YouTube transcript, English outputs + translate via GCP Translate
    print(
        f"\n--- YouTube transcript, English outputs"
        f" + gcp-translate to {target_lang} ---"
    )
    run_regression(
        model=selected_model,
        transcript_model=None,  # use YouTube transcript (no audio download)
        infographic_model=None,
        tts_model=None,
        translate=translate_arg,
        output_target=output_target,
    )

    verify_output(
        model=selected_model,
        transcript_model=None,
        infographic_model=None,
        tts_model=None,
        translate=translate_arg,
        output_target=output_target,
    )

    print(
        f"\n=== SUCCESS: --translate gcp-translate ({target_lang})"
        " Regression Passed ==="
    )


if __name__ == "__main__":
    main()
