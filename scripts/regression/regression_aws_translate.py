"""
Regression test for the --translate aws-translate feature.

Generates all LLM outputs in English using the YouTube transcript, then
translates the transcript and all outputs (summary, Q&A, tags, one-sentence
summary) using the AWS Translate service.

Run with:
    uv run --extra all python scripts/regression/regression_aws_translate.py

To test with a different language, set TRANSLATE_LANG:
    TRANSLATE_LANG=fr uv run --extra all python \
        scripts/regression/regression_aws_translate.py
"""

import os
import sys

from regression_core import (
    ARTIFACTS_DIR,
    check_cloud_stack,
    clear_artifacts,
    run_regression,
    verify_output,
)


def main():
    target_lang = os.environ.get("TRANSLATE_LANG", "es")
    translate_arg = f"aws-translate-{target_lang}"

    print(
        f"=== YouTube-to-Docs Regression: --translate aws-translate ({target_lang}) ==="
    )

    # 1. Clear Artifacts
    clear_artifacts()

    # 2. Check Cloud Stack (for the summarization model)
    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Please check your environment variables.")
        sys.exit(1)

    # 3. Use Nova Pro for summarization (all-AWS stack)
    selected_model = "bedrock-nova-pro-v1"
    print(f"\nUsing summarization model: {selected_model}")
    print(f"Translate arg: {translate_arg}")

    output_target = os.path.join(ARTIFACTS_DIR, "youtube-docs.csv")

    # 4. Run: YouTube transcript, English outputs + translate via AWS Translate
    print(
        f"\n--- YouTube transcript, English outputs"
        f" + aws-translate to {target_lang} ---"
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
        f"\n=== SUCCESS: --translate aws-translate ({target_lang})"
        " Regression Passed ==="
    )


if __name__ == "__main__":
    main()
