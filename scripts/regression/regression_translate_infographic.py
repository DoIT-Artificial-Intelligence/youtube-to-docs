"""
Regression test for --translate combined with infographic generation.

Generates an infographic from the English summary and a second infographic from
the translated summary, verifying both are present in the output.

Run with:
    uv run --extra gcp python scripts/regression/regression_translate_infographic.py

To test with a different language, set TRANSLATE_LANG:
    TRANSLATE_LANG=fr uv run --extra gcp python \
        scripts/regression/regression_translate_infographic.py
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

INFOGRAPHIC_MODEL = "gemini-2.5-flash-image"


def main():
    target_lang = os.environ.get("TRANSLATE_LANG", "es")

    print(
        f"=== YouTube-to-Docs Regression: --translate + infographic ({target_lang}) ==="
    )

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
    translate_arg = f"{selected_model}-{target_lang}"
    print(f"Translate arg:    {translate_arg}")
    print(f"Infographic model: {INFOGRAPHIC_MODEL}")

    output_target = os.path.join(ARTIFACTS_DIR, "youtube-docs.csv")

    # 4. Run: YouTube transcript + English infographic + translated infographic
    print(f"\n--- YouTube transcript, infographic in English + {target_lang} ---")
    run_regression(
        model=selected_model,
        transcript_model=None,  # use YouTube transcript (no audio download)
        infographic_model=INFOGRAPHIC_MODEL,
        tts_model=None,
        translate=translate_arg,
        output_target=output_target,
    )

    # 5. Verify both English and translated infographic columns exist
    verify_output(
        model=selected_model,
        transcript_model=None,
        infographic_model=INFOGRAPHIC_MODEL,
        tts_model=None,
        translate=translate_arg,
        output_target=output_target,
    )

    print(
        f"\n=== SUCCESS: --translate + infographic ({target_lang})"
        " Regression Passed ==="
    )


if __name__ == "__main__":
    main()
