"""
Regression test for --translate combined with --tts.

Generates an audio summary in English and a second audio summary in the target
language, verifying both audio files are present in the output.

Run with:
    uv run --extra gcp python scripts/regression/regression_translate_tts.py

To test with a different language, set TRANSLATE_LANG:
    TRANSLATE_LANG=fr uv run --extra gcp python \
        scripts/regression/regression_translate_tts.py
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

TTS_MODEL = "gemini-2.5-flash-preview-tts-Kore"


def main():
    target_lang = os.environ.get("TRANSLATE_LANG", "es")

    print(f"=== YouTube-to-Docs Regression: --translate + --tts ({target_lang}) ===")

    # 1. Clear Artifacts
    clear_artifacts()

    # 2. Check Cloud Stack
    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Please check your environment variables.")
        sys.exit(1)

    # 3. Select Model
    selected_model = get_default_model(available_models)
    print(f"\nUsing model:    {selected_model}")
    translate_arg = f"{selected_model}-{target_lang}"
    print(f"Translate arg:  {translate_arg}")
    print(f"TTS model:      {TTS_MODEL}")

    output_target = os.path.join(ARTIFACTS_DIR, "youtube-docs.csv")

    # 4. Run: YouTube transcript + English TTS + translated TTS
    print(f"\n--- YouTube transcript, TTS in English + {target_lang} ---")
    run_regression(
        model=selected_model,
        transcript_model=None,  # use YouTube transcript (no audio download)
        infographic_model=None,
        tts_model=TTS_MODEL,
        translate=translate_arg,
        output_target=output_target,
    )

    # 5. Verify both English and translated audio columns exist
    verify_output(
        model=selected_model,
        transcript_model=None,
        infographic_model=None,
        tts_model=TTS_MODEL,
        translate=translate_arg,
        output_target=output_target,
    )

    print(f"\n=== SUCCESS: --translate + --tts ({target_lang}) Regression Passed ===")


if __name__ == "__main__":
    main()
