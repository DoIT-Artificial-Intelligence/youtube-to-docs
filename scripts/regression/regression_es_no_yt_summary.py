"""
Kitchen-sink regression: AI STT + --translate + infographic (EN + translated) +
TTS (EN + translated) + two videos (English infographic+TTS and translated
infographic+TTS).

Run with:
    uv run --extra gcp python scripts/regression/regression_es_no_yt_summary.py

To test with a different language, set TRANSLATE_LANG:
    TRANSLATE_LANG=fr uv run --extra gcp python \
        scripts/regression/regression_es_no_yt_summary.py
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

TRANSCRIPT_MODEL = "gemini-3-flash-preview"
INFOGRAPHIC_MODEL = "gemini-2.5-flash-image"
TTS_MODEL = "gemini-2.5-flash-preview-tts-Kore"


def main():
    target_lang = os.environ.get("TRANSLATE_LANG", "es")

    print(
        f"=== YouTube-to-Docs Regression: Kitchen Sink "
        f"(AI STT + Translate + Infographic + TTS + Video) [{target_lang}] ==="
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
    translate_arg = f"{selected_model}-{target_lang}"

    print(f"\nUsing model:       {selected_model}")
    print(f"Transcript model:  {TRANSCRIPT_MODEL}")
    print(f"Translate arg:     {translate_arg}")
    print(f"Infographic model: {INFOGRAPHIC_MODEL}")
    print(f"TTS model:         {TTS_MODEL}")

    output_target = os.path.join(ARTIFACTS_DIR, "youtube-docs.csv")

    # 4. Run: AI STT + translate + infographic (EN + translated) + TTS + video
    print(
        f"\n--- AI STT, translate to {target_lang}, infographic + TTS in both "
        f"languages, combine into videos ---"
    )
    run_regression(
        model=selected_model,
        transcript_model=TRANSCRIPT_MODEL,
        infographic_model=INFOGRAPHIC_MODEL,
        tts_model=TTS_MODEL,
        translate=translate_arg,
        no_youtube_summary=True,
        output_target=output_target,
        combine_info_audio=True,
    )

    # 5. Verify all expected columns and files
    verify_output(
        model=selected_model,
        transcript_model=TRANSCRIPT_MODEL,
        infographic_model=INFOGRAPHIC_MODEL,
        tts_model=TTS_MODEL,
        translate=translate_arg,
        no_youtube_summary=True,
        output_target=output_target,
        combine_info_audio=True,
    )

    print(f"\n=== SUCCESS: Kitchen Sink Regression ({target_lang}) Passed ===")


if __name__ == "__main__":
    main()
