"""
Run with:
uv run --extra gcp python scripts/regression/regression_two_vids_verbose.py
"""

import regression_core

# Override the VIDEO_ID in the core module
regression_core.VIDEO_ID = "B0x2I_doX9o,Cu27fBy-kHQ"


def main():
    regression_core.clear_artifacts()

    model = "bedrock-nova-2-lite-v1"

    regression_core.run_regression(
        model=model,
        transcript_model=None,
        infographic_model=None,
        tts_model=None,
        no_youtube_summary=True,
        verbose=True,
    )

    regression_core.verify_output(
        model=model,
        transcript_model=None,
        infographic_model=None,
        tts_model=None,
        no_youtube_summary=True,
    )


if __name__ == "__main__":
    main()
