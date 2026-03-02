"""
Run with:
uv run python scripts/regression/regression_post_process.py
"""

import os
import sys

# Add the current directory to sys.path so we can find regression_core
sys.path.append(os.path.join(os.getcwd(), "scripts/regression"))

from regression_core import (
    clear_artifacts,
    run_regression,
    verify_output,
)


def main():
    print("=== YouTube-to-Docs Regression: Post-Processing Flag ===")

    # 1. Clear Artifacts
    clear_artifacts()

    # 2. Define Parameters
    # We use the default YouTube transcript (no -t flag needed)
    # and no model (-m None) to keep it simple and test just the post-processing.
    post_process_json = '{"word count": ["apple", "banana"]}'

    # 3. Run Regression
    run_regression(
        model=None,
        transcript_model=None,  # defaults to youtube
        infographic_model=None,
        tts_model=None,
        post_process=post_process_json,
        verbose=True,
    )

    # 4. Verify Output
    verify_output(
        model=None,
        transcript_model=None,
        infographic_model=None,
        tts_model=None,
        post_process=post_process_json,
        verbose=True,
    )

    print("\n=== SUCCESS: Post-Processing Regression Passed ===")


if __name__ == "__main__":
    main()
