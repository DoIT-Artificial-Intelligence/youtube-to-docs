"""
Run with:
uv run python scripts/regression/regression_gemma.py
"""

import os
import sys

from regression_core import (
    clear_artifacts,
    run_regression,
    verify_output,
)

GEMMA_MODELS = [
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
]


def check_gemma_prerequisites() -> bool:
    """Checks that GEMINI_API_KEY is set (Gemma uses the Google GenAI client)."""
    if "GEMINI_API_KEY" not in os.environ:
        print("SKIPPED: GEMINI_API_KEY not set (required for Gemma via Google GenAI).")
        return False
    return True


def test_vertex_gemma_raises():
    """Verifies that vertex-gemma-* raises NotImplementedError."""
    print("\n--- Testing vertex-gemma-* raises NotImplementedError ---")
    sys.path.insert(0, os.getcwd())
    from youtube_to_docs.llms import _query_llm

    for model in GEMMA_MODELS:
        vertex_model = f"vertex-{model}"
        try:
            _query_llm(vertex_model, "test")
            print(f"FAILED: {vertex_model} did not raise NotImplementedError")
            sys.exit(1)
        except NotImplementedError as e:
            print(f"OK: {vertex_model} -> NotImplementedError: {e}")
        except Exception as e:
            # Any other error before reaching the NotImplementedError check is a failure
            print(f"FAILED: {vertex_model} raised unexpected {type(e).__name__}: {e}")
            sys.exit(1)

    print("vertex-gemma-* NotImplementedError check PASSED.")


def main():
    print("=== YouTube-to-Docs Regression: Gemma Models Test ===")

    # 1. Verify vertex-gemma raises NotImplementedError (no API key needed)
    test_vertex_gemma_raises()

    # 2. Check prerequisites for live API calls
    if not check_gemma_prerequisites():
        sys.exit(0)

    for model in GEMMA_MODELS:
        print(f"\n--- Testing model: {model} ---")

        # 3. Clear Artifacts
        clear_artifacts()

        # 4. Run Regression (YouTube transcript, summary only)
        run_regression(
            model=model,
            transcript_model=None,
            infographic_model=None,
            tts_model=None,
            no_youtube_summary=False,
        )

        # 5. Verify Output
        verify_output(
            model=model,
            transcript_model=None,
            infographic_model=None,
            tts_model=None,
            no_youtube_summary=False,
        )

    print("\n=== SUCCESS: Gemma Regression Passed ===")


if __name__ == "__main__":
    main()
