"""
Run with:
uv run python scripts/regression/regression_nemotron_llm.py

Tests `-m nvidia-nemotron-3.5-lightning`:
- Uses a self-hosted vLLM server when `NEMOTRON_LLM_URL` is set or a compatible
  local GPU is detected (see scripts/nemotron_llm/README.md).
- Automatically falls back to build.nvidia.com (`NVIDIA_API_KEY`) when no
  compatible local GPU is detected.
"""

import os
import sys

from regression_core import (
    clear_artifacts,
    run_regression,
    verify_output,
)

from youtube_to_docs.nemotron_llm import (
    NEMOTRON_LIGHTNING_MODEL,
    check_nemotron_llm_server,
    use_self_hosted_nemotron_llm,
)


def main():
    print("=== YouTube-to-Docs Regression: NVIDIA Nemotron LLM Test ===")

    model = NEMOTRON_LIGHTNING_MODEL
    if use_self_hosted_nemotron_llm(model):
        server_ok, server_msg = check_nemotron_llm_server(model)
        if not server_ok:
            print(f"SKIPPED: {server_msg}")
            sys.exit(0)
        print(f"Backend: self-hosted vLLM ({server_msg})")
    else:
        if "NVIDIA_API_KEY" not in os.environ:
            print(
                "SKIPPED: No compatible local GPU detected and NVIDIA_API_KEY "
                "is not set."
            )
            sys.exit(0)
        print("Backend: build.nvidia.com (no compatible local GPU detected)")

    clear_artifacts()

    run_regression(
        model=model,
        transcript_model=None,
        infographic_model=None,
        tts_model=None,
        no_youtube_summary=False,
    )

    verify_output(
        model=model,
        transcript_model=None,
        infographic_model=None,
        tts_model=None,
        no_youtube_summary=False,
    )

    print("\n=== SUCCESS: NVIDIA Nemotron LLM Regression Passed ===")


if __name__ == "__main__":
    main()
