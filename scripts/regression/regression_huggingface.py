"""
End-to-end regression for Hugging Face dataset storage (`-o hf`) — single video.

Processes a single YouTube video and stores the artifacts in a Hugging Face
dataset (the name is slugified and namespaced under the authenticated user).
This is the cheap smoke test; for the full playlist use case see
regression_huggingface_playlist.py.

Requires the `HF_TOKEN` environment variable (a write token) and the
`huggingface` extra.

Env overrides:
  - HF_DATASET: target dataset name or `namespace/name` repo id.
  - HF_VIDEO:   YouTube target (video ID, playlist ID, or comma-separated IDs).

Run with:
uv run --extra huggingface --extra gcp \\
    python scripts/regression/regression_huggingface.py
"""

import os
import sys

import regression_core
from regression_core import (
    check_cloud_stack,
    get_default_model,
    run_regression,
    verify_output,
)


def run_hf_regression(default_dataset, default_video=None):
    """Run the HF storage regression for the given dataset/video target.

    ``HF_DATASET`` / ``HF_VIDEO`` env vars override ``default_dataset`` and
    ``default_video`` respectively. ``default_video`` of ``None`` keeps the
    core module's default single video.
    """
    # 1. Require an HF token up front so we fail fast with a clear message.
    if not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN environment variable is required for this regression.")
        sys.exit(1)

    dataset = os.environ.get("HF_DATASET", default_dataset)
    video = os.environ.get("HF_VIDEO", default_video)
    if video:
        regression_core.VIDEO_ID = video
    print(f"Using Hugging Face dataset: {dataset}")
    print(f"Processing YouTube target: {regression_core.VIDEO_ID}")

    # 2. Check Cloud Stack
    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Please check your environment variables.")
        sys.exit(1)

    # 3. Select Model
    selected_model = get_default_model(available_models)
    print(f"\nUsing model: {selected_model}")

    # 4. Run Regression with Hugging Face Target
    run_regression(
        selected_model,  # model
        None,  # transcript_model
        None,  # infographic_model
        None,  # tts_model
        output_target="hf",
        hugging_face_dataset=dataset,
        all_gemini_arg=None,
    )

    # 5. Verify Output from Hugging Face
    verify_output(
        selected_model,
        None,
        None,
        None,
        output_target="hf",
        hugging_face_dataset=dataset,
        all_gemini_arg=None,
    )


def main():
    print("=== YouTube-to-Docs Regression: Hugging Face Dataset Storage (Video) ===")
    run_hf_regression("youtube-to-docs-hf-test")
    print("\n=== SUCCESS: Hugging Face Dataset Storage Regression (Video) Passed ===")


if __name__ == "__main__":
    main()
