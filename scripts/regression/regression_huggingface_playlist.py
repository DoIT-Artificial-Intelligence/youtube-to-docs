"""
End-to-end regression for Hugging Face dataset storage (`-o hf`) — playlist.

Processes the "Code for America Summit 2026 Recap" YouTube playlist
(`PL65XgbSILalUtRMlH4gG8EMY9_yUlSi0_`) and stores the artifacts in a same-named
Hugging Face dataset (the name is slugified and namespaced under the
authenticated user). This mirrors the canonical use case: a playlist analysed
into a dataset.

Requires the `HF_TOKEN` environment variable (a write token) and the
`huggingface` extra. Note: this processes every video in the playlist, so it is
significantly more expensive than the single-video regression.

Env overrides:
  - HF_DATASET: target dataset name or `namespace/name` repo id.
  - HF_VIDEO:   YouTube target (playlist ID, video ID, or comma-separated IDs).

Run with:
uv run --extra huggingface --extra gcp \\
    python scripts/regression/regression_huggingface_playlist.py
"""

from regression_huggingface import run_hf_regression

# The "Code for America Summit 2026 Recap" playlist.
PLAYLIST_ID = "PL65XgbSILalUtRMlH4gG8EMY9_yUlSi0_"


def main():
    print("=== YouTube-to-Docs Regression: Hugging Face Storage (Playlist) ===")
    run_hf_regression("Code for America Summit 2026 Recap", default_video=PLAYLIST_ID)
    print("\n=== SUCCESS: Hugging Face Storage Regression (Playlist) Passed ===")


if __name__ == "__main__":
    main()
