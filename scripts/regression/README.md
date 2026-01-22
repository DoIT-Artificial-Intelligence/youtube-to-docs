# Regression Scripts

This directory contains regression tests for the `youtube-to-docs` tool.

## Core Logic

- **[regression_core.py](regression_core.py)**: Contains shared logic for clearing artifacts, checking cloud model availability, running the main CLI, and verifying output consistency (columns and files).

## Test Cases

- **[regression_en_full.py](regression_en_full.py)**: A full run in English featuring transcription, summarization, infographics, and text-to-speech. Includes secondary "from youtube" processing for summaries, QA, and speakers.
- **[regression_es_no_yt_summary.py](regression_es_no_yt_summary.py)**: A Spanish run with no secondary YouTube summary processing (`-nys` flag).
- **[regression_workspace.py](regression_workspace.py)**: A run that stores results in Google Drive (specifically a folder named `youtube-to-docs-test-drive`).
- **[regression_workspace_es.py](regression_workspace_es.py)**: A Spanish run that stores results in Google Drive using Gemini Pro models.
- **[regression_sharepoint.py](regression_sharepoint.py)**: A run that stores results in SharePoint/OneDrive (specifically a folder named `youtube-to-docs-artifacts`). Uses `foundry-gpt-5-mini` for summarization.
- **[regression_two_vids.py](regression_two_vids.py)**: Processes two videos (`B0x2I_doX9o,Cu27fBy-kHQ`) with `gemini-3-flash-preview` and no YouTube summary (`-nys`).
- **[regression_two_vids_verbose.py](regression_two_vids_verbose.py)**: Same as above but with verbose output enabled.
- **[regression_gcp_stt.py](regression_gcp_stt.py)**: Tests GCP Speech-to-Text V2 (`gcp-chirp3`) integration. Requires `GOOGLE_CLOUD_PROJECT` and optional `YTD_GCS_BUCKET_NAME` env vars.

## Usage

To run a specific test case from the project root:

```bash
# Full English run with transcript, infographic, and TTS
uv run --extra audio --extra video --extra gcp python scripts/regression/regression_en_full.py
```

or

```bash
# Spanish run with gcp model
uv run --extra gcp python scripts/regression/regression_es_no_yt_summary.py
```

or

```bash
# Google Drive storage with gcp model
uv run --extra workspace --extra gcp python scripts/regression/regression_workspace.py
```

or

```bash
# Google Drive storage (Spanish) with gcp model
uv run --extra workspace --extra gcp python scripts/regression/regression_workspace_es.py
```

or

```bash
# SharePoint storage with azure model
uv run --extra m365 --extra azure python scripts/regression/regression_sharepoint.py
```

or

```bash
# gcp model tests
uv run --extra gcp python scripts/regression/regression_two_vids.py
```

or

```bash
# gcp model tests (verbose)
uv run --extra gcp python scripts/regression/regression_two_vids_verbose.py
```

or

```bash
# GCP STT (Chirp) tests
uv run --extra gcp python scripts/regression/regression_gcp_stt.py
```
