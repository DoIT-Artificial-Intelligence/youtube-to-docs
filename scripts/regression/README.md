# Regression Scripts

This directory contains regression tests for the `youtube-to-docs` tool.

## Core Logic

- **[regression_core.py](regression_core.py)**: Contains shared logic for clearing artifacts, checking cloud model availability, running the main CLI, and verifying output consistency (columns and files).

## Test Cases

### Core / General

- **[regression_en_full.py](regression_en_full.py)**: A full run in English featuring transcription, summarization, infographics, and text-to-speech. Includes secondary "from youtube" processing for summaries, QA, and speakers.
- **[regression_es_no_yt_summary.py](regression_es_no_yt_summary.py)**: A Spanish run with no secondary YouTube summary processing (`-nys` flag).
- **[regression_two_vids.py](regression_two_vids.py)**: Processes two videos (`B0x2I_doX9o,Cu27fBy-kHQ`) with `gemini-3.5-flash-lite` and no YouTube summary (`-nys`).
- **[regression_two_vids_verbose.py](regression_two_vids_verbose.py)**: Same as above but with verbose output enabled.

### Translation

- **[regression_translate.py](regression_translate.py)**: Generates English outputs then translates to a target language (default `es`). Accepts `TRANSLATE_LANG` env var.
- **[regression_translate_infographic.py](regression_translate_infographic.py)**: Translation with infographic generation in both languages.
- **[regression_translate_tts.py](regression_translate_tts.py)**: Translation with TTS audio in both languages.

### Speech-to-Text (STT)

- **[regression_gcp_stt.py](regression_gcp_stt.py)**: Tests GCP Speech-to-Text V2 (`gcp-chirp3`) integration. Requires `GOOGLE_CLOUD_PROJECT` and optional `YTD_GCS_BUCKET_NAME` env vars.
- **[regression_gcp_stt_long.py](regression_gcp_stt_long.py)**: Same as above but for a longer video.
- **[regression_aws_transcribe.py](regression_aws_transcribe.py)**: Tests AWS Transcribe STT integration.

### Text-to-Speech (TTS)

- **[regression_gcp_tts.py](regression_gcp_tts.py)**: Tests GCP Cloud Text-to-Speech (`gcp-chirp3`) with Gemini Pro summarization (`gemini-4-argon`). Requires `GEMINI_API_KEY`.
- **[regression_aws_polly.py](regression_aws_polly.py)**: Tests AWS Polly TTS (`aws-polly`) with AWS Bedrock summarization (`bedrock-nova-2-lite-v1`). Requires AWS credentials.

### Storage

- **[regression_workspace.py](regression_workspace.py)**: Stores results in Google Drive (folder `youtube-to-docs-test-drive`).
- **[regression_workspace_es.py](regression_workspace_es.py)**: Spanish run stored in Google Drive using Gemini Pro models.
- **[regression_sharepoint.py](regression_sharepoint.py)**: Stores results in SharePoint/OneDrive. Uses `foundry-gpt-6-luna` for summarization.
- **[regression_huggingface.py](regression_huggingface.py)**: Processes a single video and stores results in a Hugging Face dataset (`-o hf`). The cheap smoke test for HF storage. Requires the `HF_TOKEN` env var (write token). Override the dataset with `HF_DATASET` or the YouTube target with `HF_VIDEO`.
- **[regression_huggingface_playlist.py](regression_huggingface_playlist.py)**: Processes the "Code for America Summit 2026 Recap" YouTube playlist (`PL65XgbSILalUtRMlH4gG8EMY9_yUlSi0_`) and stores results in a same-named Hugging Face dataset (`-o hf`). Exercises the full playlist use case (more expensive — every video is processed). Same env overrides as above.

### Translation (AWS / GCP)

- **[regression_aws_translate.py](regression_aws_translate.py)**: Tests AWS Translate integration.
- **[regression_gcp_translate.py](regression_gcp_translate.py)**: Tests Google Cloud Translation API integration.

### Corrected Captions (`-cc`)

These scripts test the complete WCAG 2.1 Level AA / Section 508 corrected SRT. Each checks the output has named speakers, no `[Speaker N]` labels, lines of at most 42 characters and no overlapping cues. The YouTube and Chirp 3 scripts accept a `CC_MODEL` env var to override the correction model (default: best available from `check_cloud_stack`).

- **[regression_cc_default.py](regression_cc_default.py)**: Bare `-cc`: transcribes with `gemini-3.5-transcribe`, reads on-screen context from the video and corrects with `gemini-4-argon`. Requires `GEMINI_API_KEY`.

  ```bash
  uv run --extra all python scripts/regression/regression_cc_default.py
  ```

- **[regression_cc_youtube.py](regression_cc_youtube.py)**: Corrects the YouTube SRT (`-cc {model}-youtube`, no transcription).

  ```bash
  uv run --extra all python scripts/regression/regression_cc_youtube.py
  ```

- **[regression_cc_stt.py](regression_cc_stt.py)**: Transcribes with GCP Chirp 3 (`-t gcp-chirp3`) and corrects that SRT in the same run. Requires `GOOGLE_CLOUD_PROJECT` and `GEMINI_API_KEY`.

  ```bash
  uv run --extra all python scripts/regression/regression_cc_stt.py
  ```

## Usage

To run a specific test case from the project root:

```bash
# Full English run with transcript, infographic, and TTS
uv run --extra audio --extra video --extra gcp python scripts/regression/regression_en_full.py

# Spanish run with gcp model
uv run --extra gcp python scripts/regression/regression_es_no_yt_summary.py

# Google Drive storage with gcp model
uv run --extra workspace --extra gcp python scripts/regression/regression_workspace.py

# Google Drive storage (Spanish) with gcp model
uv run --extra workspace --extra gcp python scripts/regression/regression_workspace_es.py

# SharePoint storage with azure model
uv run --extra m365 --extra azure python scripts/regression/regression_sharepoint.py

# Hugging Face dataset storage — single video (requires HF_TOKEN)
uv run --extra huggingface --extra gcp python scripts/regression/regression_huggingface.py

# Hugging Face dataset storage — playlist (requires HF_TOKEN)
uv run --extra huggingface --extra gcp python scripts/regression/regression_huggingface_playlist.py

# Two videos (gcp model)
uv run --extra gcp python scripts/regression/regression_two_vids.py

# Two videos verbose
uv run --extra gcp python scripts/regression/regression_two_vids_verbose.py

# GCP TTS (Chirp3)
uv run --extra gcp --extra aws python scripts/regression/regression_gcp_tts.py

# AWS Polly TTS
uv run --extra aws python scripts/regression/regression_aws_polly.py

# Corrected captions — default (gemini-3.5-transcribe + gemini-4-argon)
uv run --extra all python scripts/regression/regression_cc_default.py

# Corrected captions — YouTube SRT
uv run --extra all python scripts/regression/regression_cc_youtube.py

# Corrected captions — GCP Chirp 3 STT SRT
uv run --extra all python scripts/regression/regression_cc_stt.py
```
