# Regression Scripts

This directory contains regression tests for the `youtube-to-docs` tool.

## Core Logic

- **[regression_core.py](regression_core.py)**: Contains shared logic for clearing artifacts, checking cloud model availability, running the main CLI, and verifying output consistency (columns and files).

## Test Cases

### Core / General

- **[regression_en_full.py](regression_en_full.py)**: A full run in English featuring transcription, summarization, infographics, and text-to-speech. Includes secondary "from youtube" processing for summaries, QA, and speakers.
- **[regression_es_no_yt_summary.py](regression_es_no_yt_summary.py)**: A Spanish run with no secondary YouTube summary processing (`-nys` flag).
- **[regression_two_vids.py](regression_two_vids.py)**: Processes two videos (`B0x2I_doX9o,Cu27fBy-kHQ`) with `gemini-3-flash-preview` and no YouTube summary (`-nys`).
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

- **[regression_gcp_tts.py](regression_gcp_tts.py)**: Tests GCP Cloud Text-to-Speech (`gcp-chirp3`) with Gemini Pro summarization (`gemini-3.1-pro-preview`). Requires `GEMINI_API_KEY`.
- **[regression_aws_polly.py](regression_aws_polly.py)**: Tests AWS Polly TTS (`aws-polly`) with AWS Bedrock summarization (`bedrock-nova-2-lite-v1`). Requires AWS credentials.

### Storage

- **[regression_workspace.py](regression_workspace.py)**: Stores results in Google Drive (folder `youtube-to-docs-test-drive`).
- **[regression_workspace_es.py](regression_workspace_es.py)**: Spanish run stored in Google Drive using Gemini Pro models.
- **[regression_sharepoint.py](regression_sharepoint.py)**: Stores results in SharePoint/OneDrive. Uses `foundry-gpt-5-mini` for summarization.

### Translation (AWS / GCP)

- **[regression_aws_translate.py](regression_aws_translate.py)**: Tests AWS Translate integration.
- **[regression_gcp_translate.py](regression_gcp_translate.py)**: Tests Google Cloud Translation API integration.

### Suggested Corrected Captions (`-scc`)

These scripts test WCAG 2.1 Level AA caption correction per Section 508 guidance.
All three accept a `SCC_MODEL` env var to override the correction model (default: best available from `check_cloud_stack`).

- **[regression_scc_youtube.py](regression_scc_youtube.py)**: Fetches the YouTube SRT and corrects it with an LLM. No speaker extraction — no `[Name]` labels expected. Requires `GEMINI_API_KEY` (or another supported model's credentials).

  ```bash
  uv run --extra gcp python scripts/regression/regression_scc_youtube.py
  ```

- **[regression_scc_youtube_speakers.py](regression_scc_youtube_speakers.py)**: Fetches the YouTube SRT, runs speaker extraction, then corrects the SRT. The LLM has speaker names available and should insert `[Name]` labels on speaker changes.

  ```bash
  uv run --extra gcp python scripts/regression/regression_scc_youtube_speakers.py
  ```

- **[regression_scc_stt.py](regression_scc_stt.py)**: Downloads audio, transcribes with GCP Chirp3 (`-t gcp-chirp3`), then corrects the resulting STT SRT in the same run. The correction step uses the default (no source suffix) to automatically pick up the `gcp-chirp3` SRT. Requires `GOOGLE_CLOUD_PROJECT` and `GEMINI_API_KEY`.

  ```bash
  uv run --extra all python scripts/regression/regression_scc_stt.py
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

# Two videos (gcp model)
uv run --extra gcp python scripts/regression/regression_two_vids.py

# Two videos verbose
uv run --extra gcp python scripts/regression/regression_two_vids_verbose.py

# GCP TTS (Chirp3)
uv run --extra gcp --extra aws python scripts/regression/regression_gcp_tts.py

# AWS Polly TTS
uv run --extra aws python scripts/regression/regression_aws_polly.py

# Caption correction — YouTube SRT
uv run --extra gcp python scripts/regression/regression_scc_youtube.py

# Caption correction — YouTube SRT + speaker extraction
uv run --extra gcp python scripts/regression/regression_scc_youtube_speakers.py

# Caption correction — GCP Chirp3 STT SRT
uv run --extra all python scripts/regression/regression_scc_stt.py
```
