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

## Usage

To run a specific test case from the project root:

```bash
uv run python scripts/regression/regression_en_full.py
```

or

```bash
uv run python scripts/regression/regression_es_no_yt_summary.py
```

or

```bash
uv run python scripts/regression/regression_workspace.py
```

or

```bash
uv run python scripts/regression/regression_workspace_es.py
```

or

```bash
uv run python scripts/regression/regression_sharepoint.py
```
