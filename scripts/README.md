# Regression Scripts

This directory contains regression tests for the `youtube-to-docs` tool.

## Core Logic

- **[regression_core.py](file:///c:/Users/ray.bell/Documents/Code/youtube-to-docs/scripts/regression_core.py)**: Contains shared logic for clearing artifacts, checking cloud model availability, running the main CLI, and verifying output consistency (columns and files).

## Test Cases

- **[regression_en_full.py](file:///c:/Users/ray.bell/Documents/Code/youtube-to-docs/scripts/regression_en_full.py)**: A full run in English featuring transcription, summarization, infographics, and text-to-speech. Includes secondary "from youtube" processing for summaries, QA, and speakers.
- **[regression_es_no_yt_summary.py](file:///c:/Users/ray.bell/Documents/Code/youtube-to-docs/scripts/regression_es_no_yt_summary.py)**: A Spanish run with no secondary YouTube summary processing (`-nys` flag).
- **[regression_workspace.py](file:///c:/Users/ray.bell/Documents/Code/youtube-to-docs/scripts/regression_workspace.py)**: A run that stores results in Google Drive (specifically a folder named `youtube-to-docs-test-drive`).

## Usage

To run a specific test case:

```bash
uv run python .\scripts\regression_en_full.py
```

or

```bash
uv run python .\scripts\regression_es_no_yt_summary.py
```

or

```bash
uv run python .\scripts\regression_workspace.py
```