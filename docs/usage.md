# Usage Guide

`youtube-to-docs` is a versatile tool designed to convert YouTube content into structured documentation, including transcripts, summaries, audio, and infographics. It is primarily designed as a Command Line Interface (CLI) tool but can also be used as a Python library.

## Installation

The recommended way to run `youtube-to-docs` is using `uvx`.

### Basic Usage

For basic usage (YouTube data fetching, local CSV output, standard Gemini models):

```bash
uvx youtube-to-docs --help
```

### Optional Features (Extras)

To keep the installation footprint small, many features are optional. You can enable them by installing specific "extras":

| Extra | Description | Dependencies |
| :--- | :--- | :--- |
| `audio` | **Recommended.** Required for downloading audio (for TTS/transcription). | `yt-dlp` |
| `video` | Required for generating video files (combining audio & infographic). | `static-ffmpeg` |
| `workspace` | Required for saving to Google Drive. | `google-api-python-client`, `google-auth-oauthlib` |
| `m365` | Required for saving to SharePoint/OneDrive. | `msal`, `fastexcel`, `xlsxwriter`, `pypandoc` |
| `aws` | AWS Bedrock support. | None |
| `azure` | Required if using Azure OpenAI models. | `openai` |
| `gcp` | Required if using Google Gemini or Vertex AI models. | `google-genai` |
| `all` | Installs all of the above. | All optional dependencies. |

**How to use extras with `uvx`:**

Use the `--with` flag followed by the package name and extras in brackets.

**Example 1: Audio and Video support (Common)**
```bash
uvx --with "youtube-to-docs[audio,video]" youtube-to-docs ...
```

**Example 2: Google Drive + Audio support**
```bash
uvx --with "youtube-to-docs[workspace,audio]" youtube-to-docs ...
```

**Example 3: Everything (Full feature set)**
```bash
uvx --with "youtube-to-docs[all]" youtube-to-docs ...
```

### Installing `uv`

If you don't have `uv` installed, you can install it using the following commands:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Setup

Before running the tool, ensure your environment is correctly configured with the necessary API keys and authentication files.

### 1. Environment Variables

Set the following environment variables based on the AI providers you intend to use.

| Variable | Description | Required For |
| :--- | :--- | :--- |
| `YOUTUBE_DATA_API_KEY` | API key for the YouTube Data API v3. | Fetching video metadata. |
| `GEMINI_API_KEY` | API key for Google Gemini models. | Gemini models (`-m gemini...`). |
| `PROJECT_ID` | Google Cloud Project ID. | GCP Vertex models (`-m vertex...`). |
| `AWS_BEARER_TOKEN_BEDROCK` | AWS Bearer Token. | AWS Bedrock models (`-m bedrock...`). |
| `AZURE_FOUNDRY_ENDPOINT` | Azure Foundry Endpoint URL. | Azure Foundry models (`-m foundry...`). |
| `AZURE_FOUNDRY_API_KEY` | Azure Foundry API Key. | Azure Foundry models (`-m foundry...`). |

### 2. Storage Authentication (Optional)

If you plan to save outputs to Google Drive (`workspace`) or Microsoft SharePoint/OneDrive (`sharepoint`), you need to configure authentication files in your home directory.

#### Google Drive (Workspace)
Create a file at `~/.google_client_secret.json` with your Google Cloud OAuth 2.0 Client Secret JSON.

```json
{
  "installed": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uris": ["http://localhost"]
  }
}
```
*   **First Run**: The tool will open a browser window to authenticate and generate a `~/.token.json` file for future non-interactive use.

#### Microsoft 365 (SharePoint/OneDrive)
Create a file at `~/.azure_client.json` with your Azure App Registration details.

```json
{
    "client_id": "YOUR_CLIENT_ID",
    "authority": "https://login.microsoftonline.com/consumers"
}
```
*   **Authority**: Use `.../consumers` for personal accounts or `.../YOUR_TENANT_ID` for organizational accounts.
*   **First Run**: The tool will attempt to authenticate (silently or interactively) and cache the token in `~/.msal_token_cache.json`.

## Command Line Interface (CLI)

The main command is `youtube-to-docs`.

### Basic Usage

Running the command without arguments processes a default video:

```bash
youtube-to-docs
```

### Arguments

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `video_id` | The YouTube content to process. Can be a **Video ID**, **Playlist ID** (starts with `PL`), **Channel Handle** (starts with `@`), or a **comma-separated list** of Video IDs. | `atmGAHYpf_c` | `youtube-to-docs @mychannel` |
| `-o`, `--outfile` | Path to save the output CSV file. <br> - Local path: `my-data.csv` <br> - Google Workspace: `workspace` or `w` (saves to Drive folder `youtube-to-docs-artifacts`) or a specific Folder ID. <br> - SharePoint/OneDrive: `sharepoint` or `s` (saves to `youtube-to-docs-artifacts`). | `youtube-to-docs-artifacts/youtube-docs.csv` | `-o sharepoint` |
| `-t`, `--transcript` | The transcript source to use. Can be `'youtube'` (default) to fetch existing YouTube transcripts, or an AI model name to perform STT on extracted audio. | `youtube` | `-t gemini-2.0-flash-exp` |
| `-m`, `--model` | The LLM(s) to use for speaker extraction, Q&A generation, and summarization. Supports models from Google (Gemini), Vertex AI, AWS Bedrock, and Azure Foundry. **Can be a comma-separated list.** | `None` | `-m gemini-3-flash-preview,vertex-claude-haiku-4-5@20251001` |
| `--tts` | The TTS model and voice to use for generating audio summaries. Format: `{model}-{voice}`. | `None` | `--tts gemini-2.5-flash-preview-tts-Kore` |
| `-i`, `--infographic`| The image model to use for generating a visual summary. Supports models from Google (Gemini, Imagen), AWS Bedrock (Titan, Nova Canvas), and Azure Foundry. | `None` | `--infographic gemini-2.5-flash-image` |
| `-nys`, `--no-youtube-summary` | If set, skips generating a secondary summary from the YouTube transcript when using an AI model for the primary transcript. | `False` | `--no-youtube-summary` |
| `-l`, `--language` | The target language(s) (e.g. 'es', 'fr', 'en'). Can be a comma-separated list. Default is 'en'. | `en` | `-l es,fr` |
| `-cia`, `--combine-infographic-audio` | Combine the infographic and audio summary into a video file (MP4). Requires both `--tts` and `--infographic` to be effective. | `False` | `--combine-infographic-audio` |
| `--all` | Shortcut to use a specific model suite for everything. Supported: `'gemini-flash'`, `'gemini-pro'`, `'gemini-flash-pro-image'`. Sets models for summary, TTS, and infographic, and enables `--no-youtube-summary`. | `None` | `--all gemini-flash` |
| `--verbose` | Enable verbose output. | `False` | `--verbose` |

### Examples

**1. Summarize the default video using a single model:**
```bash
youtube-to-docs -m gemini-3-flash-preview
```

**2. Generate a transcript using Gemini 2.0 Flash and summarize:**
```bash
youtube-to-docs -t gemini-2.0-flash-exp -m gemini-3-flash-preview
```

**3. Process the default video and save to a custom CSV:**
```bash
youtube-to-docs -o my-docs.csv
```

**4. Summarize a Playlist using multiple models (Gemini and Vertex):**
```bash
youtube-to-docs PLGKTTEqwhiHHWO-jdxM1KtzTbWo6h0Ycl -m gemini-3-flash-preview,vertex-claude-haiku-4-5@20251001
```

**5. Process a Channel with Summaries, TTS, and Infographics:**
```bash
youtube-to-docs @mga-othercommittees6625 -m vertex-claude-haiku-4-5@20251001 --tts gemini-2.5-flash-preview-tts-Kore --infographic gemini-2.5-flash-image
```

**6. Generate an Infographic using AWS Bedrock:**
```bash
youtube-to-docs atmGAHYpf_c --infographic bedrock-titan-image-generator-v2:0
```

**7. Create a Video (Infographic + Audio Summary):**
```bash
youtube-to-docs atmGAHYpf_c -m gemini-3-flash-preview --tts gemini-2.5-flash-preview-tts-Kore --infographic gemini-2.5-flash-image --combine-infographic-audio
```

## CSV Column Reference

The output CSV file contains a variety of columns depending on the arguments provided. Below is a reference of the possible columns:

### Base Metadata
*   **URL**: The full YouTube video URL.
*   **Title**: The title of the video.
*   **Description**: The video description.
*   **Data Published**: The date the video was published.
*   **Channel**: The name of the YouTube channel.
*   **Tags**: Video tags (comma-separated).
*   **Duration**: The duration of the video.
*   **Transcript characters from youtube**: The total number of characters in the YouTube transcript.
*   **Transcript characters from {model}**: The total number of characters in the AI-generated transcript (if applicable).
*   **Audio File**: Path to the extracted audio file (used for AI transcription).

### Files
*   **Transcript File {type}**: Path to the saved transcript file. `{type}` is either `youtube generated`, `human generated`, or `{model} generated`. Suffix `(lang)` added for non-English.
*   **Speakers File {model}**: Path to the saved speaker extraction text file.
*   **QA File {model}**: Path to the saved Q&A Markdown file. Suffix `(lang)` added for non-English.
*   **Summary File {model}**: Path to the Markdown summary file generated by a specific model. Suffix `(lang)` added for non-English.
*   **Summary Infographic File {model} {infographic_model}**: Path to the generated infographic image.
*   **Summary Audio File {model} {tts_model} File**: Path to the generated TTS audio file. Suffix `(lang)` added for non-English.
*   **Video File**: Path to the generated MP4 video combining the infographic and audio.

### AI Outputs & Costs
*   **Speakers {model}**: The extracted list of speakers and their roles.
*   **{normalized_model} Speaker extraction cost ($)**: The estimated API cost for speaker extraction.
*   **QA Text {model}**: The full text of the Q&A pairs (also saved to the Q&A file).
*   **{normalized_model} QA cost ($)**: The estimated API cost for Q&A generation.
*   **Summary Text {model}**: The full text of the summary (also saved to the summary file).
*   **{normalized_model} summary cost ($)**: The total estimated API cost for both speaker extraction and summarization.
*   **Summary Infographic Cost {model} {infographic_model} ($)**: The estimated API cost for infographic generation.
*   **{normalized_model} STT cost ($)**: The estimated API cost for Speech-to-Text generation.

> **Note**: `{normalized_model}` refers to the model name with prefixes (like `vertex-`) and date suffixes removed for cleaner column headers.

## Speaker Extraction

When a model is specified using the `-m` or `--model` argument, the tool automatically performs speaker extraction before generating a summary.

*   **Model Matching**: The extraction uses the same model as the summary. If multiple models are provided, each will perform its own extraction.
*   **Structured Output**: It identifies speakers and their professional titles or roles (e.g., "Speaker 1 (Senator Katie Fry Hester, Co-Chair)").
*   **Cost Tracking**: The cost of speaker extraction is tracked separately in the `{model} Speaker extraction cost ($)` column and included in the total `{model} summary cost ($)`.
*   **Unknowns**: If a speaker or title cannot be identified, the tool uses the placeholder `UNKNOWN`. If no speakers are detected at all, the field is set to `NaN`.

## MCP Server

This tool also functions as a Model Context Protocol (MCP) server, allowing it to be used as a tool by AI agents (like the Gemini CLI).

The server exposes a `process_video` tool that mirrors the CLI functionality.

### Configuration

The repository includes a `gemini-extension.json` file at the root, which configures the MCP server for use with the Gemini CLI.

### Usage

Once the extension is registered with your agent, you can ask it to process videos using natural language:

> "Save a summary of https://www.youtube.com/watch?v=KuPc06JgI_A"

The agent will prompt you for any necessary details (like the model to use) and then execute the tool.

### Install as a Gemini CLI extension

```bash
gemini extensions install https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs.git
```

## Library Usage

While primarily a CLI, you can import core functions for custom workflows.

```python
from youtube_to_docs.transcript import fetch_transcript

video_id = "atmGAHYpf_c"
transcript, is_generated = fetch_transcript(video_id)
```

## Why use `youtube-to-docs`?

You might find other tools that download YouTube transcripts, but `youtube-to-docs` distinguishes itself in several ways:

1.  **Multimodal Output**: It doesn't just stop at text.
    *   **Summaries**: Uses state-of-the-art LLMs to create concise summaries.
    *   **Speaker Extraction**: Automatically identifies speakers and their titles/roles from the transcript.
    *   **Audio (TTS)**: Converts summaries into audio files, perfect for listening on the go.
    *   **Visuals (Infographics)**: Generates AI-created infographics to visually represent the content.
    *   **Videos**: Combines infographics and audio into a shareable video summary.

2.  **Structured Data (CSV/Polars)**:
    *   Instead of loose files, metadata and paths are organized into a robust CSV file using `polars`.
    *   This makes it incredibly easy to import the data into Google Sheets, Excel, or a database for further analysis or publishing.

3.  **Batch Processing**:
    *   Seamlessly handles individual videos, entire playlists, or full channels with a single command.

4.  **Multi-Provider Support**:
    *   Agnostic to the LLM provider. Whether you use Google Gemini, Vertex AI, AWS Bedrock, or Azure Foundry, you can plug in your preferred model.

5.  **Cost Awareness**:
    *   When using paid API models, it tracks and estimates the cost of summarization, saving it directly to your data file.
