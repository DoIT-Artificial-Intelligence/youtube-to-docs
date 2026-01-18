---
name: youtube-to-docs
description: "Comprehensive suite for processing YouTube videos. Use this when the user needs to: (1) Extract transcripts, (2) Generate visual infographics, (3) Create audio summaries (TTS) and videos, or (4) Perform full 'kitchen sink' processing of YouTube content."
---

# YouTube to Docs

## Overview

This skill allows you to process YouTube videos to extract transcripts, generate AI summaries, create infographics, and even produce video summaries. You have access to the `process_video` tool which handles these operations.

## Requirements & Dependencies

The `process_video` tool is a high-level interface that relies on several optional libraries ("extras") and system binaries to function. These are managed automatically when running via the provided MCP configuration or `uv`.

*   **Python Libraries**: Many features (audio extraction, video generation, cloud storage) require specific extras.
*   **System Binaries**: Features like video creation (`combine_infographic_audio`) require `ffmpeg` (handled by the `static-ffmpeg` library).
*   **Automatic Setup**: The MCP server (configured in `gemini-extension.json`) uses `uv run --all-extras` to ensure all necessary libraries are installed in a managed environment before execution.

## Workflows

### 1. Transcript Extraction

Use this when the user simply wants the text transcript of a video, without additional AI processing.

*   **Goal**: Get the raw text from a YouTube video.
*   **Tool**: `process_video`
*   **Required Argument**: `url` (The YouTube link)
*   **Defaults**: By default, `process_video` fetches the transcript from YouTube.
*   **Example Prompt**: "Get the transcript for https://www.youtube.com/watch?v=..."

### 2. Infographic Generation

Use this when the user wants a visual summary or "infographic" representing the video's content.

*   **Goal**: Create a visual summary (image).
*   **Tool**: `process_video`
*   **Required Argument**: `url` (The YouTube link)
*   **Optional Arguments**:
    *   `infographic_model`: The image generation model to use.
    *   `model`: The text model for summarization (required context for the image).
*   **Model Selection Strategy**:
    *   **Pro (Default/High Quality)**: Use if "gemini pro" is requested or no preference is stated.
        *   `model='gemini-3-pro-preview'`
        *   `infographic_model='gemini-3-pro-image-preview'`
    *   **Flash (Speed/Cost)**: Use if "gemini flash" is requested.
        *   `model='gemini-3-flash-preview'`
        *   `infographic_model='gemini-2.5-flash-image'`
-   **Alt Text (Accessibility)**: By default, `process_video` generates multimodal alt text using the summary model (image-to-text) for any created infographic. Use `alt_text_model` to override the model for this step.
-   **Confirmation**: Proceed without asking for extra confirmation unless parameters are missing.

### 3. Kitchen Sink (Comprehensive Processing)

Use this when the user asks for "everything", a "kitchen sink" run, or a "video summary". This generates transcripts, text summaries, Q&A, audio summaries (TTS), infographics, and combines them into a video file.

*   **Goal**: Generate all possible artifacts, including a video file.
*   **Tool**: `process_video`
*   **Required Argument**: `url` (The YouTube link)
*   **Optional Arguments**:
    *   `all_suite`: Shortcut to set models (`'gemini-flash'` or `'gemini-pro'`).
    *   `combine_infographic_audio`: Set to `True` to create the final video (Requires `video` extra).
    *   `verbose`: Set to `True` for detailed logging.
    *   `languages`: Target language code (e.g., 'es', 'fr', 'en').
*   **Model Selection Strategy**:
    *   **Pro (Default)**: `all_suite='gemini-pro'` (best for video quality).
    *   **Flash**: `all_suite='gemini-flash'` (faster).
*   **Language Handling**:
    *   "spanish" or "es" -> `languages='es'`
    *   "french" or "fr" -> `languages='fr'`
    *   Default -> `languages='en'`

### 4. Custom / Advanced Usage

Use this when the user specifies particular models or output locations.

*   **Output Locations**:
    *   **Local**: Default.
    *   **Google Drive**: `output_file='workspace'` (Requires `workspace` extra).
    *   **SharePoint**: `output_file='sharepoint'` (Requires `m365` extra).
*   **Transcription Source**:
    *   Default is YouTube captions.
    *   To use AI for transcription (STT), set `transcript_source` to a model name (e.g., `'gemini-3-flash-preview'`) (Requires `audio` extra).

## Tool Reference: `process_video`

| Argument | Description | Required Extra | Examples |
| :--- | :--- | :--- | :--- |
| `url` | **Required**. YouTube URL, ID, Playlist ID, or Channel Handle. | - | `https://youtu.be/...`, `@channel` |
| `model` | LLM for summaries/Q&A. | `gcp` / `azure` | `gemini-3-flash-preview` |
| `infographic_model` | Model for generating the infographic image. | `gcp` | `gemini-3-pro-image-preview` |
| `alt_text_model` | Model for generating multimodal alt text for the infographic. | `gcp` | `gemini-3-flash-preview` |
| `tts_model` | Model for text-to-speech audio. | `gcp` | `gemini-2.5-flash-preview-tts-Kore` |
| `all_suite` | Shortcut to apply a suite of models. | `gcp`, `audio`, `video` | `gemini-pro`, `gemini-flash` |
| `combine_infographic_audio` | Boolean. If True, creates an MP4 video. | `video` | `True` |
| `languages` | Target language(s). | - | `es`, `fr`, `en` |
| `output_file` | Destination for the CSV report. | `workspace` / `m365` | `workspace`, `sharepoint` |
| `transcript_source` | Source for transcript (default: 'youtube'). | `audio` (for AI STT) | `gemini-3-flash-preview` |

## Examples

**User**: "Get me a transcript of this video."
**Action**: Call `process_video(url='...')`

**User**: "Make an infographic for this video using Gemini Pro."
**Action**: Call `process_video(url='...', model='gemini-3-pro-preview', infographic_model='gemini-3-pro-image-preview')`

**User**: "Do a kitchen sink run on this video in Spanish."
**Action**: Call `process_video(url='...', all_suite='gemini-pro', combine_infographic_audio=True, verbose=True, languages='es')`

**User**: "Summarize this playlist and save it to Drive."
**Action**: Call `process_video(url='PL...', model='gemini-3-flash-preview', output_file='workspace')`

## Development & CLI Usage

While this skill primarily uses the `process_video` tool, you can also run the underlying CLI manually for testing or development.

**Note on CLI Syntax**: The video URL/ID is a **positional** argument and is **required**. Do NOT use `--url`.

**Always use `uv` to run the tool** (do not use `python` directly) to ensure dependencies are correctly resolved:

```bash
# General Syntax:
uv run youtube-to-docs <video_url_or_id> [options]

# Example: Get transcript
uv run youtube-to-docs https://www.youtube.com/watch?v=B0x2I_doX9o

# Example: Kitchen sink with gemini-pro suite
uv run youtube-to-docs B0x2I_doX9o --all gemini-pro --verbose
```

See `docs/usage.md` for full documentation and `docs/development.md` for setup details.

**MCP Configuration:**
The MCP server definition is located in `gemini-extension.json`. It is explicitly configured to use `uv` with `--all-extras` to ensure the correct environment and dependencies are used:

```json
"command": "uv",
"args": [ ..., "run", "--all-extras", "python", "-m", "youtube_to_docs.mcp_server" ]
```