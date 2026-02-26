# YouTube to Docs Extension

This extension provides an MCP server to process YouTube videos using the `youtube-to-docs` library.

## Tools

### `process_video`

Processes a YouTube video to generate transcripts, summaries, Q&A, and infographics.

Here are the following args for the tool:

- **url**: The YouTube URL, video ID, playlist ID, or comma-separated list of IDs.
- **output_file**: (Optional) Path to save the output CSV. Defaults to `youtube-to-docs-artifacts/youtube-docs.csv`. Can be a local path, `workspace` (or `w`) for Google Drive, `sharepoint` (or `s`) for Microsoft SharePoint, or `none` (or `n`) to skip saving to a file.
- **transcript_source**: (Optional) Source for the transcript. Defaults to 'youtube' (fetches existing). set to an AI model name (e.g., 'gemini-3-flash-preview', 'gcp-chirp3') to perform STT on extracted audio. For `gcp-` models, `YTD_GCS_BUCKET_NAME` env var is recommended.
- **model**: (Optional) The LLM model(s) to use for summarization, Q&A, speaker extraction and tag generation (e.g., 'gemini-3-flash-preview'). Can be a comma-separated list.
- **tts_model**: (Optional) The TTS model and voice to use (e.g., 'gemini-2.5-flash-preview-tts-Kore', 'gemini-2.5-pro-preview-tts-Kore', 'gcp-chirp3', 'gcp-chirp3-Kore', 'aws-polly', 'aws-polly-Ruth').
- **infographic_model**: (Optional) The image model to use for generating an infographic (e.g., 'gemini-2.5-flash-image' or 'gemini-3-pro-image-preview').
- **alt_text_model**: (Optional) The LLM model to use for generating multimodal alt text for the infographic. Defaults to the summary model.
- **no_youtube_summary**: (Optional) If `True`, skips generating a secondary summary from the YouTube transcript when using an AI model for the primary transcript.
- **translate**: (Optional) Translate all outputs to a target language. Format: `{model}-{language}` (e.g., 'gemini-3-flash-preview-es', 'bedrock-nova-2-lite-v1-fr', 'aws-translate-es', 'gcp-translate-es'). Use `aws-translate` to use the AWS Translate service directly, or `gcp-translate` to use Google Cloud Translation API directly. Defaults to English only.
- **combine_infographic_audio**: (Optional) If `True`, combines the infographic and audio summary into a video file (MP4). Requires `tts_model` and `infographic_model`.
- **suggest_corrected_captions**: (Optional) Suggest WCAG 2.1 Level AA / Section 508 corrected captions using an LLM. Format: `{model}` or `{model}-{source}` (e.g., `'gemini-3-flash-preview'`, `'gemini-3-flash-preview-youtube'`, `'gemini-3-flash-preview-gcp-chirp3'`). Source can be `youtube` or an STT model name used for transcription; omit source to auto-detect the most recent AI-generated SRT. Output saved to `suggested-corrected-caption-files/`.
- **all_suite**: (Optional) Shortcut to use a specific model suite for everything (e.g. 'gemini-flash', 'gemini-pro', 'gemini-flash-pro-image', 'gcp-pro').
- **verbose**: (Optional) If `True`, enables verbose output.

## Usage Instructions

1.  **Identify Intent**: When a user asks to "download a summary", "process a video", "get a transcript", "generate an infographic", or "create an audio summary" for a YouTube URL.
    - **Model Shorthand**:
      - "gemini" or "gemini flash" -> **Flash**:
        - Summary/Transcript: `gemini-3-flash-preview`
        - TTS: `gemini-2.5-flash-preview-tts-Kore`
        - Infographic: `gemini-2.5-flash-image`
      - "gemini pro" -> **Pro**:
        - Summary/Transcript: `gemini-3.1-pro-preview`
        - TTS: `gemini-2.5-pro-preview-tts-Kore`
        - Infographic: `gemini-3-pro-image-preview`
      - "aws" -> **AWS**:
        - Summary/Transcript: `bedrock-nova-2-lite-v1`
        - TTS: `aws-polly-Ruth`
        - Infographic: `gemini-2.5-flash-image` (fallback)
2.  **Slash Commands**:
    - `/ytt <url>`: Fetches only the YouTube transcript (uses `ytt.toml`).
    - `/infographic <url> <family>`: Generates an infographic and summary.
      - `<family>` can be "gemini flash" or "gemini pro".
      - Defaults to Flash summary (`gemini-3-flash-preview`) with relevant image model.
    - `/ks <url> <family> <language>`: Kitchen Sink - generates everything and combines into a video.
      - `<family>` can be "gemini flash" or "gemini pro". Defaults to "gemini pro".
      - `<language>` can be "spanish", "french", etc. Defaults to "en".
      - Equivalent to `youtube_to_docs --all gemini-pro --verbose --combine-infographic-audio`.
    - `/scc <url> [model] [source]`: Suggest WCAG 2.1 / Section 508 corrected captions.
      - `[model]` defaults to `gemini-3-flash-preview`.
      - `[source]` can be "youtube" or an STT model name (e.g. "gcp-chirp3"); omit to auto-detect.
3.  **Clarify Parameters**:
    - **Model**: If not specified, ask: "What model do you want to use for the summary?".
    - **Output Location**: If not specified, ask: "Where you want the output file saved? Is the default location (youtube-to-docs-artifacts/youtube-docs.csv) okay?"
    - **Additional Features**: If appropriate, ask if they want to:
      - Generate an infographic (needs `infographic_model`).
      - Generate multimodal alt text for the infographic (uses `alt_text_model`, defaults to summary model).
      - Generate an audio summary (needs `tts_model`).
      - Create a video summary (needs `combine_infographic_audio`, `tts_model`, and `infographic_model`).
      - Use a specific AI model for transcription (needs `transcript_source`).
      - Translate to other languages (needs `translate`, format: `{model}-{language}` e.g. `gemini-3-flash-preview-es`, `aws-translate-es`, or `gcp-translate-es`).
      - Suggest corrected captions for WCAG 2.1 / Section 508 compliance (needs `suggest_corrected_captions`, format: `{model}` or `{model}-{source}` e.g. `gemini-3-flash-preview`, `gemini-3-flash-preview-youtube`).
4.  **Artifact Storage**: Note that granular textual artifacts are saved to:
    - `tag-files/`: AI-generated tags.
    - `one-sentence-summary-files/`: One-sentence summaries.
    - `infographic-alt-text/`: Multimodal infographic alt text.
    - `srt-files/`: SRT transcript files (both YouTube and AI generated).
    - `qa-files/`: Q&A markdown files (now includes "Timestamp" and "Timestamp URL" as markdown hyperlinks).
    - `suggested-corrected-caption-files/`: LLM-suggested WCAG 2.1 / Section 508 corrected SRT files (diff-style, changed segments only).
    - These are linked as columns in the output CSV.
5.  **Timestamp Accuracy**: The server automatically cross-references YouTube SRT timestamps when generating AI Q&A to ensure pinpoint accuracy for "Timestamp URL" links.
6.  **Execute**: Call `process_video` with the gathered parameters.
