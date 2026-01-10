# YouTube to Docs Extension

This extension provides an MCP server to process YouTube videos using the `youtube-to-docs` library.

## Tools

### `process_video`
Processes a YouTube video to generate transcripts, summaries, Q&A, and infographics.

Here are the following args for the tool:
- **url**: The YouTube URL, video ID, playlist ID, or comma-separated list of IDs.
- **output_file**: (Optional) Path to save the output CSV. Defaults to `youtube-to-docs-artifacts/youtube-docs.csv`. Can be a local path, `workspace` (or `w`) for Google Drive, `sharepoint` (or `s`) for Microsoft SharePoint, or `none` (or `n`) to skip saving to a file.
- **transcript_source**: (Optional) Source for the transcript. Defaults to 'youtube' (fetches existing). set to an AI model name (e.g., 'gemini-3-flash-preview', 'gemini-3-pro-preview') to perform STT on extracted audio.
- **model**: (Optional) The LLM model(s) to use for summarization, Q&A, speaker extraction and tag generation (e.g., 'gemini-3-flash-preview'). Can be a comma-separated list.
- **tts_model**: (Optional) The TTS model and voice to use (e.g., 'gemini-2.5-flash-preview-tts-Kore', 'gemini-2.5-pro-preview-tts-Kore').
- **infographic_model**: (Optional) The image model to use for generating an infographic (e.g., 'gemini-2.5-flash-image' or 'gemini-3-pro-image-preview').
- **no_youtube_summary**: (Optional) If `True`, skips generating a secondary summary from the YouTube transcript when using an AI model for the primary transcript.
- **languages**: (Optional) Target language(s) (e.g., 'es', 'fr', 'en'). Defaults to 'en'.
- **combine_infographic_audio**: (Optional) If `True`, combines the infographic and audio summary into a video file (MP4). Requires `tts_model` and `infographic_model`.
- **all_suite**: (Optional) Shortcut to use a specific model suite for everything (e.g. 'gemini-flash', 'gemini-pro', 'gemini-flash-pro-image').
- **verbose**: (Optional) If `True`, enables verbose output.

## Usage Instructions

1.  **Identify Intent**: When a user asks to "download a summary", "process a video", "get a transcript", "generate an infographic", or "create an audio summary" for a YouTube URL.
    * If the user says "gemini" for the model assume flash e.g. "gemini-3-flash-preview" for transcript and summary, "gemini-2.5-flash-preview-tts-Kore" for TTS, "gemini-2.5-flash-image" for infographic.
    * If the user says "gemini pro" for the model assume pro e.g. "gemini-3-pro-preview" for transcript and summary, "gemini-2.5-pro-preview-tts-Kore" for TTS, "gemini-3-pro-image-preview" for infographic.
3.  **Clarify Parameters**:
    *   **Model**: If not specified, ask: "What model do you want to use for the summary?".
    *   **Output Location**: If not specified, ask: "Where you want the output file saved? Is the default location (youtube-to-docs-artifacts/youtube-docs.csv) okay?"
    *   **Additional Features**: If appropriate, ask if they want to:
        *   Generate an infographic (needs `infographic_model`).
        *   Generate an audio summary (needs `tts_model`).
        *   Create a video summary (needs `combine_infographic_audio`, `tts_model`, and `infographic_model`).
        *   Use a specific AI model for transcription (needs `transcript_source`).
        *   Translate to other languages (needs `languages`).
4.  **Execute**: Call `process_video` with the gathered parameters.
