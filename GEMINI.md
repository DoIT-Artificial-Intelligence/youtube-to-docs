# YouTube to Docs Extension

This extension provides an MCP server to process YouTube videos using the `youtube-to-docs` library.

## Tools

### `process_video`
Processes a YouTube video to generate transcripts, summaries, Q&A, and infographics.

- **url**: The YouTube URL, video ID, playlist ID, or comma-separated list of IDs.
- **output_file**: (Optional) Path to save the output CSV. Defaults to `youtube-to-docs-artifacts/youtube-docs.csv`.
- **transcript_source**: (Optional) Source for the transcript. Defaults to 'youtube' (fetches existing). set to an AI model name (e.g., 'gemini-3-flash-preview') to perform STT on extracted audio.
- **model**: (Optional) The LLM model(s) to use for summarization, Q&A, and speaker extraction (e.g., 'gemini-3-flash-preview'). Can be a comma-separated list.
- **tts_model**: (Optional) The TTS model and voice to use (e.g., 'gemini-2.5-flash-preview-tts-Kore').
- **infographic_model**: (Optional) The image model to use for generating an infographic (e.g., 'gemini-2.5-flash-image').
- **no_youtube_summary**: (Optional) If `True`, skips generating a secondary summary from the YouTube transcript when using an AI model for the primary transcript.
- **languages**: (Optional) Target language(s) (e.g., 'es', 'fr', 'en'). Defaults to 'en'.
- **combine_infographic_audio**: (Optional) If `True`, combines the infographic and audio summary into a video file (MP4). Requires `tts_model` and `infographic_model`.
- **all_suite**: (Optional) Shortcut to use a specific model suite for everything (e.g. 'gemini-flash', 'gemini-pro').
- **verbose**: (Optional) If `True`, enables verbose output.

## Usage Instructions

1.  **Identify Intent**: When a user asks to "download a summary", "process a video", "get a transcript", "generate an infographic", or "create an audio summary" for a YouTube URL.
2.  **Clarify Parameters**:
    *   **Model**: If not specified, ask: "What model do you want to use for the summary?".
    *   **Output Location**: If not specified, ask: "Where you want the output file saved? Is the default location (youtube-to-docs-artifacts/youtube-docs.csv) okay?"
    *   **Additional Features**: If appropriate, ask if they want to:
        *   Generate an infographic (needs `infographic_model`).
        *   Generate an audio summary (needs `tts_model`).
        *   Create a video summary (needs `combine_infographic_audio`, `tts_model`, and `infographic_model`).
        *   Use a specific AI model for transcription (needs `transcript_source`).
        *   Translate to other languages (needs `languages`).
3.  **Execute**: Call `process_video` with the gathered parameters.
4.  **Report**: Present the result (summary or confirmation) to the user.