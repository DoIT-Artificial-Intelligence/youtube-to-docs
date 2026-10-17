# YouTube to Docs Extension

This extension provides an MCP server to process YouTube videos using the `youtube-to-docs` library.

## Tools

### `process_video`

Processes a YouTube video to generate transcripts, summaries, Q&A, and infographics.

Here are the following args for the tool:

- **url**: The YouTube URL, video ID, playlist ID, or comma-separated list of IDs.
- **output_file**: (Optional) Path to save the output CSV. Defaults to `youtube-to-docs-artifacts/youtube-docs.csv`. Can be a local `.csv` path inside `youtube-to-docs-artifacts/`, `workspace` (or `w`) for Google Drive, `sharepoint` (or `s`) for Microsoft SharePoint, `hf` for a Hugging Face dataset (requires `hugging_face_dataset` and the `HF_TOKEN` env var), `memory` (or `m`) to keep artifacts in memory (no files on disk), or `none` (or `n`) to skip saving to a file.
- **hugging_face_dataset**: (Optional) The Hugging Face dataset repository to store artifacts in when `output_file` is `hf`. Can be a full repo id (e.g. `username/my-dataset`) or a plain name (e.g. `Code for America Summit 2026 Recap`) which is slugified and namespaced under the authenticated user. If omitted and the input is a playlist, the playlist's title is used as the dataset name. Requires the `HF_TOKEN` environment variable.
- **transcript_source**: (Optional) Source for the transcript. Defaults to 'youtube' (fetches existing), or to the STT model that `corrected_captions` corrects. set to an AI model name (e.g., 'gemini-3.5-transcribe', 'gemini-3.5-flash-lite', 'gcp-chirp3') to perform STT on extracted audio. `gemini-3.5-transcribe` is Google's dedicated STT model and returns word-level timestamps plus speaker diarization, so its SRT has real offsets and `[Speaker N]` labels on speaker changes. For `gcp-` models, `YTD_GCS_BUCKET_NAME` env var is recommended.
- **model**: (Optional) The LLM model(s) to use for summarization, Q&A, speaker extraction and tag generation (e.g., 'gemini-3.5-flash-lite'). Can be a comma-separated list. Defaults to `None` (Transcript only), but using the `-m` flag without an argument defaults to `gemini-3.5-flash-lite`.
- **tts_model**: (Optional) The TTS model and voice to use (e.g., 'gemini-3.8-flash-lite-tts-Kore', 'gemini-3.8-flash-lite-tts-Kore', 'gcp-chirp3', 'gcp-chirp3-Kore', 'aws-polly', 'aws-polly-Ruth').
- **infographic_model**: (Optional) The image model to use for generating an infographic (e.g., 'gemini-nano-banana-2.1' or 'gemini-3-pro-image'). Can be a comma-separated list to generate one infographic per model.
- **alt_text_model**: (Optional) The LLM model to use for generating multimodal alt text for the infographic. Defaults to the summary model.
- **no_youtube_summary**: (Optional) If `True`, skips generating a secondary summary from the YouTube transcript when using an AI model for the primary transcript.
- **from_video**: (Optional) Also generate the summary, one sentence summary, speakers and Q&A directly from the video using Gemini video understanding (no download needed). Value is the processing mode: `agentic` (the model navigates the video, loading only what it needs; Gemini 3.5 Flash-Lite and later, e.g. `gemini-3.8-flash`, `gemini-4-argon`), `static` (frames at 1 FPS; any Gemini model), or `agentic,static` to run both. Outputs are labelled `from video-agentic` / `from video-static`.
- **translate**: (Optional) Translate all outputs to a target language. Format: `{model}-{language}` (e.g., 'gemini-3.5-flash-lite-es', 'bedrock-nova-2-lite-v1-fr', 'aws-translate-es', 'gcp-translate-es'). Use `aws-translate` to use the AWS Translate service directly, or `gcp-translate` to use Google Cloud Translation API directly. Defaults to English only.
- **combine_infographic_audio**: (Optional) If `True`, combines the infographic and audio summary into a video file (MP4). Requires `tts_model` and `infographic_model`.
- **corrected_captions**: (Optional) Make the video accessible under WCAG 2.1 Level AA / Section 508 in one run: corrected captions (SRT in `corrected-caption-files/`, WebVTT in `vtt-files/`) with real speaker names on every change and corrected words, sounds and timing; audio description of the slides and other visual information (WebVTT descriptions tracks in `vtt-files/` and a descriptive transcript in `descriptive-transcript-files/`); and in `video-files/` the original video plus a copy with the captions burned in and the descriptions narrated (`tts_model`, default `gemini-3.8-flash-tts-Kore`). Also writes an Able Player page in `vtt-files/` for checking the WebVTT files. Use `'default'` for the best results (`gemini-4-argon` on a `gemini-3.5-transcribe` transcript, transcribed in the same run), or `{model}` / `{model}-{source}`, e.g. `'gemini-4-argon-youtube'`. Takes about 20 minutes for a 3-hour video.
- **post_process**: (Optional) Post-process the transcript with JSON operations. Example: `'{"word count": "apple"}'` counts occurrences of 'apple' in the transcript. Values can be a list for multiple words: `'{"word count": ["apple", "banana"]}'`. Results are added as columns in the output CSV.
- **all_suite**: (Optional) Shortcut to use a specific model suite for everything (e.g. 'gemini-flash', 'gemini-pro', 'gemini-flash-pro-image', 'gcp-pro').
- **verbose**: (Optional) If `True`, enables verbose output.

## Usage Instructions

1.  **Identify Intent**: When a user asks to "download a summary", "process a video", "get a transcript", "generate an infographic", or "create an audio summary" for a YouTube URL.
    - **Model Shorthand**:
      - "gemini" or "gemini flash" -> **Flash**:
        - Summary/Transcript: `gemini-3.5-flash-lite`
        - TTS: `gemini-3.8-flash-lite-tts-Kore`
        - Infographic: `gemini-nano-banana-2.1`
      - "gemini pro" -> **Pro**:
        - Summary/Transcript: `gemini-4-argon`
        - TTS: `gemini-3.8-flash-tts-Kore`
        - Infographic: `gemini-3-pro-image`
      - "aws" -> **AWS**:
        - Summary/Transcript: `bedrock-nova-2-lite-v1`
        - TTS: `aws-polly-Ruth`
        - Infographic: `gemini-3-pro-image` (fallback)
2.  **Slash Commands**:
    - `/ytt <url>`: Fetches only the YouTube transcript (uses `ytt.toml`).
    - `/infographic <url> <family>`: Generates an infographic and summary.
      - `<family>` can be "gemini flash" or "gemini pro".
      - Defaults to Flash summary (`gemini-3.5-flash-lite`) with relevant image model.
    - `/ks <url> <family> <language>`: Kitchen Sink - generates everything and combines into a video.
      - `<family>` can be "gemini flash" or "gemini pro". Defaults to "gemini pro".
      - `<language>` can be "spanish", "french", etc. Defaults to "en".
      - Equivalent to `youtube_to_docs --all gemini-pro --verbose --combine-infographic-audio`.
    - `/cc <url> [model] [source]`: Make the video accessible: WCAG 2.1 / Section 508 corrected captions, audio description and accessible videos.
      - Defaults to `corrected_captions='default'` (`gemini-4-argon` on a `gemini-3.5-transcribe` transcript).
      - `[source]` can be "youtube" or an STT model name (e.g. "gcp-chirp3"); `[model]` e.g. `gemini-3.5-flash-lite` for a cheaper run.
3.  **Clarify Parameters**:
    - **Model**: Defaults to `gemini-3.5-flash-lite`. If the user has a specific preference not covered by the default, ask: "What model do you want to use for the summary?".
    - **Output Location**: If not specified, ask: "Where you want the output file saved? Is the default location (youtube-to-docs-artifacts/youtube-docs.csv) okay?"
    - **Additional Features**: If appropriate, ask if they want to:
      - Generate an infographic (needs `infographic_model`). If `-i` is used as a flag without a value (e.g. `uvx youtube-to-docs ... -i`), it defaults to `gemini-3-pro-image`.
      - Generate multimodal alt text for the infographic (uses `alt_text_model`, defaults to summary model).
      - Generate an audio summary (needs `tts_model`).
      - Create a video summary (needs `combine_infographic_audio`, `tts_model`, and `infographic_model`).
      - Use a specific AI model for transcription (needs `transcript_source`).
      - Translate to other languages (needs `translate`, format: `{model}-{language}` e.g. `gemini-3.5-flash-lite-es`, `aws-translate-es`, or `gcp-translate-es`).
      - Make the video accessible for WCAG 2.1 / Section 508: corrected captions, audio description and accessible videos (needs `corrected_captions`, `'default'` for the best, or `{model}-{source}` e.g. `gemini-4-argon-youtube`).
4.  **Artifact Storage**: Note that granular textual artifacts are saved to:
    - `tag-files/`: AI-generated tags.
    - `one-sentence-summary-files/`: One-sentence summaries.
    - `infographic-alt-text/`: Multimodal infographic alt text.
    - `infographic-prompts/`: The exact text prompt sent to the image model to generate each infographic.
    - `summary-prompts/`: The exact text prompt sent to the LLM to generate each summary.
    - `srt-files/`: SRT transcript files (both YouTube and AI generated).
    - `qa-files/`: Q&A markdown files (now includes "Timestamp" and "Timestamp URL" as markdown hyperlinks).
    - `corrected-caption-files/`: Complete WCAG 2.1 / Section 508 corrected SRT files.
    - `vtt-files/`: WebVTT captions and descriptions tracks for websites, and an Able Player page for checking them.
    - `descriptive-transcript-files/`: The dialogue by speaker with descriptions of the visual information.
    - `video-files/`: The original video and a copy with captions burned in and descriptions narrated.
    - `video-context-files/`: On-screen names, speaking times, terms and events read from the video (used for corrected captions).
    - These are linked as columns in the output CSV.
5.  **Timestamp Accuracy**: The server automatically cross-references YouTube SRT timestamps when generating AI Q&A to ensure pinpoint accuracy for "Timestamp URL" links.
6.  **Execute**: Call `process_video` with the gathered parameters.
