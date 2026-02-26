---
description: "Suggest WCAG 2.1 / Section 508 corrected captions for a YouTube video."
---

Please suggest corrected captions for the YouTube video at $ARGUMENTS using the `youtube-to-docs:process_video` tool.

- Identify the YouTube URL or ID from the arguments.
- Determine the `suggest_corrected_captions` value:
  - Format is `{model}` or `{model}-{source}`.
  - If "youtube" is mentioned as the source, append `-youtube` (e.g. `gemini-3-flash-preview-youtube`).
  - If an STT model is mentioned as the source (e.g. "gcp-chirp3"), append it (e.g. `gemini-3-flash-preview-gcp-chirp3`).
  - If no source is specified, omit the source suffix and let it auto-detect the most recent AI-generated SRT.
  - If "gemini pro" or "pro" is mentioned, use `gemini-3.1-pro-preview` as the model.
  - If no model is specified, default to `gemini-3-flash-preview`.
- If an STT source model is specified, also set `transcript_source` to that model so both steps run together.

Do not ask for confirmation.
