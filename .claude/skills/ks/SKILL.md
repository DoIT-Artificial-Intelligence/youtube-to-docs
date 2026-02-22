---
description: "(Kitchen Sink) Process a YouTube video with all features (summary, Q&A, infographic, audio, and video)."
---

Please process the YouTube video at $ARGUMENTS using the `youtube-to-docs:process_video` tool with the most comprehensive settings.

- Identify the YouTube URL or ID from the arguments.
- Check for model keywords:
  - If "gemini flash" or "flash" is mentioned, set `all_suite='gemini-flash'`.
  - If "gemini pro" or "pro" is mentioned (or if no model is specified), set `all_suite='gemini-pro'`.
  - If "gcp" is mentioned, set `all_suite='gcp-pro'`.
- Check for language keywords:
  - If "spanish" or "es" is mentioned, set `translate='gemini-3-flash-preview-es'`.
  - If "french" or "fr" is mentioned, set `translate='gemini-3-flash-preview-fr'`.
  - If no language is specified, omit `translate` (defaults to English only).
- Set the following additional parameters:
  - `verbose=True`
  - `combine_infographic_audio=True`

Execute the `youtube-to-docs:process_video` tool with these parameters. Do not ask for confirmation.
