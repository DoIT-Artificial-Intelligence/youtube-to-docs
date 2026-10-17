---
description: "Create WCAG 2.1 / Section 508 compliant corrected captions (SRT) for a YouTube video."
---

Please create corrected captions for the YouTube video at $ARGUMENTS using the `youtube-to-docs:process_video` tool.

- Identify the YouTube URL or ID from the arguments.
- Set `corrected_captions`:
  - By default use `'default'`: the best captions, with `gemini-4-argon` correcting a `gemini-3.5-transcribe` transcript (transcribed in the same run) and speaker names and spellings read from the video.
  - If "youtube" is mentioned as the source, use `gemini-4-argon-youtube` (corrects the YouTube captions; no transcription).
  - If an STT model is mentioned as the source (e.g. "gcp-chirp3"), use `gemini-4-argon-{stt model}` (e.g. `gemini-4-argon-gcp-chirp3`).
  - If a faster or cheaper model is requested ("flash", "cheap"), use `gemini-3.5-flash-lite` as the model, e.g. `gemini-3.5-flash-lite-gemini-3.5-transcribe`.
- If a captioned video, burned-in captions or open captions are requested, also set `corrected_captions_video=True`.

Do not ask for confirmation.
