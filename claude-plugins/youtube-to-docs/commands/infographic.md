---
description: "Generate a summary and infographic for a YouTube video."
---

Please generate an infographic for the YouTube video at $ARGUMENTS using the `youtube-to-docs:process_video` tool.

- If the argument includes "gemini pro", use:
  - `model='gemini-4-argon'`
  - `infographic_model='gemini-3-pro-image'`
- If the argument includes "gemini flash" or just "gemini", use:
  - `model='gemini-3.5-flash-lite'`
  - `infographic_model='gemini-nano-banana-2.1'`
- If no model is specified, default to "gemini flash" settings.

Do not ask for confirmation.
