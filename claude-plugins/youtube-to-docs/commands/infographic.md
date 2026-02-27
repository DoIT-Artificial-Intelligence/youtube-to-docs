---
description: "Generate a summary and infographic for a YouTube video."
---

Please generate an infographic for the YouTube video at $ARGUMENTS using the `youtube-to-docs:process_video` tool.

- If the argument includes "gemini pro", use:
  - `model='gemini-3-flash-preview'`
  - `infographic_model='gemini-3-pro-image-preview'`
- If the argument includes "gemini flash" or just "gemini", use:
  - `model='gemini-3-flash-preview'`
  - `infographic_model='gemini-3.1-flash-image-preview'`
- If no model is specified, default to "gemini pro" settings.

Do not ask for confirmation.
