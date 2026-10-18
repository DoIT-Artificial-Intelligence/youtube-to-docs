# youtube-to-docs
[![PyPI version](https://img.shields.io/pypi/v/youtube-to-docs.svg)](https://pypi.org/project/youtube-to-docs/)

Click on the image below to see a demo of YouTube to docs:

<div align="center">
  <a href="https://www.youtube.com/watch?v=H6Jw10bmMcI">
    <img src="https://img.youtube.com/vi/H6Jw10bmMcI/0.jpg" alt="Watch the video" width="300" />
  </a>
</div>

Convert YouTube videos into structured docs, summaries, audio, and visual assets for easier discovery.

View all available CLI options:

```bash
uvx youtube-to-docs --help
```

### Optional Features

To keep the installation light, some features are optional. You can enable them by specifying "extras":

- `audio`: Required for downloading audio and video from YouTube (uses `yt-dlp`, with `yt-dlp-ejs` and Deno to solve YouTube's challenges).
- `video`: Required for video generation (uses `static-ffmpeg`).
- `workspace`: Required for Google Drive integration.
- `m365`: Required for Microsoft SharePoint/OneDrive integration.
- `huggingface`: Required for storing artifacts to a Hugging Face dataset (uses `huggingface_hub`).
- `aws`: AWS Bedrock support.
- `azure`: Required for Azure OpenAI models.
- `gcp`: Required for Google Gemini, Vertex AI models, and GCP Cloud TTS (uses `google-genai` and google-cloud libs).
- `app`: Web app with browser-based UI (uses `fastapi`, `uvicorn`).
- `all`: Installs everything.

**Example: Run with audio and video support**
```bash
uvx --with "youtube-to-docs[audio,video]" youtube-to-docs ...
```

**Example: Run with everything**
```bash
uvx --with "youtube-to-docs[all]" youtube-to-docs ...
```

*Note: The commands above require `uv`. You can install it via:*
*   **macOS/Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
*   **Windows**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

Install as a plugin (the `process_video` MCP server and the `youtube-to-docs` skill) in Claude Code:

```bash
claude plugin marketplace add DoIT-Artificial-Intelligence/youtube-to-docs
claude plugin install youtube-to-docs@youtube-to-docs
```

or in Antigravity:

```bash
agy plugin install https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs
```

Then ask for what you want, e.g. "Make the best accessible captions for https://www.youtube.com/watch?v=B0x2I_doX9o", or run `/youtube-to-docs <url>`.

Run as a web app:

```bash
uvx --from "youtube-to-docs[all]" youtube-to-docs-app
```

Then open http://localhost:8000 in your browser. See the [Web App docs](https://doit-artificial-intelligence.github.io/youtube-to-docs/usage/#web-app) for details.

*Created with the help of AI. All artifacts have been checked and work as expected.*
