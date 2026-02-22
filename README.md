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

- `audio`: Required for TTS and audio processing (uses `yt-dlp`).
- `video`: Required for video generation (uses `static-ffmpeg`).
- `workspace`: Required for Google Drive integration.
- `m365`: Required for Microsoft SharePoint/OneDrive integration.
- `aws`: AWS Bedrock support.
- `azure`: Required for Azure OpenAI models.
- `gcp`: Required for Google Gemini and Vertex AI models (uses `google-genai`).
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

Install as a Gemini CLI extension:

```bash
gemini extensions install https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs.git
```

Install as a Claude Code plugin:

```bash
/plugin marketplace add DoIT-Artificial-Intelligence/youtube-to-docs
```

*Created with the help of AI. All artifacts have been checked and work as expected.*
