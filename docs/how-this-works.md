# How This Works

`youtube-to-docs` takes a YouTube video, playlist or channel and turns each video into a set of documents: a transcript, speakers, Q&A, summaries, tags, an infographic, an audio summary and, with `-cc`, accessible captions and audio description. Every output is saved as a file and listed in one CSV.

## Watch the overview

<video controls preload="metadata" poster="../assets/explainer/poster.webp" width="100%">
  <source src="../assets/explainer/explainer.mp4" type="video/mp4">
  <track kind="captions" src="../assets/explainer/captions.vtt" srclang="en" label="English" default>
  Your browser can't play this video. <a href="../assets/explainer/explainer.mp4">Download the explainer video (MP4)</a>.
</video>

??? note "Transcript of the video"

    --8<-- "docs/assets/explainer/transcript.md"

## The pipeline at a glance

1. **[Point it at YouTube](#1-input).** A video URL or ID, a playlist, an `@channel`, or a comma-separated list.
2. **[Get the words](#2-transcripts).** YouTube's captions, or a fresh speech-to-text transcript with word timings and speaker labels. Either way you get a transcript and an SRT file.
3. **[Let the model read](#3-text-outputs).** An LLM on Gemini, Vertex AI, AWS Bedrock or Azure Foundry writes the speakers, Q&A, summaries and tags.
4. **[Turn it into media](#4-media-and-translation).** An infographic with alt text, an audio summary, a video that combines them, and translations.
5. **[Make the video accessible](#5-accessibility).** `-cc` writes corrected WCAG 2.1 AA captions and an audio description.
6. **[You stay in control](#6-review-storage-and-cost).** Every output is a file you can edit, saved locally or to Google Drive, SharePoint or Hugging Face, with costs in the CSV.

Each step checks for its output before it runs, so a second run only does what's missing and keeps your edits.

## 1. Input

The entry point (`youtube_to_docs/main.py`) accepts:

| Input | Example |
| :--- | :--- |
| Video URL | `https://www.youtube.com/watch?v=atmGAHYpf_c` |
| Video ID | `atmGAHYpf_c` |
| Playlist | `PL8ZxoInteClyHaiReuOHpv6Z4SPrXtYtW` |
| Channel handle (its Uploads playlist) | `@mga-hgo1740` |
| Comma-separated list | `KuPc06JgI_A,GalhDyf3F8g` |

`youtube_to_docs.transcript.resolve_video_ids` uses the YouTube Data API to expand playlists and channels into their videos, which are then processed one at a time.

## 2. Transcripts

`-t` / `--transcript` chooses where the words come from:

- **YouTube** (default): the video's existing captions, either uploaded or automatic. Automatic captions come from speech recognition, so they can contain errors, and they aren't always available straight away.
- **Speech-to-text**: an AI model name (e.g. `gemini-3.5-transcribe`, `gcp-chirp3`, `aws-transcribe`) downloads the audio with `yt-dlp` and transcribes it again.

Both give a plain-text transcript and an `.srt` file with timings. The SRT is used later to put timestamps on the Q&A and to make corrected captions.

??? info "Speech-to-text details"

    - `gemini-3.5-transcribe` is Google's dedicated speech-to-text model. It's called with a transcription config that asks for word timestamps and speaker diarization, so the SRT is built from real word offsets rather than times the model guessed, with `[Speaker N]` labels on speaker changes. Audio over 15 MB is uploaded with the Gemini Files API and deleted afterwards. The API reports audio input tokens but not transcript output tokens, so the STT cost column covers the input side only.
    - Other Gemini models (e.g. `gemini-3.5-flash-lite`) can be prompted to transcribe too.
    - `gcp-` models (Cloud Speech-to-Text V2) need `GOOGLE_CLOUD_PROJECT`; `YTD_GCS_BUCKET_NAME` is recommended.
    - `aws-transcribe` needs `YTD_S3_BUCKET_NAME`.
    - With an AI transcript, a second summary is also written from the YouTube transcript for comparison. `--no-youtube-summary` skips it.

## 3. Text outputs

`-m` / `--model` chooses the LLM (`youtube_to_docs/llms.py`). The prefix picks the provider:

| Provider | Prefix | Example |
| :--- | :--- | :--- |
| Google Gemini and Gemma | `gemini-`, `gemma-` | `gemini-3.5-flash-lite` (the default for a bare `-m`) |
| Google Cloud Vertex AI | `vertex-` | `vertex-claude-haiku-5-5` |
| AWS Bedrock | `bedrock-` (or a bare `claude-` or `nova-`) | `bedrock-claude-haiku-5-5`, `bedrock-gpt-6-luna` |
| Azure Foundry | `foundry-` | `foundry-gpt-6-luna` |

OpenAI GPT models on Bedrock are called through the Bedrock Mantle Chat Completions endpoint. A comma-separated list runs several models side by side, each with its own columns.

For each video, the model writes:

| Output | Made from | Contents |
| :--- | :--- | :--- |
| Speakers | Transcript | Each speaker's name and title or role |
| Q&A | Transcript, speakers and the SRT | A table of Questioner, Question, Responder, Answer, Timestamp and a timestamp link to that second of the video |
| Summary | Transcript and video details | A Markdown summary |
| One-sentence summary | Summary | One sentence for a catalog or listing |
| Tags | Summary | Up to 5 comma-separated topic tags |
| Description | Summary, the SRT and tags | A YouTube description: the summary, `Chapters:` with one `MM:SS Title` line per section, and the tags as hashtags |

The SRT is passed with the Q&A prompt as a *timing reference*, so each answer's timestamp comes from the captions' real timings rather than a guess.

### Model suites

`-a` / `--all` sets the text model, transcript, infographic and voice in one go:

| Suite | Text model | Transcript | Infographic | Voice |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-flash` | `gemini-3.5-flash-lite` | `gemini-3.5-flash-lite` | `gemini-nano-banana-2.1` | `gemini-3.8-flash-lite-tts-Kore` |
| `gemini-pro` | `gemini-4-argon` | `gemini-4-argon` | `gemini-3-pro-image` | `gemini-3.8-flash-tts-Kore` |
| `gemini-flash-pro-image` | `gemini-3.5-flash-lite` | `gemini-3.5-flash-lite` | `gemini-3-pro-image` | `gemini-3.8-flash-lite-tts-Kore` |
| `gcp-pro` | `gemini-4-argon` | `gcp-chirp3` | `gemini-3-pro-image` | `gcp-chirp3` |
| `anthropic-opus` | `claude-opus-5-5` (Bedrock) | `aws-transcribe` | `gemini-3-pro-image` | `aws-polly-Ruth` |

### Video understanding (`--from-video`)

`--from-video agentic|static` (or `agentic,static`) sends the YouTube URL to Gemini through the Interactions API, without downloading the video. Gemini writes a summary, one-sentence summary, speakers, Q&A and on-screen context (`video-context-files/`) from the video itself, so it can use slides, charts and name plates that a transcript misses. These are saved next to the transcript-based outputs under the source `video-agentic` or `video-static`, and get infographics and audio too.

- **Agentic** (Gemini 3.5 Flash-Lite and later, e.g. `gemini-3.8-flash`, `gemini-4-argon`): the model moves around the video, loading transcript, frames or audio only when it needs them. Works on multi-hour videos.
- **Static**: frames are sampled at 1 FPS for the whole video. Limited to videos that fit in the model's context window (about 3 hours at the default resolution).

The first request loads the video, and later requests continue the same server-side session. Requests run in the background and are polled, with a 30-minute timeout. Costs include the model's reasoning (thought tokens) and the content it loads (tool use tokens). The transcript-based pipeline still runs alongside.

## 4. Media and translation

| Flag | Output | Details |
| :--- | :--- | :--- |
| `-i` / `--infographic` | One infographic per summary, in `infographic-files/` | Gemini (`gemini-3-pro-image` for a bare `-i`, `gemini-nano-banana-2.1`, `gemini-3.1-flash-lite-image`) or Azure Foundry GPT Image 2.5 (`foundry-gpt-image-2.5-sunburst`, `foundry-gpt-image-2.5-flare`). 16:9 landscape at the model's highest resolution, so it works as a thumbnail. The prompt (video title and summary) is saved in `infographic-prompts/`. |
| `--alt-text-model` | Alt text for each infographic, in `infographic-alt-text/` | A multimodal model looks at the image itself and describes its layout and content. Defaults to the summary model. Prefixes such as "Alt text:" are removed. |
| `--tts` | The summary read aloud, in `audio-files/` | Gemini TTS (e.g. `gemini-3.8-flash-lite-tts-Kore`), Google Cloud (`gcp-chirp3`) or Amazon Polly (`aws-polly-Ruth`). |
| `--combine-infographic-audio` | An MP4 of the infographic with the audio, in `video-files/` | Made with `static-ffmpeg`, so there's no FFmpeg to install. Scaled to fit 3840×2160. |

See [Performance](performance.md#infographics) for how long each image model takes and what it costs.

### Translation

`--translate {model}-{language}` (e.g. `gemini-3.5-flash-lite-es`) makes every output again in a second language. Use `aws-translate-{language}` or `gcp-translate-{language}` to translate with AWS Translate or Google Cloud Translation instead of an LLM. Long texts are split to fit each service's limits.

Everything is made in English first, then for the target language:

1. **Transcript and SRT**: YouTube's transcript in that language if there is one, otherwise the English transcript and SRT are translated.
2. **Summaries, Q&A and tags**: written again from the translated transcript with the same model.
3. **Infographic, audio and video**: made in both languages when `-i`, `--tts` or `--combine-infographic-audio` are set.

File names end in `({model}-{language})`, e.g. `(gcp-translate-es)`, to show how they were translated. CSV columns use the shorter `(es)`.

## 5. Accessibility

`-cc` / `--corrected-captions` makes the original video accessible under WCAG 2.1 Level AA and Section 508. Without a value it uses `gemini-4-argon-gemini-3.5-transcribe`: the audio is transcribed with `gemini-3.5-transcribe` (unless `-t` is set) and the captions are corrected with `gemini-4-argon`. The format is `{model}` or `{model}-{source}`, e.g. `gemini-4-argon-youtube`.

**Captions** (`corrected-caption-files/`, and WebVTT in `vtt-files/`):

1. **Timing.** The source SRT is aligned word by word with a second transcript of the same audio (YouTube's captions, or the STT transcript when correcting YouTube's), and cues with broken timestamps are re-timed.
2. **Speakers.** Every cue is given a speaker, in windows of about 20 minutes, from diarization labels, the second transcript's speaker changes and the **on-screen context**: names and titles read from name plates, with the times each person is seen speaking. With Gemini, this context comes from one agentic request to the video, unless `--from-video` already made it.
3. **Text.** Sections are corrected in parallel against the second transcript and the on-screen spellings: misheard words and names, punctuation, fillers, non-speech sounds and missed speech. The model edits text by cue ID and never writes timestamps, so long videos can't drift out of sync.
4. **Layout.** Words are laid out as captions of at most two 42-character lines over 1–7 seconds, broken at sentence and clause boundaries, with `[Name]` on each change of speaker and no overlapping cues.

**Audio description**: scene changes in the downloaded video are described by the model from the frames and nearby dialogue, and fitted into pauses in the speech. The descriptions are saved as WebVTT description tracks (`vtt-files/`, standard and extended) and a descriptive transcript (`descriptive-transcript-files/`).

`video-files/` gets the original video and a copy with the captions burned in and the descriptions narrated. `vtt-files/` also gets an [Able Player](https://ableplayer.github.io/ableplayer/) page for checking the tracks in a browser, and a self-contained version that streams the video from YouTube, for pasting into a site such as Google Sites. See [Corrected Captions and Audio Description](usage.md#corrected-captions-and-audio-description) for the options.

## 6. Review, storage and cost

**People have the final say.** Every output is a file, or a Google Doc or Word file on Drive or SharePoint. When a step finds its output already exists, it reads it back instead of making it again, so later steps use any edits you've made, such as a corrected transcript. To make an output again, delete it. See [Review and edit the outputs](usage.md#review-and-edit-the-outputs).

**Storage.** `-o` / `--outfile` chooses where the CSV and files go:

| Value | Saves to |
| :--- | :--- |
| A file path (default `youtube-to-docs-artifacts/youtube-docs.csv`) | Your computer |
| `workspace` / `w`, or a folder ID | Google Drive, folder `youtube-to-docs-artifacts` |
| `sharepoint` / `s` | SharePoint or OneDrive, folder `youtube-to-docs-artifacts` |
| `hf` | A Hugging Face dataset (`--hugging-face-dataset`) |
| `memory` / `m` | Memory only, with no files on disk (used by the web app) |
| `none` / `n` | Nowhere; results are only in the log |

Storage is abstracted in `youtube_to_docs/storage.py` (`LocalStorage`, `GoogleDriveStorage`, `M365Storage`, `HuggingFaceStorage`, `MemoryStorage`, `NullStorage`), so the rest of the pipeline doesn't depend on where files go.

**Cost.** Every API call's token usage is priced from `youtube_to_docs/prices.py`, and the cost of each step (speakers, Q&A, summary, tags, transcription, infographic, alt text and corrected captions) is saved in its own CSV column. See [Pricing](pricing.md).

## Interfaces

The CLI, MCP server and web app all run the same `main()` pipeline.

### CLI

```bash
youtube-to-docs <url> [options]
```

Run `youtube-to-docs --help` for every option, or see the [Usage Guide](usage.md).

### MCP Server

`youtube_to_docs/mcp_server.py` exposes a `process_video` tool over the [Model Context Protocol](https://modelcontextprotocol.io), so AI assistants such as Claude can run the pipeline. It's included in the Claude Code and Antigravity plugin; see [MCP Server](usage.md#mcp-server) to install it.

### Web App

A FastAPI app (`youtube_to_docs/app.py`) gives a browser form for submitting videos and watching progress:

```bash
youtube-to-docs-app
# App available at: http://localhost:8000
```

| Endpoint | Purpose |
| :--- | :--- |
| `POST /api/process` | Submits a job and returns a `job_id` |
| `GET /api/jobs/{job_id}` | Job status, output log and artifact list |
| `GET /api/jobs/{job_id}/stream` | Job output in real time (Server-Sent Events) |
| `GET /api/artifacts/{path}` | Downloads an artifact |
| `GET /api/model-suites` | The model suites |

Jobs run in the background with `asyncio.to_thread`, and artifacts are collected from the output folders when a job finishes.

## Output folder

The CSV (written with `polars`) has a row per video, with its details, the path to every file and each step's cost. The files are organized by type:

```text
youtube-to-docs-artifacts/
├── youtube-docs.csv              # One row per video: details, file paths, costs
├── transcript-files/             # Plain-text transcripts
├── srt-files/                    # Timed SRT transcripts
├── audio-files/                  # Downloaded audio (for speech-to-text) and audio summaries
├── speaker-extraction-files/     # Speakers and their titles
├── qa-files/                     # Q&A tables with timestamp links
├── summary-files/                # Markdown summaries
├── summary-prompts/              # The exact prompts sent for each summary
├── one-sentence-summary-files/   # One-sentence summaries
├── tag-files/                    # Topic tags
├── descriptions-files/           # YouTube descriptions: summary, chapters, hashtags
├── video-context-files/          # On-screen context from the video (--from-video, -cc)
├── infographic-files/            # Infographic images
├── infographic-prompts/          # The exact prompts sent for each infographic
├── infographic-alt-text/         # Alt text for each infographic
├── corrected-caption-files/      # WCAG 2.1 AA / Section 508 SRT captions (-cc)
├── vtt-files/                    # WebVTT captions, description tracks and the Able Player page (-cc)
├── descriptive-transcript-files/ # Dialogue plus visual descriptions (-cc)
└── video-files/                  # Infographic + audio videos, and the captioned, described video (-cc)
```

## About the explainer video

The video at the top of this page was made from code in [`scripts/docs/explainer/build.py`](https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs/blob/main/scripts/docs/explainer/build.py). The slides are HTML rendered by headless Chrome, so the text on screen is exact. The narration is read one sentence at a time by Gemini TTS (`gemini-3.8-flash-tts`, voice Kore), so each caption is timed to its sentence exactly. To change the video, edit the slides or narration in that file and run:

```bash
uv run --extra all scripts/docs/explainer/build.py
```
