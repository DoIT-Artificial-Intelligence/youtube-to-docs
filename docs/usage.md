# Usage Guide

`youtube-to-docs` turns a YouTube video, playlist or channel into documents: a transcript, speakers, Q&A, summaries, tags, an infographic, an audio summary and, with `-cc`, accessible captions and audio description. It's a command-line tool, and it also runs as a web app, an MCP server and a Python library.

## Quick start

1. Install [`uv`](https://docs.astral.sh/uv/):

    === "macOS/Linux"

        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    === "Windows"

        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

2. Set `YOUTUBE_DATA_API_KEY` and `GEMINI_API_KEY` (see [Environment variables](#environment-variables)).
3. Run it on a video:

    ```bash
    uvx --with "youtube-to-docs[all]" youtube-to-docs https://www.youtube.com/watch?v=atmGAHYpf_c -m
    ```

This saves the transcript, speakers, Q&A, tags and summaries to `youtube-to-docs-artifacts/`, and lists them in `youtube-to-docs-artifacts/youtube-docs.csv`.

## Installation

Run the tool with `uvx`, so there's nothing to install apart from `uv`:

```bash
uvx youtube-to-docs --help
```

The base package fetches YouTube transcripts and metadata and saves to local files. Other features are optional extras. Add them with `--with`:

```bash
uvx --with "youtube-to-docs[audio,video]" youtube-to-docs ...
uvx --with "youtube-to-docs[all]" youtube-to-docs ...
```

| Extra | Needed for | Installs |
| :--- | :--- | :--- |
| `audio` | **Recommended.** Downloading audio and video from YouTube (AI transcription, `-cc`) | `yt-dlp[default]`, `deno` |
| `video` | Making video files (`-cia`, `-cc`) | `static-ffmpeg` |
| `gcp` | Gemini, Gemma and Vertex AI models, and Google Cloud speech, TTS and translation | `google-genai`, `google-cloud-speech`, `google-cloud-storage`, `google-cloud-texttospeech`, `google-cloud-translate` |
| `aws` | AWS Bedrock, Transcribe, Polly and Translate | `boto3`, `botocore[crt]` |
| `azure` | Azure Foundry models | `openai` |
| `workspace` | Saving to Google Drive | `google-auth-oauthlib`, `pypandoc`, `pypandoc_binary` |
| `m365` | Saving to SharePoint or OneDrive | `msal`, `fastexcel`, `xlsxwriter`, `pypandoc`, `pypandoc_binary` |
| `huggingface` | Saving to a Hugging Face dataset | `huggingface_hub` |
| `app` | The [web app](#web-app) | `fastapi`, `uvicorn` |
| `all` | All of the above | |

## Setup

### Environment variables

Set the variables for the providers you use.

| Variable | Needed for |
| :--- | :--- |
| `YOUTUBE_DATA_API_KEY` | Video metadata, playlists and channels (YouTube Data API v3) |
| `GEMINI_API_KEY` | Gemini and Gemma models (`gemini-…`, `gemma-…`), including `gemini-3.5-transcribe` and Gemini TTS |
| `PROJECT_ID` | Vertex AI models (`vertex-…`). `VERTEX_LOCATION` sets the region (default `global` for newer Claude models, otherwise `us-east5`). |
| `GOOGLE_CLOUD_PROJECT` | GCP Speech-to-Text (`-t gcp-chirp3`). `YTD_GCS_BUCKET_NAME` sets the bucket for temporary audio (default `youtube-to-docs`). |
| `AWS_BEARER_TOKEN_BEDROCK` | AWS Bedrock models (`bedrock-…`) |
| `YTD_S3_BUCKET_NAME` | AWS Transcribe (`-t aws-transcribe`), for temporary audio |
| `AZURE_FOUNDRY_ENDPOINT`, `AZURE_FOUNDRY_API_KEY` | Azure Foundry models (`foundry-…`) |
| `HF_TOKEN` | Saving to Hugging Face (`-o hf`) |
| `YTD_CC_WORKERS` | Optional. Parallel `-cc` requests (default 6). Lower it if you hit rate limits. |

Google Cloud services (Speech-to-Text, Cloud TTS, Translation) use [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials), and AWS services other than Bedrock use your usual AWS credentials.

### Google Drive and SharePoint sign-in

You only need this to save outputs to Google Drive (`-o workspace`) or SharePoint/OneDrive (`-o sharepoint`).

**Google Drive.** Save your Google Cloud OAuth 2.0 client secret as `~/.google_client_secret.json`:

```json
{
  "installed": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uris": ["http://localhost"]
  }
}
```

The first run opens a browser to sign in and saves `~/.google_client_token.json` for later runs.

**SharePoint/OneDrive.** Save your Azure app registration as `~/.azure_client.json`:

```json
{
  "client_id": "YOUR_CLIENT_ID",
  "authority": "https://login.microsoftonline.com/consumers"
}
```

Use `.../consumers` for personal accounts or `.../YOUR_TENANT_ID` for an organization. The first run signs you in and caches the token in `~/.msal_token_cache.json`.

## Command line

```bash
youtube-to-docs [VIDEO] [OPTIONS]
```

`VIDEO` is a YouTube URL, a video ID, a playlist ID (starts with `PL`), a channel handle (starts with `@`) or a comma-separated list of video IDs. Without it, the tool processes the sample video `atmGAHYpf_c`.

With no options, you get the transcript only. Add options for the outputs you want. Options marked *bare* can be used without a value to get the default shown.

### Options

| Option | What it does | Default |
| :--- | :--- | :--- |
| `-m`, `--model` | Speakers, Q&A, tags and summaries with an LLM. Comma-separate to compare models. | Bare: `gemini-3.5-flash-lite` |
| `-t`, `--transcript` | Where the transcript comes from: `youtube`, or a speech-to-text model such as `gemini-3.5-transcribe`, `gcp-chirp3` or `aws-transcribe`. | `youtube` |
| `-i`, `--infographic` | An infographic, with alt text, from an image model. Comma-separate for one per model. | Bare: `gemini-3-pro-image` |
| `--alt-text-model` | The LLM that writes the infographic alt text. | The `-m` model |
| `--tts` | An audio summary. Format: `{model}-{voice}`, e.g. `gemini-3.8-flash-lite-tts-Kore`, `gcp-chirp3-Kore` or `aws-polly-Ruth`. | None |
| `-cia`, `--combine-infographic-audio` | An MP4 that plays the audio summary over the infographic. Needs `-i` and `--tts`. | Off |
| `-tr`, `--translate` | Translate every output. Format: `{model}-{language}`, e.g. `gemini-3.5-flash-lite-es`, `aws-translate-es` or `gcp-translate-french`. | None |
| `-cc`, `--corrected-captions` | Corrected captions and audio description. See [Corrected captions and audio description](#corrected-captions-and-audio-description). | Bare: `gemini-4-argon-gemini-3.5-transcribe` |
| `-fv`, `--from-video` | Also make the summary, speakers and Q&A from the video itself with Gemini: `agentic`, `static` or `agentic,static`. | None |
| `-nys`, `--no-youtube-summary` | With an AI transcript, skip the extra summary of the YouTube transcript. | Off |
| `-a`, `--all` | Use a model suite for summary, TTS and infographic (and set `-nys`): `gemini-flash`, `gemini-pro`, `gemini-flash-pro-image`, `gcp-pro` or `anthropic-opus`. | None |
| `-pp`, `--post-process` | Extra columns from the transcript. Supports `word count`, e.g. `'{"word count": ["apple", "banana"]}'`. | None |
| `-o`, `--outfile` | Where to save: a CSV path, `workspace` (`w`) or a Drive folder ID, `sharepoint` (`s`), `hf`, `memory` (`m`) or `none` (`n`). | `youtube-to-docs-artifacts/youtube-docs.csv` |
| `--hugging-face-dataset` | The dataset for `-o hf`: a repo ID (`username/my-dataset`) or a name. | The playlist title |
| `--verbose` | More detailed logs. | Off |
| `-V`, `--version` | Show the version. | |

`-i`, `-cc` and `-fv` turn on `-m gemini-3.5-flash-lite` if you don't set `-m`. Model names and prices are on the [Pricing](pricing.md) page, and speed and cost comparisons are on the [Performance](performance.md) page.

### Examples

**Transcript only:**

```bash
youtube-to-docs atmGAHYpf_c
```

**Summaries, Q&A, speakers and tags with the default model:**

```bash
youtube-to-docs atmGAHYpf_c -m
```

**Compare two models across a playlist:**

```bash
youtube-to-docs PLGKTTEqwhiHHWO-jdxM1KtzTbWo6h0Ycl -m gemini-3.5-flash-lite,vertex-claude-haiku-5-5
```

**Transcribe the audio with Gemini 3.5 Transcribe** (word timestamps and speaker labels in the SRT):

```bash
youtube-to-docs atmGAHYpf_c -t gemini-3.5-transcribe -m
```

**A whole channel with infographics and audio summaries:**

```bash
youtube-to-docs @mga-othercommittees6625 -m -i --tts gemini-3.8-flash-lite-tts-Kore
```

**A video summary** (the infographic with the audio summary):

```bash
youtube-to-docs atmGAHYpf_c -m -i --tts gemini-3.8-flash-lite-tts-Kore -cia
```

**Everything in English and Spanish:**

```bash
youtube-to-docs atmGAHYpf_c -m -i --tts gemini-3.8-flash-lite-tts-Kore -cia -tr gemini-3.5-flash-lite-es
```

**Compare infographics from Gemini and Azure Foundry:**

```bash
youtube-to-docs atmGAHYpf_c -m -i gemini-3-pro-image,foundry-gpt-image-2.5-sunburst
```

**Accessible captions and audio description:**

```bash
youtube-to-docs atmGAHYpf_c -cc
```

**Save to Google Drive:**

```bash
uvx --with "youtube-to-docs[all]" youtube-to-docs atmGAHYpf_c -m -o workspace
```

**Count words in the transcript:**

```bash
youtube-to-docs atmGAHYpf_c -pp '{"word count": ["apple", "banana"]}'
```

## Corrected captions and audio description

`-cc` / `--corrected-captions` makes a video accessible under WCAG 2.1 Level AA (1.2.2 Captions, 1.2.5 Audio Description) and [Section 508](https://www.section508.gov/create/captions-transcripts/), following the DCMP Captioning and Description Keys.

```bash
youtube-to-docs VIDEO_ID -cc
```

Without a value it uses `gemini-4-argon-gemini-3.5-transcribe`: the audio is transcribed with `gemini-3.5-transcribe` (word timestamps and speaker diarization) unless `-t` is set, and `gemini-4-argon` reads on-screen context from the video, corrects the captions and describes the visual information. A 3-hour meeting takes about 20 minutes and roughly $3 in API calls.

Other ways to run it:

```bash
# Correct the YouTube captions instead of transcribing
youtube-to-docs VIDEO_ID -cc gemini-4-argon-youtube

# Correct a GCP Chirp 3 transcript with a cheaper model
youtube-to-docs VIDEO_ID -t gcp-chirp3 -cc gemini-3.5-flash-lite-gcp-chirp3
```

### Outputs

| Output | Folder | Use |
| :--- | :--- | :--- |
| `… corrected captions (from {source}).srt` | `corrected-caption-files/` | Captions for YouTube or any player |
| `… captions (from {source}).vtt` | `vtt-files/` | Website: `<track kind="captions">` (browsers don't read SRT) |
| `… descriptions.vtt` | `vtt-files/` | Website: `<track kind="descriptions">`, descriptions fitted into pauses (standard audio description, AA) |
| `… extended descriptions.vtt` | `vtt-files/` | Website: every description where its visual appears, for players that pause the video while reading (Able Player does by default; extended audio description, WCAG 1.2.7) |
| `… descriptive transcript.md` | `descriptive-transcript-files/` | The dialogue by speaker with every description; publish next to the video (WCAG 1.2.3 and 1.2.8) |
| `{id} - {title} - original video.mp4` | `video-files/` | The source video for a website player |
| `… captioned and described video (from {source}).mp4` | `video-files/` | A standalone accessible copy: captions burned in (white on an 80% opaque box) and the descriptions narrated in the pauses |
| `… player.html` | `vtt-files/` | An Able Player page with the original video, the WebVTT tracks and an interactive transcript, for checking them (it opens straight from disk) and as a template for a website |

### How it works

1. **Timing.** The source SRT is aligned word by word with a second transcript of the same audio (the YouTube captions, or the STT transcript when correcting YouTube's). Cues with broken timestamps (for example, words returned minutes off) are re-timed from the second transcript.
2. **Speakers.** The speaker of every cue is identified in ~20-minute windows from the diarization labels, the second transcript's speaker changes (`>>`), the times the video shows each person speaking, introductions, hand-offs and roles. Names come from on-screen name plates where available.
3. **Text.** Sections of ~150 cues are corrected in parallel against the second transcript and the on-screen spellings: misheard words, names, organizations and acronyms, punctuation and capitalization, clean verbatim (fillers and stutters removed, false starts marked with em dashes), numbers, and non-speech sounds such as `[laughter]`. Speech missing from the source but present in the second transcript (often short replies) is restored. The model edits text by cue id and never writes timestamps, so long videos can't be truncated or drift out of sync.
4. **Layout.** The corrected words are re-segmented into captions of at most two lines of 42 characters, 1–7 seconds long, broken at sentence and clause boundaries (not inside names or after articles). `[Name]` labels mark every change of speaker, crosstalk is placed at the nearest sentence break, quick exchanges share a two-line caption, and captions never overlap.
5. **Audio description.** The video is downloaded and scanned for scene changes (new slides, changes of shot); the model looks at the frame at each change with the dialogue around it and writes a short spoken version and a full description of what isn't said aloud: slide titles, key on-screen text, the main figures in charts and tables, votes and video clips. Standard audio description is spoken only in pauses, so the short versions are fitted into pauses while their visual is on screen (most important first, shortened if needed), narrated with `--tts` (default `gemini-3.8-flash-tts-Kore`) and mixed in with the programme audio lowered. Nothing is narrated over speech; the extended track and the descriptive transcript carry every description.

Audio description needs a Gemini model; with other models `-cc` produces the captions only.

### Choosing the source

The format is `-cc {model}` or `-cc {model}-{source}`.

| Source | Behaviour | Example |
| :--- | :--- | :--- |
| STT model name | Corrects the SRT from that STT run, transcribing it in the same run unless `-t` is set. | `-cc gemini-4-argon-gemini-3.5-transcribe` |
| `youtube` | Corrects the YouTube SRT, using any STT SRT for the video as the second transcript. | `-cc gemini-4-argon-youtube` |
| Omitted | Corrects the `-t` STT SRT, else the most recent AI SRT, else the YouTube SRT. | `-cc gemini-4-argon` |

If the requested source SRT doesn't exist (for example, `-t youtube -cc`), the available SRT is corrected instead, with a warning.

### Publishing

- **Website:** host the original video with `captions.vtt` and a descriptions track in a player that speaks descriptions, such as [Able Player](https://ableplayer.github.io/ableplayer/); the generated `player.html` shows the markup. Link the descriptive transcript next to the video.
- **YouTube:** upload the captioned and described video, or upload the original with the corrected SRT as its captions (YouTube has no description tracks, so audio description needs the described copy).

### Checking the WebVTT files

1. **Validate them.** Paste a `.vtt` file into the [W3C WebVTT validator](https://w3c.github.io/webvtt.js/parser.html); it should report no errors.
2. **Play them.** Open the `… player.html` file in a browser. Captions are on; select the Description button to hear the descriptions read aloud (the video pauses while each is read), and the Transcript button for an interactive transcript of the dialogue and descriptions. To hear the descriptions through a screen reader instead, add `data-desc-reader="screenreader"` to the `<video>` element and turn on NVDA, JAWS or VoiceOver.

    The page works straight from disk: browsers won't load `.vtt` files into a page opened that way, so it carries copies of the tracks and uses those. To load the `.vtt` files themselves, as a website would, serve the artifacts folder with a server that supports range requests (Python's built-in `http.server` doesn't, so the browser can't seek in a long video):

    ```bash
    cd youtube-to-docs-artifacts
    uv run --no-project --with rangehttpserver python -m RangeHTTPServer 8080 --bind 127.0.0.1
    ```

    Then open `http://127.0.0.1:8080/vtt-files/` and select the `… player.html` file.

3. **Spot-check timing.** In VLC, open the original video and add the captions `.vtt` (Subtitle > Add Subtitle File...) to check that captions line up with the speech.

## Review and edit the outputs

youtube-to-docs is human-centered: the models write a first draft, and people review it and have the final say. Every output is a file you can edit (a Google Doc or Word file when saving to Google Drive or SharePoint), and the CSV lists where each one is saved.

Outputs that already exist aren't regenerated. When you run the same video again, the tool reads your edited files and uses them for any step that hasn't run yet. For example:

1. Get the transcript only (the default): `youtube-to-docs <url>`.
2. Fix the transcript in `transcript-files/` (and the matching `.srt` in `srt-files/` if you plan to make captions).
3. Run again with the outputs you want, such as `-m -i --tts gemini-3.8-flash-lite-tts-Kore` or `-cc`. The summary, Q&A, infographic, audio and captions are built from your corrected transcript.

You can also edit a summary, Q&A, speakers or alt text file directly. On the next run, the edited text is loaded back into the CSV, and outputs made later, such as the infographic and audio, use it.

To make an output again, delete its file and remove the video's row from the CSV (or clear that output's columns), then run again. The tool finds your other files, including the edited transcript, by their file names.

## CSV columns

The CSV has one row per video. Which columns appear depends on the options you use. In column names, `{model}` is the model that made the output, and `{normalized_model}` is the model name without prefixes like `vertex-` or date suffixes.

### Video details

- **URL**, **Title**, **Description**, **Data Published**, **Channel**, **Duration**: From YouTube.
- **Tags**: The video's YouTube tags.
- **Tags {transcript} {model} model**: Up to 5 tags generated from the transcript.
- **Transcript characters from youtube**, **Transcript characters from {model}**: The length of each transcript.
- **Audio File**: The downloaded audio, used for AI transcription.

### Files

Paths to the saved outputs. Translated outputs end in the language code, such as `(es)`.

- **Transcript File {type}**: The transcript. `{type}` is `youtube generated`, `human generated` or `{model} generated`.
- **SRT File {type}**: The transcript as SRT captions.
- **Speakers File {model}**, **QA File {model}**, **Summary File {model}**, **One Sentence Summary File {model}**, **Tags File {model}**: The text outputs. The Q&A table includes **Timestamp** and **Timestamp URL** columns that link to the moment in the video.
- **Summary Infographic File {model} {infographic_model}**, **Summary Infographic Alt Text File {model} {infographic_model}**: The infographic and its alt text.
- **Summary Audio File {model} {tts_model} File**: The audio summary.
- **Video File**, **Video File (lang)**: The infographic and audio summary video (`-cia`), in English and the translated language.
- **Corrected Captions File {model} from {source}**, **Captions VTT File {model} from {source}**: The corrected captions as SRT and WebVTT (`-cc`).
- **Descriptions VTT File {model}**, **Extended Descriptions VTT File {model}**, **Descriptive Transcript File {model}**: The audio description (`-cc`).
- **Original Video File**, **Captioned And Described Video File {model} from {source}**, **Player Page**: The original video, the copy with captions burned in and descriptions narrated, and the Able Player page (`-cc`).
- **Post-process: word count({word})**: How many times `{word}` appears in the transcript (case-insensitive, whole words), one column per word (`-pp`).

### Text and costs

- **Speakers {model}**: The speakers and their titles or roles, e.g. "Speaker 1 (Senator Katie Fry Hester, Co-Chair)". `UNKNOWN` marks a name or title that couldn't be found, and `NaN` means no speakers were found.
- **QA Text {model}**, **Summary Text {model}**, **One Sentence Summary {model}**, **Summary Infographic Alt Text {model} {infographic_model}**: The text of each output.
- **{normalized_model} Speaker extraction cost ($)**, **{normalized_model} QA cost ($)**, **{normalized_model} tags cost from {transcript} ($)**: Estimated API cost of each step.
- **{normalized_model} summary cost ($)**: Speaker extraction plus summary.
- **Summary Infographic Cost {model} {infographic_model} ($)**, **Summary Infographic Alt Text Cost {model} {infographic_model} ($)**: Infographic and alt text costs.
- **{normalized_model} STT cost ($)**: Speech-to-text cost. For `gemini-3.5-transcribe` the API reports audio input tokens but not output tokens, so this understates the cost.
- **{normalized_model} corrected captions cost from {source} ($)**: The `-cc` speaker identification and text correction, including thinking tokens. The on-screen context and the audio description are in **{normalized_model} video context cost from video-agentic ($)** and **{normalized_model} audio description cost ($)** (narration not included).

## Web app

The web app runs the same pipeline in your browser, with the logs streamed live and links to every output when it finishes.

```bash
uvx --with "youtube-to-docs[app]" youtube-to-docs-app
```

Then open [http://localhost:8000](http://localhost:8000).

| Flag | What it does | Default |
| :--- | :--- | :--- |
| `--host` | Host to bind to. `0.0.0.0` exposes it on your network, and the app has no sign-in. | `127.0.0.1` |
| `--port` | Port to bind to. | `8000` |
| `--reload` | Reload when the code changes (for development). | Off |

The app builds the same arguments as the CLI and calls the same code, so the results match. It also has a REST API:

| Method | Endpoint | Returns |
| :--- | :--- | :--- |
| GET | `/` | The web page |
| GET | `/api/model-suites` | The `--all` model suites |
| POST | `/api/process` | Starts a job and returns its ID |
| GET | `/api/jobs/{job_id}` | The job's status, output and artifacts |
| GET | `/api/jobs/{job_id}/stream` | The job's logs as Server-Sent Events |
| GET | `/api/artifacts/{path}` | A generated file |

## MCP server

The tool also runs as a Model Context Protocol (MCP) server, so AI agents can call it. Its `process_video` tool takes the same options as the CLI.

The repository is a plugin for Claude Code and Antigravity. It adds the MCP server and a `youtube-to-docs` skill that tells the agent how to use it.

=== "Claude Code"

    ```bash
    claude plugin marketplace add DoIT-Artificial-Intelligence/youtube-to-docs
    claude plugin install youtube-to-docs@youtube-to-docs
    ```

=== "Antigravity"

    ```bash
    agy plugin install https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs
    ```

Then ask in plain language, for example "Save a summary of https://www.youtube.com/watch?v=KuPc06JgI_A". The agent asks for anything it needs, such as the model, and calls the tool. You can also run the skill directly: `/youtube-to-docs <url> kitchen sink gemini pro spanish` or `/youtube-to-docs <url> captions`.

Other MCP clients can start the server with:

```bash
uv run --no-project --with "youtube-to-docs[all]" python -m youtube_to_docs.mcp_server
```

## Python library

You can also import the functions directly:

```python
from youtube_to_docs.transcript import fetch_transcript

# Returns (text, is_generated, segments), or None if there is no transcript
transcript, is_generated, segments = fetch_transcript("atmGAHYpf_c")
```
