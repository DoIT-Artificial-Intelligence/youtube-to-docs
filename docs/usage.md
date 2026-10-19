# Usage Guide

`youtube-to-docs` is a versatile tool designed to convert YouTube content into structured documentation, including transcripts, summaries, audio, and infographics. It is primarily designed as a Command Line Interface (CLI) tool but can also be used as a Python library.

## Installation

The recommended way to run `youtube-to-docs` is using `uvx`.

### Basic Usage

For basic usage (YouTube data fetching, local CSV output, standard Gemini models):

```bash
uvx youtube-to-docs --help
```

### Optional Features (Extras)

To keep the installation footprint small, many features are optional. You can enable them by installing specific "extras":

| Extra       | Description                                                              | Dependencies                                                                               |
| :---------- | :----------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- |
| `audio`     | **Recommended.** Required for downloading audio and video from YouTube (for transcription and `-cc`). | `yt-dlp[default]` (includes `yt-dlp-ejs`), `deno` (the JavaScript runtime yt-dlp needs for YouTube) |
| `video`     | Required for generating video files (combining audio & infographic).     | `static-ffmpeg`                                                                            |
| `workspace` | Required for saving to Google Drive.                                     | `google-auth-oauthlib`, `pypandoc`, `pypandoc_binary`                                      |
| `m365`      | Required for saving to SharePoint/OneDrive.                              | `msal`, `fastexcel`, `xlsxwriter`, `pypandoc`, `pypandoc_binary`                           |
| `aws`       | AWS Bedrock support.                                                     | `boto3`, `botocore[crt]`                                                                   |
| `azure`     | Required if using Azure OpenAI models.                                   | `openai`                                                                                   |
| `gcp`       | Required if using Google Gemini, Gemma, or Vertex AI models.             | `google-genai`, `google-cloud-speech`, `google-cloud-storage`, `google-cloud-texttospeech`, `google-cloud-translate` |
| `app`       | Web app with browser-based UI.                                           | `fastapi`, `uvicorn`                                                                       |
| `all`       | Installs all of the above.                                               | All optional dependencies.                                                                 |

**How to use extras with `uvx`:**

Use the `--with` flag followed by the package name and extras in brackets.

**Example 1: Audio and Video support (Common)**

```bash
uvx --with "youtube-to-docs[audio,video]" youtube-to-docs ...
```

**Example 2: Google Drive + Audio support**

```bash
uvx --with "youtube-to-docs[workspace,audio]" youtube-to-docs ...
```

**Example 3: Everything (Full feature set)**

```bash
uvx --with "youtube-to-docs[all]" youtube-to-docs ...
```

### Installing `uv`

If you don't have `uv` installed, you can install it using the following commands:

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Setup

Before running the tool, ensure your environment is correctly configured with the necessary API keys and authentication files.

### 1. Environment Variables

Set the following environment variables based on the AI providers you intend to use.

| Variable                   | Description                                      | Required For                                                                            |
| :------------------------- | :----------------------------------------------- | :-------------------------------------------------------------------------------------- |
| `YOUTUBE_DATA_API_KEY`     | API key for the YouTube Data API v3.             | Fetching video metadata.                                                                |
| `GEMINI_API_KEY`           | API key for Google Gemini/Gemma models.          | Gemini/Gemma models (`-m gemini...`, `-m gemma...`).                                    |
| `PROJECT_ID`               | Google Cloud Project ID.                         | GCP Vertex models (`-m vertex...`), GCP STT (`-t gcp...`) and GCP TTS (`--tts gcp...`). |
| `YTD_GCS_BUCKET_NAME`      | Google Cloud Storage bucket name (write access). | GCP STT models (`-t gcp...`) for temp audio storage.                                    |
| `AWS_BEARER_TOKEN_BEDROCK` | AWS Bearer Token.                                | AWS Bedrock models (`-m bedrock...`).                                                   |
| `AZURE_FOUNDRY_ENDPOINT`   | Azure Foundry Endpoint URL.                      | Azure Foundry models (`-m foundry...`).                                                 |
| `AZURE_FOUNDRY_API_KEY`    | Azure Foundry API Key.                           | Azure Foundry models (`-m foundry...`).                                                 |

### 2. Storage Authentication (Optional)

If you plan to save outputs to Google Drive (`workspace`) or Microsoft SharePoint/OneDrive (`sharepoint`), you need to configure authentication files in your home directory.

#### Google Drive (Workspace)

Create a file at `~/.google_client_secret.json` with your Google Cloud OAuth 2.0 Client Secret JSON.

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

- **First Run**: The tool will open a browser window to authenticate and generate a `~/.google_client_token.json` file for future non-interactive use.

#### Microsoft 365 (SharePoint/OneDrive)

Create a file at `~/.azure_client.json` with your Azure App Registration details.

```json
{
  "client_id": "YOUR_CLIENT_ID",
  "authority": "https://login.microsoftonline.com/consumers"
}
```

- **Authority**: Use `.../consumers` for personal accounts or `.../YOUR_TENANT_ID` for organizational accounts.
- **First Run**: The tool will attempt to authenticate (silently or interactively) and cache the token in `~/.msal_token_cache.json`.

## Command Line Interface (CLI)

The main command is `youtube-to-docs`.

### Basic Usage

Running the command without arguments processes a default video:

```bash
youtube-to-docs
```

### Arguments

| Argument                               | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Default                                      | Example                                                                         |
| :------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------- | :------------------------------------------------------------------------------ |
| `video_id`                             | The YouTube content to process. Can be a **YouTube URL**, **Video ID**, **Playlist ID** (starts with `PL`), **Channel Handle** (starts with `@`), or a **comma-separated list** of Video IDs.                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `atmGAHYpf_c`                                | `youtube-to-docs @mychannel`                                                    |
| `-o`, `--outfile`                      | Path to save the output CSV file. <br> - Local path: `my-data.csv` <br> - Google Workspace: `workspace` or `w` (saves to Drive folder `youtube-to-docs-artifacts`) or a specific Folder ID. <br> - SharePoint/OneDrive: `sharepoint` or `s` (saves to `youtube-to-docs-artifacts`). <br> - Memory: `memory` or `m` (keeps artifacts in memory, no files on disk). <br> - No-op: `none` or `n` (skips saving to a file, results are printed to the console).                                                                                                                                                                                                                                                                               | `youtube-to-docs-artifacts/youtube-docs.csv` | `-o n`                                                                          |
| `-t`, `--transcript`                   | The transcript source to use. Can be `'youtube'` (default) to fetch existing YouTube transcripts, or an AI model name to perform STT on extracted audio (e.g. `gemini-3.5-transcribe` for Google's dedicated STT model, `gemini...` for the Gemini API, `gcp-chirp3` for GCP Speech-to-Text V2, `aws-transcribe` for AWS Transcribe). `gemini-3.5-transcribe` returns word-level timestamps and speaker diarization, so its SRT has real offsets and `[Speaker N]` labels on speaker changes. When `-cc` corrects an STT transcript, that model becomes the default so it is transcribed in the same run.                                                                                                                                                                                                                                                                                                                                                                                                                          | `youtube`                                    | `-t gemini-3.5-transcribe`                                                       |
| `-m`, `--model`                        | The LLM(s) to use for speaker extraction, Q&A generation, tag generation, and summarization. Supports models from Google (Gemini), Vertex AI, AWS Bedrock, and Azure Foundry. **Can be a comma-separated list.**                                                                                                                                                                                                                                                                                                                                                                                                                                            | `None` (Transcript only)                     | `-m gemini-3.5-flash-lite,vertex-claude-haiku-5-5`                    |
| `--tts`                                | The TTS model and voice to use for generating audio summaries. Format: `{model}-{voice}`. Supports Gemini models (e.g., `gemini-3.8-flash-lite-tts-Kore`, `gemini-3.8-flash-lite-tts-Kore`) and GCP Cloud TTS (e.g., `gcp-chirp3-Kore`).                                                                                                                                                                                                                                                         | `None`                                       | `--tts gemini-3.8-flash-lite-tts-Kore`                                      |
| `-i`, `--infographic`                  | The image model to use for generating a visual summary. Supports Google Gemini image models (Nano Banana). **Can be a comma-separated list** (one infographic per model).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `None`                                       | `--infographic gemini-nano-banana-2.1,gemini-3-pro-image`              |
| `--alt-text-model`                     | The LLM model to use for generating multimodal alt text for the infographic. Defaults to the summary model.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `None`                                       | `--alt-text-model gemini-3.5-flash-lite`                                       |
| `-nys`, `--no-youtube-summary`         | If set, skips generating a secondary summary from the YouTube transcript when using an AI model for the primary transcript.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `False`                                      | `--no-youtube-summary`                                                          |
| `-fv`, `--from-video`                  | Also generate the summary, one sentence summary, speakers and Q&A directly from the video with Gemini video understanding (no download needed). Value is the processing mode: `agentic` (the model navigates the video, loading only what it needs; Gemini 3.5 Flash-Lite and later, e.g. `gemini-3.8-flash`, `gemini-4-argon`) or `static` (frames at 1 FPS; any Gemini model). Use `agentic,static` to compare. Outputs are labelled `from video-agentic` / `from video-static`.                                                                                                                                                                                     | `None`                                       | `-fv agentic`                                                                   |
| `-tr`, `--translate`                   | Translate all outputs to a target language after generating in English. Format: `{model}-{language}` e.g. `gemini-3.5-flash-lite-es`, or `aws-translate-{language}` / `gcp-translate-{language}` to use AWS Translate or Google Cloud Translation directly (e.g. `aws-translate-es`, `gcp-translate-es`). The tool first tries to fetch a native YouTube transcript in the target language; if unavailable, it translates the English transcript. Summaries, Q&A, tags, one-sentence summaries, transcripts, and SRT files are all translated. When combined with `--tts` or `--infographic`, assets are produced in both English and the target language. | `None`                                       | `-tr gemini-3.5-flash-lite-es`, `-tr aws-translate-es`, `-tr gcp-translate-es` |
| `-cia`, `--combine-infographic-audio`  | Combine the infographic and audio summary into a video file (MP4). Requires both `--tts` and `--infographic` to be effective. When used with `--translate`, one video is created per language.                                                                                                                                                                                                                                                                                                                                                                                                                                                              | `False`                                      | `--combine-infographic-audio`                                                   |
| `--all`                                | Shortcut to use a specific model suite for everything. Supported: `'gemini-flash'`, `'gemini-pro'`, `'gemini-flash-pro-image'`, `'gcp-pro'`, `'anthropic-opus'`. Sets models for summary, TTS, and infographic, and enables `--no-youtube-summary`.                                                                                                                                                                                                                                                                                                                                                                                                                             | `None`                                       | `--all gemini-flash`                                                            |
| `-cc`, `--corrected-captions`          | Make the video accessible (WCAG 2.1 AA / [Section 508](https://www.section508.gov/create/captions-transcripts/)): corrected captions (SRT + WebVTT) with speakers named, audio description (WebVTT tracks + descriptive transcript), the original video, a copy with captions burned in and descriptions narrated, and an Able Player test page. Format: `{model}` or `{model}-{source}`; without a value uses `gemini-4-argon-gemini-3.5-transcribe`. See [Corrected Captions and Audio Description](#corrected-captions-and-audio-description).                                                                                                           | `None`                                       | `-cc`                                                                           |
| `-pp`, `--post-process`                | Post-process the transcript with JSON operations. Supported operations: `word count` (case-insensitive, whole-word). Values can be a single string or a list. Results are added as new columns in the output CSV (e.g. `Post-process: word count(apple)`).                                                                                                                                                                                                                                                                                                                                                                                                  | `None`                                       | `-pp '{"word count": ["apple", "banana"]}'`                                     |
| `--verbose`                            | Enable verbose output.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `False`                                      | `--verbose`                                                                     |

### Examples

**1. Summarize the default video using a single model:**

```bash
youtube-to-docs
```

**2. Generate a transcript using Gemini 2.0 Flash and summarize (using default model):**

```bash
youtube-to-docs -t gemini-2.0-flash-exp
```

**3. Process the default video and save to a custom CSV:**

```bash
youtube-to-docs -o my-docs.csv
```

**4. Summarize a Playlist using multiple models (Gemini and Vertex):**

```bash
youtube-to-docs PLGKTTEqwhiHHWO-jdxM1KtzTbWo6h0Ycl -m gemini-3.5-flash-lite,vertex-claude-haiku-5-5
```

**5. Process a Channel with Summaries, TTS, and Infographics:**

```bash
youtube-to-docs @mga-othercommittees6625 -m vertex-claude-haiku-5-5 --tts gemini-3.8-flash-lite-tts-Kore --infographic gemini-nano-banana-2.1
```

**6. Compare infographics from two image models:**

```bash
youtube-to-docs atmGAHYpf_c -m gemini-3.5-flash-lite --infographic gemini-nano-banana-2.1,gemini-3-pro-image
```

**7. Create a Video (Infographic + Audio Summary):**

```bash
youtube-to-docs atmGAHYpf_c -m gemini-3.5-flash-lite --tts gemini-3.8-flash-lite-tts-Kore --infographic gemini-nano-banana-2.1 --combine-infographic-audio
```

**8. Translate all outputs to Spanish:**

```bash
youtube-to-docs atmGAHYpf_c -m gemini-3.5-flash-lite -tr gemini-3.5-flash-lite-es
```

**9. Full multilingual pipeline — English + Spanish summaries, audio, infographics, and videos:**

```bash
youtube-to-docs atmGAHYpf_c -m gemini-3.5-flash-lite -tr gemini-3.5-flash-lite-es --tts gemini-3.8-flash-lite-tts-Kore --infographic gemini-nano-banana-2.1 --combine-infographic-audio
```

**10. Make the video accessible: corrected captions, audio description and accessible videos (transcribes with `gemini-3.5-transcribe`, corrects and describes with `gemini-4-argon`):**

```bash
youtube-to-docs atmGAHYpf_c -cc
```

**11. Correct the YouTube captions instead (no transcription):**

```bash
youtube-to-docs atmGAHYpf_c -cc gemini-4-argon-youtube
```

**12. Correct an existing GCP Chirp 3 transcript with a cheaper model:**

```bash
youtube-to-docs atmGAHYpf_c -t gcp-chirp3 -cc gemini-3.5-flash-lite-gcp-chirp3
```

**13. Post-process a transcript to count word occurrences:**

```bash
youtube-to-docs atmGAHYpf_c -pp '{"word count": ["apple", "banana"]}'
```

**14. Transcribe with Gemini 3.5 Transcribe (word timestamps + speaker diarization):**

```bash
youtube-to-docs atmGAHYpf_c -t gemini-3.5-transcribe
```

The resulting SRT is built from the model's word-level offsets and carries a
`[Speaker N]` label wherever the speaker changes. Requires `GEMINI_API_KEY`.

## Corrected Captions and Audio Description

`-cc` / `--corrected-captions` makes a video accessible under WCAG 2.1 Level AA (1.2.2 Captions, 1.2.5 Audio Description) and [Section 508](https://www.section508.gov/create/captions-transcripts/), following the DCMP Captioning and Description Keys.

```bash
youtube-to-docs VIDEO_ID -cc
```

Without a value it uses `gemini-4-argon-gemini-3.5-transcribe`: the audio is transcribed with `gemini-3.5-transcribe` (word timestamps and speaker diarization) unless `-t` is set, and `gemini-4-argon` reads on-screen context from the video, corrects the captions and describes the visual information. A 3-hour meeting takes about 20 minutes and roughly $3 in API calls.

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

### Source argument rules

The format is `-cc {model}` or `-cc {model}-{source}`.

| Source         | Behaviour                                                                                                                             | Example                                |
| :------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------- |
| STT model name | Corrects the SRT from that STT run, transcribing it in the same run unless `-t` is set.                                                | `-cc gemini-4-argon-gemini-3.5-transcribe` |
| `youtube`      | Corrects the YouTube SRT, using any STT SRT for the video as the second transcript.                                                   | `-cc gemini-4-argon-youtube`            |
| Omitted        | Corrects the `-t` STT SRT, else the most recent AI SRT, else the YouTube SRT.                                                         | `-cc gemini-4-argon`                    |

If the requested source SRT doesn't exist (for example, `-t youtube -cc`), the available SRT is corrected instead, with a warning. Outputs that already exist aren't regenerated; delete them to run again. Parallel requests default to 6; set `YTD_CC_WORKERS` to lower this if you hit rate limits.

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

## CSV Column Reference

The output CSV file contains a variety of columns depending on the arguments provided. Below is a reference of the possible columns:

### Base Metadata

- **URL**: The full YouTube video URL.
- **Title**: The title of the video.
- **Description**: The video description.
- **Data Published**: The date the video was published.
- **Channel**: The name of the YouTube channel.
- **Tags**: Video tags from YouTube (comma-separated).
- **Tags {Transcript} {Model} model**: Up to 5 AI-generated tags based on the transcript.
- **Duration**: The duration of the video.
- **Transcript characters from youtube**: The total number of characters in the YouTube transcript.
- **Transcript characters from {model}**: The total number of characters in the AI-generated transcript (if applicable).
- **Audio File**: Path to the extracted audio file (used for AI transcription).

### Files

- **Transcript File {type}**: Path to the saved transcript file. `{type}` is either `youtube generated`, `human generated`, or `{model} generated`. Suffix `(lang)` added for non-English.
- **SRT File {type}**: Path to the saved SRT transcript file. Standardized timestamps for accessibility and timing.
- **Speakers File {model}**: Path to the saved speaker extraction text file.
- **QA File {model}**: Path to the saved Q&A Markdown file. Includes **Timestamp** and **Timestamp URL** (markdown hyperlink) columns. Suffix `(lang)` added for non-English.
- **Summary File {model}**: Path to the Markdown summary file generated by a specific model. Suffix `(lang)` added for non-English.
- **One Sentence Summary File {model}**: Path to the one-sentence summary file.
- **Tags File {model}**: Path to the AI-generated tags file.
- **Summary Infographic File {model} {infographic_model}**: Path to the generated infographic image.
- **Summary Infographic Alt Text File {model} {infographic_model}**: Path to the generated alt text file.
- **Summary Audio File {model} {tts_model} File**: Path to the generated TTS audio file. Suffix `(lang)` added for non-English.
- **Video File**: Path to the generated MP4 video combining the English infographic and audio.
- **Video File (lang)**: Path to the translated language video (e.g., `Video File (es)`). Only present when `--translate` and `--combine-infographic-audio` are both used.
- **Corrected Captions File {model} from {source}**, **Captions VTT File {model} from {source}**: The corrected captions as SRT and WebVTT (`-cc`).
- **Descriptions VTT File {model}**, **Extended Descriptions VTT File {model}**, **Descriptive Transcript File {model}**: The audio description (`-cc`).
- **Original Video File**, **Captioned And Described Video File {model} from {source}**, **Player Page**: The original video, the copy with captions burned in and descriptions narrated, and the Able Player page (`-cc`).
- **Post-process: word count({word})**: Count of occurrences of `{word}` in the transcript (case-insensitive, whole-word match). One column per word when using `-pp`.

### AI Outputs & Costs

- **Speakers {model}**: The extracted list of speakers and their roles.
- **{normalized_model} Speaker extraction cost ($)**: The estimated API cost for speaker extraction.
- **QA Text {model}**: The full text of the Q&A pairs (also saved to the Q&A file).
- **{normalized_model} QA cost ($)**: The estimated API cost for Q&A generation.
- **Summary Text {model}**: The full text of the summary (also saved to the summary file).
- **One Sentence Summary {model}**: The one-sentence summary text.
- **Summary Infographic Alt Text {model} {infographic_model}**: The full multimodal alt text for the infographic.
- **{normalized_model} summary cost ($)**: The total estimated API cost for both speaker extraction and summarization.
- **Summary Infographic Cost {model} {infographic_model} ($)**: The estimated API cost for infographic generation.
- **Summary Infographic Alt Text Cost {model} {infographic_model} ($)**: The estimated API cost for alt text generation.
- **{normalized_model} tags cost from {transcript} ($)**: The estimated API cost for AI tag generation.
- **{normalized_model} STT cost ($)**: The estimated API cost for Speech-to-Text generation. For `gemini-3.5-transcribe` the API reports audio input tokens but not transcript output tokens, so this figure covers the input side only and understates the true cost.
- **{normalized_model} corrected captions cost from {source} ($)**: The estimated API cost of correcting the captions with `-cc` (speaker identification and text correction, including thinking tokens). The on-screen context request and the audio description are recorded separately as `{normalized_model} video context cost from video-agentic ($)` and `{normalized_model} audio description cost ($)` (narration not included).

> **Note**: `{normalized_model}` refers to the model name with prefixes (like `vertex-`) and date suffixes removed for cleaner column headers.

## Speaker Extraction

When a model is specified using the `-m` or `--model` argument, the tool automatically performs speaker extraction before generating a summary.

- **Model Matching**: The extraction uses the same model as the summary. If multiple models are provided, each will perform its own extraction.
- **Structured Output**: It identifies speakers and their professional titles or roles (e.g., "Speaker 1 (Senator Katie Fry Hester, Co-Chair)").
- **Cost Tracking**: The cost of speaker extraction is tracked separately in the `{model} Speaker extraction cost ($)` column and included in the total `{model} summary cost ($)`.
- **Unknowns**: If a speaker or title cannot be identified, the tool uses the placeholder `UNKNOWN`. If no speakers are detected at all, the field is set to `NaN`.

## Web App

`youtube-to-docs` includes a browser-based web interface powered by FastAPI. It provides the same functionality as the CLI in an interactive UI with real-time log streaming.

### Running the Web App

```bash
uvx --with "youtube-to-docs[app]" youtube-to-docs-app
```

This starts a local server at [http://localhost:8000](http://localhost:8000).

#### Options

| Flag       | Description                  | Default     |
| :--------- | :--------------------------- | :---------- |
| `--host`   | Host to bind to. Use `0.0.0.0` to expose on your network (the app has no authentication). | `127.0.0.1` |
| `--port`   | Port to bind to.             | `8000`      |
| `--reload` | Enable auto-reload for dev.  | `False`     |

**Example: Run on a custom port with auto-reload**

```bash
uvx --with "youtube-to-docs[app]" youtube-to-docs-app --port 3000 --reload
```

### Features

- **Model suite dropdown** — Select from pre-configured suites (gemini-flash, gemini-pro, etc.) or configure individual models.
- **All CLI parameters** — Transcript source, TTS, infographic, translation, corrected captions, post-processing, and output file are all configurable.
- **Real-time log streaming** — Processing output streams to the browser via Server-Sent Events (SSE) as the job runs.
- **Artifact browser** — After processing completes, generated files (summaries, Q&A, SRT files, infographics, audio, video) are listed with download links.

### How It Works

1. Enter a YouTube URL or Video ID and select your options.
2. Click **Process Video** — this starts a background job on the server.
3. Logs stream to the browser in real time.
4. When the job completes, any generated artifacts are listed with links to view or download them.

The web app builds CLI arguments internally and calls the same `main()` function as the CLI, so behavior is identical. Jobs run in background threads, allowing the server to handle multiple requests.

### API Endpoints

The web app exposes a REST API that the frontend uses:

| Method | Endpoint                     | Description                              |
| :----- | :--------------------------- | :--------------------------------------- |
| GET    | `/`                          | Serves the HTML frontend.                |
| GET    | `/api/model-suites`          | Returns available model suite configs.   |
| POST   | `/api/process`               | Starts a processing job. Returns job ID. |
| GET    | `/api/jobs/{job_id}`         | Returns job status, output, artifacts.   |
| GET    | `/api/jobs/{job_id}/stream`  | SSE stream of real-time log output.      |
| GET    | `/api/artifacts/{path}`      | Serves a generated artifact file.        |

## MCP Server

This tool also works as a Model Context Protocol (MCP) server, so AI agents can use it as a tool. The server exposes a `process_video` tool that mirrors the CLI.

### Install as a plugin

The repository is a plugin for Claude Code and Antigravity. It adds the MCP server (defined in `mcp_config.json`) and one `youtube-to-docs` skill that tells the agent how to call it.

Claude Code:

```bash
claude plugin marketplace add DoIT-Artificial-Intelligence/youtube-to-docs
claude plugin install youtube-to-docs@youtube-to-docs
```

Antigravity:

```bash
agy plugin install https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs
```

### Usage

Ask for what you want in plain language:

> "Save a summary of https://www.youtube.com/watch?v=KuPc06JgI_A"

The agent asks for any details it needs (like the model to use) and then calls the tool. You can also run the skill directly, e.g. `/youtube-to-docs <url> kitchen sink gemini pro spanish` or `/youtube-to-docs <url> captions`.

### Other MCP clients

Any MCP client can run the server with:

```bash
uv run --no-project --with "youtube-to-docs[all]" python -m youtube_to_docs.mcp_server
```

## Library Usage

While primarily a CLI, you can import core functions for custom workflows.

```python
from youtube_to_docs.transcript import fetch_transcript

video_id = "atmGAHYpf_c"
transcript, is_generated = fetch_transcript(video_id)
```

## Why use `youtube-to-docs`?

You might find other tools that download YouTube transcripts, but `youtube-to-docs` distinguishes itself in several ways:

1.  **Multimodal Output**: It doesn't just stop at text.
    - **Summaries**: Uses state-of-the-art LLMs to create concise summaries.
    - **Speaker Extraction**: Automatically identifies speakers and their titles/roles from the transcript.
    - **Audio (TTS)**: Converts summaries into audio files, perfect for listening on the go.
    - **Tags**: Automatically generates up to 5 relevant tags for the video content.
    - **Visuals (Infographics)**: Generates AI-created infographics to visually represent the content.
    - **Timestamps & SRT**: Automatically generates `.srt` files and includes precision timestamps in Q&A tables, cross-referencing YouTube's own timing data for pinpoint accuracy when using AI transcripts.
    - **Videos**: Combines infographics and audio into a shareable video summary.

2.  **Structured Data (CSV/Polars)**:
    - Instead of loose files, metadata and paths are organized into a robust CSV file using `polars`.
    - This makes it incredibly easy to import the data into Google Sheets, Excel, or a database for further analysis or publishing.

3.  **Batch Processing**:
    - Seamlessly handles individual videos, entire playlists, or full channels with a single command.

4.  **Multi-Provider Support**:
    - Agnostic to the LLM provider. Whether you use Google Gemini, Vertex AI, AWS Bedrock, or Azure Foundry, you can plug in your preferred model.

5.  **Cost Awareness**:
    - When using paid API models, it tracks and estimates the cost of summarization, saving it directly to your data file.
