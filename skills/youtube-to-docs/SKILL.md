---
name: youtube-to-docs
description: "Process YouTube videos, playlists and channels with the youtube-to-docs MCP server. Use this when the user wants to: (1) get a transcript, (2) summarize a video or generate Q&A, speakers and tags, (3) make an infographic, (4) make an audio or video summary, (5) translate the outputs, (6) make the video accessible (WCAG 2.1 / Section 508 corrected captions, audio description), or (7) do a full 'kitchen sink' run."
argument-hint: "<url> [transcript | infographic | kitchen sink | captions] [gemini flash | gemini pro] [language]"
---

# YouTube to Docs

You have the `process_video` tool from the `youtube-to-docs` MCP server. One call handles a video, a playlist, a channel or a comma-separated list of IDs. Each call writes a row per video to a CSV, with links to the files it creates.

If this skill was run with arguments (`$ARGUMENTS`), take the URL from them, pick the matching workflow below, and call `process_video` without asking for confirmation. Otherwise, work out the workflow from the request. Ask only about what you can't infer:

- **Model**: if the request needs a summary and names no model, ask "What model do you want to use for the summary?"
- **Output location**: if none is given, ask "Where do you want the output saved? Is the default location (`youtube-to-docs-artifacts/youtube-docs.csv`) okay?"
- **Extras**: where it fits, offer an infographic, an audio summary, a video summary, AI transcription, translation, corrected captions, or transcript post-processing.

## Model shorthand

| User says                     | `model`                  | `tts_model`                      | `infographic_model`      | `all_suite`    |
| :---------------------------- | :----------------------- | :------------------------------- | :----------------------- | :------------- |
| "gemini", "gemini flash", "flash" | `gemini-3.5-flash-lite`  | `gemini-3.8-flash-lite-tts-Kore` | `gemini-nano-banana-2.1` | `gemini-flash` |
| "gemini pro", "pro"           | `gemini-4-argon`         | `gemini-3.8-flash-tts-Kore`      | `gemini-3-pro-image`     | `gemini-pro`   |
| "aws"                         | `bedrock-nova-2-lite-v1` | `aws-polly-Ruth`                 | `gemini-3-pro-image`     | -              |
| "gcp"                         | -                        | -                                | -                        | `gcp-pro`      |

## Workflows

### Transcript only

"Get the transcript", "ytt".

`process_video(url='...')` fetches the YouTube captions. No model is needed.

### Summary and infographic

"Make an infographic", "summarize this".

- Flash (the default when no family is given): `model='gemini-3.5-flash-lite'`, `infographic_model='gemini-nano-banana-2.1'`
- Pro: `model='gemini-4-argon'`, `infographic_model='gemini-3-pro-image'`

The tool also writes multimodal alt text for each infographic with the summary model. Set `alt_text_model` to use a different model.

### Kitchen sink

"Everything", "kitchen sink", "ks", "video summary".

- `all_suite`: `'gemini-pro'` by default, `'gemini-flash'` if flash is mentioned, `'gcp-pro'` if gcp is mentioned
- `combine_infographic_audio=True`, `verbose=True`
- Language: omit `translate` for English only. For another language, use its code (spanish → `es`, french → `fr`) in `translate='gemini-3.5-flash-lite-{lang}'`. If "aws translate" or "gcp translate" is mentioned, use `'aws-translate-{lang}'` or `'gcp-translate-{lang}'` instead.

### Accessible video: corrected captions and audio description (WCAG 2.1 / Section 508)

"Fix the captions", "make this compliant", "audio description", "best captions", "cc".

- `corrected_captions='default'` gives the best results (recommended). It runs `gemini-4-argon` on a `gemini-3.5-transcribe` transcript made in the same run, with names and speaking times read from the video.
- If YouTube is named as the source, use `'gemini-4-argon-youtube'`, which starts from the YouTube captions and skips transcription.
- If an STT model is named as the source, use `'gemini-4-argon-{stt model}'`, e.g. `'gemini-4-argon-gcp-chirp3'`.
- If a cheaper or faster model is wanted ("flash", "cheap"), use `'gemini-3.5-flash-lite-gemini-3.5-transcribe'`.

One run writes:

- corrected captions as SRT and WebVTT, with `[Name]` on every speaker change
- the audio description, as WebVTT descriptions tracks and a descriptive transcript
- the original video, and a copy with the captions burned in and the descriptions narrated (narrated with `tts_model`, default `gemini-3.8-flash-tts-Kore`)
- an Able Player page in `vtt-files/` for checking the WebVTT files. It opens straight from disk.

A 3-hour meeting takes about 20 minutes and costs about $3.

### From the video itself

"Summarize from the video", "get the speakers' names right".

`from_video='agentic'` also generates the summary, one-sentence summary, speakers and Q&A with Gemini video understanding. `agentic` lets the model move through the video and needs Gemini 3.5 Flash-Lite or later. `static` samples frames at 1 FPS and works with any Gemini model. `'agentic,static'` runs both.

### AI transcription

Set `transcript_source` to an STT model, e.g. `'gemini-3.5-transcribe'`, `'gemini-3.5-flash-lite'` or `'gcp-chirp3'`. `gemini-3.5-transcribe` returns word-level timestamps and adds `[Speaker N]` labels on speaker changes. `gcp-` models need `GOOGLE_CLOUD_PROJECT`, and `YTD_GCS_BUCKET_NAME` is recommended. With an AI transcript, the tool also summarizes the YouTube transcript; set `no_youtube_summary=True` to skip that.

### Output locations

- Local CSV (the default): `youtube-to-docs-artifacts/youtube-docs.csv`, or another `.csv` path inside `youtube-to-docs-artifacts/`
- Google Drive: `output_file='workspace'`
- SharePoint: `output_file='sharepoint'`
- Hugging Face dataset: `output_file='hf'`, plus `hugging_face_dataset` (a repo id like `username/my-dataset`, or a plain name to slugify under the signed-in user; a playlist uses its title by default). Needs `HF_TOKEN`.
- In memory, with no files on disk: `output_file='memory'`
- No CSV: `output_file='none'`

## Tool reference: `process_video`

| Argument                    | Description                                                                                                         | Examples                                                             |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------- |
| `url`                       | **Required.** YouTube URL, video ID, playlist ID, channel handle, or comma-separated IDs.                          | `https://youtu.be/...`, `PL...`, `@channel`                          |
| `model`                     | LLM(s) for the summary, Q&A, speakers and tags. Comma-separated for more than one.                                  | `gemini-3.5-flash-lite`                                              |
| `infographic_model`         | Image model(s) for infographics. Comma-separated for one infographic per model.                                     | `gemini-nano-banana-2.1`, `gemini-3-pro-image`                       |
| `alt_text_model`            | Model for the infographic alt text. Defaults to the summary model.                                                  | `gemini-3.5-flash-lite`                                              |
| `tts_model`                 | Text-to-speech model and voice.                                                                                     | `gemini-3.8-flash-lite-tts-Kore`, `gcp-chirp3-Kore`, `aws-polly-Ruth` |
| `all_suite`                 | Model suite for everything.                                                                                         | `gemini-flash`, `gemini-pro`, `gemini-flash-pro-image`, `gcp-pro`    |
| `combine_infographic_audio` | Combine the infographic and audio summary into an MP4. Needs `tts_model` and `infographic_model`.                   | `True`                                                               |
| `translate`                 | Translate all outputs: `{model}-{language}`, `aws-translate-{language}` or `gcp-translate-{language}`.              | `gemini-3.5-flash-lite-es`, `aws-translate-es`, `gcp-translate-es`   |
| `corrected_captions`        | Make the video accessible (see above). `'default'` for the best, or `{model}` / `{model}-{source}`.                 | `default`, `gemini-4-argon-youtube`                                  |
| `from_video`                | Also generate outputs from the video itself.                                                                        | `agentic`, `static`, `agentic,static`                                |
| `transcript_source`         | Transcript source. Defaults to `youtube`, or to the STT model that `corrected_captions` corrects.                   | `gemini-3.5-transcribe`, `gcp-chirp3`                                |
| `no_youtube_summary`        | Skip the extra summary of the YouTube transcript when an AI transcript is used.                                     | `True`                                                               |
| `post_process`              | JSON operations on the transcript. The results become CSV columns.                                                  | `'{"word count": "apple"}'`, `'{"word count": ["apple", "banana"]}'` |
| `output_file`               | Where the CSV goes (see above).                                                                                     | `workspace`, `sharepoint`, `hf`, `memory`, `none`                    |
| `hugging_face_dataset`      | Dataset for `output_file='hf'`.                                                                                     | `username/my-dataset`                                                |
| `verbose`                   | Verbose logging.                                                                                                    | `True`                                                               |

## Where the files go

Files are saved in folders next to the CSV, and each one is linked from a CSV column:

- `srt-files/`: SRT transcripts (YouTube and AI)
- `summary-files/`, `one-sentence-summary-files/`, `tag-files/`
- `qa-files/`: Q&A tables, with "Timestamp" and "Timestamp URL" links checked against the YouTube SRT timings
- `infographic-alt-text/`, `infographic-prompts/`, `summary-prompts/`
- `corrected-caption-files/`: corrected SRT captions
- `vtt-files/`: WebVTT captions and descriptions tracks, and the Able Player page
- `descriptive-transcript-files/`: the dialogue by speaker, with descriptions of the visual information
- `video-files/`: the original video, and the copy with captions burned in and descriptions narrated
- `video-context-files/`: on-screen names, speaking times, terms and events read from the video

## Examples

| Request                                                           | Call                                                                                                                           |
| :---------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| "Get me a transcript of this video."                              | `process_video(url='...')`                                                                                                     |
| "Make an infographic for this video using Gemini Pro."            | `process_video(url='...', model='gemini-4-argon', infographic_model='gemini-3-pro-image')`                                     |
| "Do a kitchen sink run on this video in Spanish."                 | `process_video(url='...', all_suite='gemini-pro', combine_infographic_audio=True, verbose=True, translate='gemini-3.5-flash-lite-es')` |
| "Summarize this playlist and save it to Drive."                   | `process_video(url='PL...', model='gemini-3.5-flash-lite', output_file='workspace')`                                           |
| "Make the best accessible captions for this video."               | `process_video(url='...', corrected_captions='default')`                                                                       |
| "Correct the YouTube captions for this video for accessibility."  | `process_video(url='...', corrected_captions='gemini-4-argon-youtube')`                                                        |
| "Make an audio-described version of this video."                  | `process_video(url='...', corrected_captions='default')`                                                                       |
| "Count how many times 'apple' appears in the transcript."         | `process_video(url='...', post_process='{"word count": "apple"}')`                                                             |
| "Summarize this hearing from the video and get the names right."  | `process_video(url='...', model='gemini-4-argon', from_video='agentic')`                                                       |

## Without the MCP server

If the `process_video` tool isn't available, run the CLI with `uv`. The URL or ID is a positional argument, so don't use `--url`:

```bash
uvx --from "youtube-to-docs[all]" youtube-to-docs B0x2I_doX9o                  # transcript
uvx --from "youtube-to-docs[all]" youtube-to-docs B0x2I_doX9o --all gemini-pro # kitchen sink
uvx --from "youtube-to-docs[all]" youtube-to-docs B0x2I_doX9o -cc              # accessible video
```

See [docs/usage.md](https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs/blob/main/docs/usage.md) for every option.
