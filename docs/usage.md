# Usage Guide

`youtube-to-docs` is a versatile tool designed to convert YouTube content into structured documentation, including transcripts, summaries, audio, and infographics. It is primarily designed as a Command Line Interface (CLI) tool but can also be used as a Python library.

## Command Line Interface (CLI)

The main command is `youtube-to-docs`.

### Basic Usage

Running the command without arguments processes a default video:

```bash
youtube-to-docs
```

### Arguments

| Argument | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `video_id` | The YouTube content to process. Can be a **Video ID**, **Playlist ID** (starts with `PL`), **Channel Handle** (starts with `@`), or a **comma-separated list** of Video IDs. | `atmGAHYpf_c` | `youtube-to-docs @mychannel` |
| `-o`, `--outfile` | Path to save the output CSV file. | `youtube-docs.csv` | `-o my-data.csv` |
| `-m`, `--model` | The LLM to use for summarization. Supports models from Google (Gemini), Vertex AI, AWS Bedrock, and Azure Foundry. | `None` | `-m gemini-3-flash-preview` |
| `--tts` | The TTS model and voice to use for generating audio summaries. Format: `{model}-{voice}`. | `None` | `--tts gemini-2.5-flash-preview-tts-Kore` |
| `--infographic`| The image model to use for generating a visual summary. | `None` | `--infographic gemini-2.5-flash-image` |

### Examples

**1. Process the default video and save to a custom CSV:**
```bash
youtube-to-docs -o youtube-docs.csv
```

**2. Summarize a Playlist using Google Gemini:**
```bash
youtube-to-docs PLGKTTEqwhiHHWO-jdxM1KtzTbWo6h0Ycl -m gemini-3-flash-preview
```

**3. Process a Channel with Summaries, TTS, and Infographics:**
```bash
youtube-to-docs @mga-othercommittees6625 -m vertex-claude-haiku-4-5@20251001 --tts gemini-2.5-flash-preview-tts-Kore --infographic gemini-2.5-flash-image
```

## Library Usage

While primarily a CLI, you can import core functions for custom workflows.

```python
from youtube_to_docs.transcript import fetch_transcript
from youtube_to_docs.llms import generate_summary

# Fetch a transcript
video_id = "atmGAHYpf_c"
transcript, is_generated = fetch_transcript(video_id)
print(f"Transcript (Generated: {is_generated}):\n{transcript[:100]}...")

# Generate a summary (requires API key setup)
# Note: This is a simplified example; you might need to handle context/config
# summary_text, in_tokens, out_tokens = generate_summary("gemini-1.5-flash", transcript, "Video Title", "http://url...")
```

## Why use `youtube-to-docs`?

You might find other tools that download YouTube transcripts, but `youtube-to-docs` distinguishes itself in several ways:

1.  **Multimodal Output**: It doesn't just stop at text.
    *   **Summaries**: Uses state-of-the-art LLMs to create concise summaries.
    *   **Audio (TTS)**: Converts summaries into audio files, perfect for listening on the go.
    *   **Visuals (Infographics)**: Generates AI-created infographics to visually represent the content.

2.  **Structured Data (CSV/Polars)**:
    *   Instead of loose files, metadata and paths are organized into a robust CSV file using `polars`.
    *   This makes it incredibly easy to import the data into Google Sheets, Excel, or a database for further analysis or publishing.

3.  **Batch Processing**:
    *   Seamlessly handles individual videos, entire playlists, or full channels with a single command.

4.  **Multi-Provider Support**:
    *   Agnostic to the LLM provider. Whether you use Google Gemini, Vertex AI, AWS Bedrock, or Azure Foundry, you can plug in your preferred model.

5.  **Cost Awareness**:
    *   When using paid API models, it tracks and estimates the cost of summarization, saving it directly to your data file.
