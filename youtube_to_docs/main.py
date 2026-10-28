import argparse
import codecs
import io
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

import isodate
import polars as pl
from rich import print as rprint
from rich_argparse import RichHelpFormatter

from youtube_to_docs.audio_description import process_audio_description
from youtube_to_docs.captions import process_corrected_captions
from youtube_to_docs.infographic import (
    build_infographic_prompt,
    generate_infographic,
    image_file_extension,
)
from youtube_to_docs.llms import (
    build_description,
    build_summary_prompt,
    extract_speakers,
    generate_chapters,
    generate_description_text,
    generate_one_sentence_summary,
    generate_qa,
    generate_summary,
    generate_tags,
    get_model_pricing,
    parse_chapters,
)
from youtube_to_docs.models import MODEL_SUITES
from youtube_to_docs.post_process import post_process_transcript
from youtube_to_docs.providers import (
    MultimodalProvider,
    STTProvider,
    get_provider,
)
from youtube_to_docs.storage import (
    GoogleDriveStorage,
    HuggingFaceStorage,
    LocalStorage,
    M365Storage,
    MemoryStorage,
    NullStorage,
)
from youtube_to_docs.transcript import (
    download_video,
    extract_audio,
    extract_playlist_id,
    fetch_transcript,
    format_as_srt,
    get_playlist_title,
    get_video_details,
    get_youtube_service,
    resolve_video_ids,
)
from youtube_to_docs.translate import (
    parse_corrected_captions_arg,
    parse_translate_arg,
    translate_text,
)
from youtube_to_docs.tts import process_tts
from youtube_to_docs.utils import (
    format_clickable_path,
    get_version,
    is_error_response,
    normalize_model_name,
    reorder_columns,
)
from youtube_to_docs.video import (
    process_accessible_video,
    process_videos,
    save_original_video,
    write_player_page,
)
from youtube_to_docs.video_understanding import parse_video_modes, process_from_video

# `-cc` without a value: correct the dedicated Gemini STT transcript (word
# timestamps and diarization) with Gemini's strongest model.
CORRECTED_CAPTIONS_DEFAULT = "gemini-4-argon-gemini-3.5-transcribe"


class VerboseRow(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        rprint(f"Updated column: {key}")

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        if args:
            if isinstance(args[0], dict):
                for k in args[0]:
                    rprint(f"Updated column: {k}")
        for k in kwargs:
            rprint(f"Updated column: {k}")


def _use_utf8_output() -> None:
    """
    Writes output as UTF-8. On Windows, piped or redirected output otherwise
    uses the ANSI code page (e.g. cp1252), which can't encode the arrows in the
    help text or many video titles, and raises UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        if (
            isinstance(stream, io.TextIOWrapper)
            and codecs.lookup(stream.encoding).name != "utf-8"
        ):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(args_list: list[str] | None = None) -> "MemoryStorage | None":
    _use_utf8_output()
    local_temp_dir = tempfile.mkdtemp(prefix="ytd_temp_")
    try:
        return _main_impl(args_list, local_temp_dir=local_temp_dir)
    finally:
        shutil.rmtree(local_temp_dir, ignore_errors=True)


def _main_impl(
    args_list: list[str] | None = None,
    local_temp_dir: str | None = None,
) -> "MemoryStorage | None":
    # Define styles for the help output
    RichHelpFormatter.styles["argparse.args"] = "cyan italic"
    RichHelpFormatter.styles["argparse.groups"] = "bold yellow"
    RichHelpFormatter.styles["argparse.metavar"] = "bold magenta"

    version = get_version()

    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"YouTube-to-docs version: {version}",
        help="Show the YouTube-to-docs version and exit.",
    )
    parser.add_argument(
        "video_id",
        nargs="?",
        default="atmGAHYpf_c",
        help=(
            "Can be one of: \n"
            "A YouTube URL e.g. `https://www.youtube.com/watch?v=atmGAHYpf_c`\n"
            "A Video ID e.g. `atmGAHYpf_c`\n"
            "Playlist ID (starts with `PL` e.g. `PL8ZxoInteClyHaiReuOHpv6Z4SPrXtYtW`)\n"
            "Channel Handle (starts with `@` e.g. `@mga-hgo1740`)\n"
            "Comma-separated list of Video IDs. (e.g. `KuPc06JgI_A,GalhDyf3F8g`)"
        ),
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="youtube-to-docs-artifacts/youtube-docs.csv",
        help=(
            "Can be one of: \n"
            "Local file path to save the output CSV file.\n"
            "`workspace` or `w` to store to Google Drive or a workspace folder ID.\n"
            "`sharepoint` or `s` to store to Microsoft SharePoint.\n"
            "`hf` to store to a Hugging Face dataset (see --hugging-face-dataset).\n"
            "`memory` or `m` to keep artifacts in memory (no files on disk).\n"
            "`none` or `n` to skip saving to a file (results will be in the log)."
        ),
    )
    parser.add_argument(
        "--hugging-face-dataset",
        default=None,
        help=(
            "The Hugging Face dataset repository to store artifacts in when "
            "`--outfile hf` is used. \n"
            "Can be a full repo id (e.g. `username/my-dataset`) or a plain name "
            "(e.g. `Code for America Summit 2026 Recap`) which is slugified and "
            "namespaced under the authenticated user. \n"
            "If omitted and the input is a playlist, the playlist's title is used "
            "as the dataset name. \n"
            "Requires the `HF_TOKEN` environment variable."
        ),
    )
    parser.add_argument(
        "-t",
        "--transcript",
        default=None,
        help=(
            "The transcript source to use. \n"
            "Can be `youtube` (default) to fetch existing YouTube transcripts, \n"
            "AI model name (e.g. `gemini-3.5-transcribe`, `gemini-4-argon`, "
            "`gemini-3.5-flash-lite`, `gcp-chirp3`, `aws-transcribe`) to "
            "perform STT on extracted audio. \n"
            "`gemini-3.5-transcribe` is Google's dedicated STT model: it returns "
            "word-level timestamps and speaker diarization, so the SRT carries real "
            "offsets and `[Speaker N]` labels on speaker changes. \n"
            "When `-cc` corrects an STT transcript (by default "
            "`gemini-3.5-transcribe`), that model is the default instead, so it "
            "is transcribed in the same run. \n"
            "For `gcp-` models (Cloud Speech-to-Text V2), `GOOGLE_CLOUD_PROJECT` "
            "is required and `YTD_GCS_BUCKET_NAME` is optional/recommended. \n"
            "For `aws-transcribe`, `YTD_S3_BUCKET_NAME` is required."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="?",
        const="gemini-3.5-flash-lite",
        default=None,
        help=(
            "The LLM to use for speaker extraction, Q&A generation, tag generation, "
            "and summarization.\n"
            "Can be one of: \n"
            "Gemini model (e.g., `gemini-3.5-flash-lite`)\n"
            "GCP Vertex model (prefixed with `vertex-`; e.g. "
            "`vertex-claude-haiku-5-5`)\n"
            "AWS Bedrock model (prefixed with `bedrock-`; e.g. "
            "`bedrock-claude-haiku-5-5`\n"
            "`bedrock-nova-2-lite-v1`)\n"
            "Azure Foundry model (prefix with `foundry-`; e.g. "
            "`foundry-gpt-6-luna`)\n"
            "Can also be a comma-separated list of models (e.g. "
            "`gemini-3.5-flash-lite,bedrock-claude-haiku-5-5`).\n"
            "Defaults to `None` (Transcript only)."
        ),
    )
    parser.add_argument(
        "--tts",
        default=None,
        help=(
            "Generate an audio file for each summary by reading it aloud with a "
            "text-to-speech model. When used with --translate, audio is also "
            "generated for the translated summaries in the target language. \n"
            "Format: `{model}-{voice}` e.g. `gemini-3.8-flash-lite-tts-Kore` \n"
            "GCP Cloud TTS: `gcp-chirp3` or `gcp-chirp3-{voice}` "
            "(e.g. `gcp-chirp3-Kore`)\n"
            "AWS Polly: `aws-polly` (defaults to Ruth) or `aws-polly-{voice}` "
            "(e.g. `aws-polly-Ruth`)"
        ),
    )
    parser.add_argument(
        "-i",
        "--infographic",
        nargs="?",
        const="gemini-3-pro-image",
        default=None,
        help=(
            "The image model to use for generating an infographic. "
            "If set as a flag without a value, defaults to "
            "`gemini-3-pro-image`. "
            "e.g. `gemini-nano-banana-2.1`, `gemini-3-pro-image` or "
            "`foundry-gpt-image-2.5-sunburst`. "
            "Can be a comma-separated list to generate one infographic per model "
            "(e.g. `gemini-nano-banana-2.1,gemini-3-pro-image`)."
        ),
    )
    parser.add_argument(
        "--alt-text-model",
        default=None,
        help=(
            "The LLM model to use for generating alt text for the infographic. "
            "Defaults to the model used for summarization."
        ),
    )
    parser.add_argument(
        "-nys",
        "--no-youtube-summary",
        action="store_true",
        help=(
            "If set, skips generating a secondary summary from the YouTube "
            "transcript when using an AI model for the primary transcript."
        ),
    )
    parser.add_argument(
        "-fv",
        "--from-video",
        default=None,
        help=(
            "Also generate the summary, one sentence summary, speakers and Q&A "
            "directly from the video using Gemini video understanding (no "
            "download needed). Value is the processing mode: `agentic` (the model "
            "navigates the video, loading only what it needs; Gemini 3.5 "
            "Flash-Lite and later, e.g. gemini-3.8-flash, gemini-4-argon) or "
            "`static` (frames at 1 FPS; any Gemini model). Use `agentic,static` "
            "to run both for comparison. Outputs are labelled "
            "`from video-agentic` / `from video-static`."
        ),
    )
    parser.add_argument(
        "-tr",
        "--translate",
        default=None,
        help=(
            "Translate LLM outputs to a target language after generating in English. "
            "Format: `{model}-{language}` e.g. `gemini-3.5-flash-lite-spanish` \n"
            "or `gemini-3.5-flash-lite-es` or `bedrock-nova-2-lite-v1-fr`\n"
            "or `aws-translate-spanish` or `gcp-translate-french`\n"
            "Both language names (e.g. 'spanish', 'french', 'korean') and "
            "ISO codes (e.g. 'es', 'fr', 'ko') are accepted.\n"
            "Use `aws-translate` to use AWS Translate, or `gcp-translate` to use "
            "Google Cloud Translation API, instead of an LLM. "
            "All summaries, Q&A, tags, and one-sentence summaries are generated in "
            "English first, then translated using the specified model and language."
        ),
    )
    parser.add_argument(
        "-cia",
        "--combine-infographic-audio",
        action="store_true",
        help="Combine the infographic and audio summary into a video file.",
    )
    parser.add_argument(
        "-a",
        "--all",
        help=(
            "Shortcut to use a specific model suite for everything. \n"
            "Supported values: \n"
            "`gemini-flash`: summarization (`gemini-3.5-flash-lite`), "
            "TTS (`gemini-3.8-flash-lite-tts-Kore`), "
            "and Infographic (`gemini-nano-banana-2.1`). \n"
            "`gemini-pro`: summarization (`gemini-4-argon`), "
            "TTS (`gemini-3.8-flash-tts-Kore`), "
            "and Infographic (`gemini-3-pro-image`). \n"
            "`gemini-flash-pro-image`: summarization (`gemini-3.5-flash-lite`), "
            "TTS (`gemini-3.8-flash-lite-tts-Kore`), "
            "and Infographic (`gemini-3-pro-image`). \n"
            "`gcp-pro`: summarization (`gemini-4-argon`), "
            "TTS (`gcp-chirp3`), "
            "and Infographic (`gemini-3-pro-image`). \n"
            "`anthropic-opus`: summarization (`claude-opus-5-5`), "
            "TTS (`aws-polly-Ruth`), "
            "and Infographic (`gemini-3-pro-image`). \n"
            "Also sets `--no-youtube-summary`."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output.",
    )
    parser.add_argument(
        "-cc",
        "--corrected-captions",
        nargs="?",
        const=CORRECTED_CAPTIONS_DEFAULT,
        default=None,
        help=(
            "Make the video accessible under WCAG 2.1 Level AA and Section 508 "
            "(https://www.section508.gov/create/captions-transcripts/): \n"
            "- Corrected captions (`srt-files/` SRT, `transcript-files/` text "
            "and `vtt-files/` WebVTT): timing checked word by word against a second "
            "transcript, every speaker named on each change (`[Name]`) from "
            "diarization and the names and speaking times seen in the video, "
            "misheard words, names, punctuation, fillers, sounds and missed "
            "speech corrected, laid out as two lines of 42 characters over 1-7 "
            "seconds. \n"
            "- Audio description of the visual information a blind viewer would "
            "miss (slides, chart figures, on-screen text, votes): WebVTT "
            "descriptions tracks in `vtt-files/` (standard, fitted into pauses, "
            "and extended) and a descriptive transcript in "
            "`descriptive-transcript-files/`. \n"
            "- `video-files/`: the original video, and a copy with the captions "
            "burned in and the descriptions narrated in the pauses (`--tts`, "
            "default `gemini-3.8-flash-tts-Kore`). \n"
            "- `vtt-files/... player.html`: an Able Player page to check the "
            "WebVTT tracks in a browser (serve the folder over HTTP). \n"
            "The model reads on-screen context from the video, describes it and "
            "corrects the text (Gemini models; other models correct the text "
            "only). \n\n"
            f"Without a value, uses `{CORRECTED_CAPTIONS_DEFAULT}` (and transcribes "
            "with `gemini-3.5-transcribe` unless `-t` is set). Without `-m`, the "
            "summary and other text outputs use `gemini-4-argon`. \n\n"
            "Format: `{model}` or `{model}-{source}` \n\n"
            "Source rules:\n"
            "  - `youtube` → correct the YouTube SRT "
            "(e.g. `-cc gemini-4-argon-youtube`).\n"
            "  - Any transcript model name → correct the SRT from that STT run, "
            "transcribing in the same run unless `-t` is set "
            "(e.g. `-cc gemini-4-argon-gcp-chirp3`).\n"
            "  - Omit source → the `-t` STT SRT, else the most recent AI SRT, "
            "else the YouTube SRT. \n\n"
            "Examples:\n"
            "  Everything, best quality:      -cc\n"
            "  Correct YouTube captions:      -cc gemini-4-argon-youtube\n"
            "  Cheaper model:                 "
            "-cc gemini-3.5-flash-lite-gemini-3.5-transcribe"
        ),
    )
    parser.add_argument(
        "-pp",
        "--post-process",
        default=None,
        help=(
            "Post-process the transcript with JSON operations. "
            "Example: '{\"word count\": \"apple\"}' counts occurrences of 'apple' "
            "in the transcript. Values can be a list for multiple words: "
            '\'{"word count": ["apple", "banana"]}\''
        ),
    )

    args = parser.parse_args(args_list)

    rprint(f"Running YouTube-to-docs version: {version}")

    # Without an explicit -m, -cc uses Gemini's strongest model and -i or -fv
    # the cheaper gemini-3.5-flash-lite.
    if args.model is None:
        if args.corrected_captions:
            args.model = "gemini-4-argon"
        elif args.infographic or args.from_video:
            args.model = "gemini-3.5-flash-lite"

    corrected_captions_model = None
    corrected_captions_source = None
    if args.corrected_captions:
        corrected_captions_model, corrected_captions_source = (
            parse_corrected_captions_arg(args.corrected_captions)
        )
        # Transcribe the SRT that -cc corrects in the same run, unless -t is set.
        if args.transcript is None and corrected_captions_source not in (
            None,
            "youtube",
        ):
            args.transcript = corrected_captions_source

    try:
        from_video_modes = parse_video_modes(args.from_video)
    except ValueError as e:
        parser.error(str(e))

    verbose = args.verbose

    if not verbose:
        # Suppress noisy logs from external libraries unless verbose is enabled.
        logging.getLogger("google_genai").setLevel(logging.ERROR)
        logging.getLogger("google.generativeai").setLevel(logging.ERROR)
        logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
        logging.getLogger("googleapiclient.discovery").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        # Set root logger to WARNING to catch others
        logging.getLogger().setLevel(logging.WARNING)

    def vprint(*args, **kwargs):
        if verbose:
            rprint(*args, **kwargs)

    if args.all in MODEL_SUITES:
        suite = MODEL_SUITES[args.all]
        if args.model is None:
            args.model = suite["model"]
        if args.tts is None:
            args.tts = suite["tts"]
        if args.infographic is None:
            args.infographic = suite["infographic"]
        if args.transcript is None:
            args.transcript = suite["transcript"]
        args.no_youtube_summary = True
    if args.transcript is None:
        args.transcript = "youtube"

    transcript_arg = args.transcript
    video_id_input = args.video_id
    outfile = args.outfile
    hugging_face_dataset = args.hugging_face_dataset
    model_names_arg = args.model
    tts_arg = args.tts
    infographic_arg = args.infographic
    alt_text_model_arg = args.alt_text_model
    no_youtube_summary = args.no_youtube_summary
    translate_arg = args.translate

    combine_info_audio = args.combine_infographic_audio
    post_process_arg = args.post_process
    model_names = model_names_arg.split(",") if model_names_arg else []
    infographic_models = (
        [m.strip() for m in infographic_arg.split(",") if m.strip()]
        if infographic_arg
        else []
    )
    translate_model = None
    translate_lang = None
    if translate_arg:
        translate_model, translate_lang = parse_translate_arg(translate_arg)

    languages = ["en"]
    if translate_lang:
        languages.append(translate_lang)

    youtube_service = get_youtube_service()

    video_ids = resolve_video_ids(video_id_input, youtube_service)

    # Setup Output Directories
    # Setup Storage
    if outfile.lower() in ("none", "n"):
        vprint("Using Null storage. No files will be saved.")
        storage = NullStorage()
        outfile_path = "none.csv"
        base_dir = "."
    elif outfile in ("memory", "m"):
        vprint("Using Memory storage. Artifacts kept in memory.")
        storage = MemoryStorage()
        outfile_path = "memory.csv"
        base_dir = "."
    elif outfile in ("sharepoint", "s"):
        vprint(f"Using SharePoint storage. Output: {outfile}")
        storage = M365Storage()
        outfile_path = "youtube-docs.csv"
        base_dir = "."
    elif outfile.lower() == "hf":
        # If no dataset name was given and the input is a playlist, fall back to
        # the playlist's title as the dataset name.
        hf_dataset = hugging_face_dataset
        if not hf_dataset:
            playlist_id = extract_playlist_id(video_id_input)
            if playlist_id:
                playlist_title = get_playlist_title(playlist_id, youtube_service)
                if playlist_title:
                    vprint(
                        f"Using playlist title as Hugging Face dataset: "
                        f"{playlist_title}"
                    )
                    hf_dataset = playlist_title
        vprint(f"Using Hugging Face storage. Dataset: {hf_dataset}")
        storage = HuggingFaceStorage(hf_dataset)
        outfile_path = "youtube-docs.csv"
        base_dir = "."
    elif outfile in ("workspace", "w") or (
        len(outfile) > 20
        and "." not in outfile
        and "/" not in outfile
        and "\\" not in outfile
    ):
        # Heuristic: "workspace" or looks like a Folder ID
        vprint(f"Using Google Workspace storage. Output: {outfile}")
        storage = GoogleDriveStorage(outfile)
        outfile_path = "youtube-docs.csv"  # Relative to root_folder_id
        base_dir = "."
    else:
        vprint(f"Using Local storage. Output: {outfile}")
        storage = LocalStorage()
        output_dir = os.path.dirname(outfile)
        storage.ensure_directory(output_dir)  # Ensure parent of CSV exists
        base_dir = output_dir if output_dir else "."
        outfile_path = outfile  # Keep original path for local

    transcripts_dir = os.path.join(base_dir, "transcript-files")
    summaries_dir = os.path.join(base_dir, "summary-files")
    infographics_dir = os.path.join(base_dir, "infographic-files")
    speakers_dir = os.path.join(base_dir, "speaker-extraction-files")
    qa_dir = os.path.join(base_dir, "qa-files")
    audio_dir = os.path.join(base_dir, "audio-files")
    video_dir = os.path.join(base_dir, "video-files")
    one_sentence_summaries_dir = os.path.join(base_dir, "one-sentence-summary-files")
    tags_dir = os.path.join(base_dir, "tag-files")
    chapters_dir = os.path.join(base_dir, "chapter-files")
    descriptions_dir = os.path.join(base_dir, "descriptions-files")
    alt_text_dir = os.path.join(base_dir, "infographic-alt-text")
    infographic_prompts_dir = os.path.join(base_dir, "infographic-prompts")
    summary_prompts_dir = os.path.join(base_dir, "summary-prompts")
    srt_dir = os.path.join(base_dir, "srt-files")
    vtt_dir = os.path.join(base_dir, "vtt-files")
    descriptive_transcripts_dir = os.path.join(base_dir, "descriptive-transcript-files")
    video_context_dir = os.path.join(base_dir, "video-context-files")

    # Local temp dir for processing (Audio/TTS require local files)
    local_temp_dir = local_temp_dir or tempfile.mkdtemp(prefix="ytd_temp_")
    local_audio_dir = os.path.join(local_temp_dir, "audio-files")
    os.makedirs(local_audio_dir, exist_ok=True)

    storage.ensure_directory(transcripts_dir)
    storage.ensure_directory(summaries_dir)
    storage.ensure_directory(infographics_dir)
    storage.ensure_directory(speakers_dir)
    storage.ensure_directory(qa_dir)
    storage.ensure_directory(audio_dir)
    storage.ensure_directory(video_dir)
    storage.ensure_directory(one_sentence_summaries_dir)
    storage.ensure_directory(tags_dir)
    storage.ensure_directory(chapters_dir)
    storage.ensure_directory(descriptions_dir)
    storage.ensure_directory(alt_text_dir)
    storage.ensure_directory(infographic_prompts_dir)
    storage.ensure_directory(summary_prompts_dir)
    storage.ensure_directory(srt_dir)
    if corrected_captions_model:
        storage.ensure_directory(vtt_dir)
        storage.ensure_directory(descriptive_transcripts_dir)
    if from_video_modes or corrected_captions_model:
        storage.ensure_directory(video_context_dir)

    # Load existing CSV if it exists
    existing_df = storage.load_dataframe(outfile_path)
    if existing_df is not None:
        vprint(f"Loaded existing data from {outfile} ({len(existing_df)} rows)")
    else:
        vprint(f"No existing data found at {outfile}. Starting fresh.")

    rprint(f"Processing {len(video_ids)} videos.")
    rprint(f"Processing Videos: {video_ids}")
    if outfile in ("s", "sharepoint"):
        display_outfile = "SharePoint"
    elif isinstance(storage, HuggingFaceStorage):
        display_outfile = f"Hugging Face dataset ({storage.repo_id})"
    elif outfile in ("w", "workspace"):
        display_outfile = "Google Drive (Workspace)"
    elif outfile in ("n", "none"):
        display_outfile = "None (log only)"
    elif outfile in ("m", "memory"):
        display_outfile = "Memory (in-memory only)"
    else:
        display_outfile = str(Path(outfile).resolve())
    rprint(f"Saving to: {display_outfile}")

    if model_names:
        vprint(f"Summarizing using models: {model_names}")
    if translate_model and translate_lang:
        vprint(f"Translation: {translate_model} -> {translate_lang}")

    rows = []

    for i, video_id in enumerate(video_ids, 1):
        url = f"https://www.youtube.com/watch?v={video_id}"
        rprint(f"Processing Video ID: {video_id}")
        # Check if video already exists in CSV
        existing_row = None
        if existing_df is not None and "URL" in existing_df.columns:
            matches = existing_df.filter(pl.col("URL") == url)
            if not matches.is_empty():
                existing_row = matches.to_dicts()[0]

        # Determine if we need to process this video at all
        needs_details = existing_row is None

        # Prepare row data, starting with existing or empty
        row = existing_row.copy() if existing_row else {}
        if verbose:
            row = VerboseRow(row)
        row["URL"] = url  # Ensure URL is there

        # --- Language Independent Logic ---

        # Get Details
        if needs_details:
            details = get_video_details(video_id, youtube_service)
            if not details:
                continue
            (
                video_title,
                description,
                publishedAt,
                channelTitle,
                tags,
                video_duration,
                _,
                video_duration_seconds,
            ) = details
            row.update(
                {
                    "Title": video_title,
                    "Description": description,
                    "Data Published": publishedAt,
                    "Channel": channelTitle,
                    "Tags": tags,
                    "Duration": video_duration,
                }
            )
        else:
            video_title = row.get("Title", "")
            video_duration = row.get("Duration", "")
            # Approximate duration from string if not fresh, or 0.0
            video_duration_seconds = 0.0
            if video_duration and isinstance(video_duration, str):
                try:
                    if "T" in video_duration:
                        # YouTube duration is in ISO 8601 format (e.g., PT1H2M3S)
                        duration_obj = isodate.parse_duration(video_duration)
                        video_duration_seconds = duration_obj.total_seconds()
                    elif ":" in video_duration:
                        # Handle HH:MM:SS format from existing CSV
                        parts = video_duration.split(":")
                        if len(parts) == 3:  # HH:MM:SS
                            h, m, s = map(int, parts)
                            video_duration_seconds = h * 3600 + m * 60 + s
                        elif len(parts) == 2:  # MM:SS
                            m, s = map(int, parts)
                            video_duration_seconds = m * 60 + s
                except Exception:
                    # Fallback to 0.0 if parsing fails
                    video_duration_seconds = 0.0

        display_title = video_title if video_title else video_id
        print(f"(Video {i} of {len(video_ids)}) Video Title: {display_title}")

        safe_title = re.sub(r'[\\/*?:"><|]', "_", video_title).replace("\n", " ")
        safe_title = safe_title.replace("\r", "")

        # Initial Save: Create the sheet with basic metadata if it's a new video
        if needs_details:
            try:
                # Create a temporary DF with just this new row
                temp_row_df = pl.DataFrame([row])

                # Combine with existing data
                # (excluding this video if it was somehow there, but we checked)
                # existing_df contains all OTHER videos.
                if existing_df is not None:
                    # We might need to align columns if row has new keys
                    # handled by diagonal concat
                    current_save_df = pl.concat(
                        [existing_df, temp_row_df], how="diagonal"
                    )
                else:
                    current_save_df = temp_row_df

                if "Data Published" in current_save_df.columns:
                    current_save_df = current_save_df.sort(
                        "Data Published", descending=True
                    )

                current_save_df = reorder_columns(current_save_df)
                storage.save_dataframe(current_save_df, outfile_path)
                vprint(f"Created/Updated {outfile} with initial details.")
            except Exception as e:
                print(f"Warning: Could not perform initial save: {e}")

        # Audio Extraction (needed if AI transcript is requested,
        # regardless of language)
        # Check storage first if missing in row
        if transcript_arg != "youtube":
            audio_link_or_path = row.get("Audio File")
            if not audio_link_or_path or not storage.exists(str(audio_link_or_path)):
                # Check if it exists in storage (by convention)
                expected_audio = os.path.join(audio_dir, f"{video_id}.m4a")
                if storage.exists(expected_audio):
                    vprint(f"Found existing audio file: {expected_audio}")
                    row["Audio File"] = (
                        storage.get_full_path(expected_audio)
                        if hasattr(storage, "get_full_path")
                        else expected_audio
                    )

        audio_file_path = row.get("Audio File", "")
        local_audio_path = ""

        # If we need audio for generation (STT not "youtube"), ensures we have logic
        if transcript_arg != "youtube":
            # If we already have a link/path
            if audio_file_path and storage.exists(audio_file_path):
                # It exists in storage.
                pass
            else:
                # Need to extract
                rprint(f"Extracting audio for {transcript_arg}...")
                local_audio_path = extract_audio(
                    video_id,
                    local_audio_dir,
                )
                if local_audio_path:
                    # Upload to storage
                    target_audio_path = os.path.join(audio_dir, f"{video_id}.m4a")
                    uploaded_path_or_link = storage.upload_file(
                        local_audio_path, target_audio_path, content_type="audio/mp4"
                    )
                    vprint(
                        "Audio saved to: "
                        f"{format_clickable_path(uploaded_path_or_link)}"
                    )
                    row["Audio File"] = uploaded_path_or_link
                    audio_file_path = uploaded_path_or_link

        # --- Language Dependent Logic ---
        for language in languages:
            rprint(f"--- Processing Language: {language} ---")

            col_suffix = f" ({language})" if language != "en" else ""
            if language != "en" and translate_model:
                lang_str = f" ({translate_model}-{language})"
            else:
                lang_str = f" ({language})" if language != "en" else ""

            col_youtube = f"Transcript File youtube generated{col_suffix}"
            col_human = f"Transcript File human generated{col_suffix}"
            col_srt = f"SRT File youtube{col_suffix}"

            # --- YouTube Transcript Fetching ---
            youtube_transcript = ""
            is_generated = False
            srt_content = ""

            # Check storage if not in row
            if not row.get(col_youtube) and not row.get(col_human):
                gen_path = os.path.join(
                    transcripts_dir,
                    f"youtube generated{lang_str} - {video_id} - {safe_title}.txt",
                )
                human_path = os.path.join(
                    transcripts_dir,
                    f"human generated{lang_str} - {video_id} - {safe_title}.txt",
                )
                if storage.exists(human_path):
                    row[col_human] = human_path
                elif storage.exists(gen_path):
                    row[col_youtube] = gen_path

            # Load from row
            if row.get(col_youtube):
                path = row[col_youtube]
                if path and storage.exists(str(path)):
                    youtube_transcript = storage.read_text(str(path))
                    is_generated = True
            elif row.get(col_human):
                path = row[col_human]
                if path and storage.exists(str(path)):
                    youtube_transcript = storage.read_text(str(path))
                    is_generated = False

            # Load SRT if available
            if youtube_transcript and row.get(col_srt):
                srt_path = row[col_srt]
                if srt_path and storage.exists(str(srt_path)):
                    srt_content = storage.read_text(str(srt_path))
            elif youtube_transcript:
                # Try to find it on disk
                expected_srt_path = os.path.join(
                    srt_dir,
                    f"{'youtube' if is_generated else 'human'} generated{lang_str} - "
                    f"{video_id} - {safe_title}.srt",
                )
                if storage.exists(expected_srt_path):
                    srt_content = storage.read_text(expected_srt_path)
                    row[col_srt] = expected_srt_path

            # If no existing transcript, fetch from YouTube
            if not youtube_transcript:
                result = fetch_transcript(video_id, language=language)
                if result:
                    youtube_transcript, is_generated, transcript_data = result
                    prefix = (
                        f"youtube generated{lang_str} - "
                        if is_generated
                        else f"human generated{lang_str} - "
                    )
                    filename = f"{prefix}{video_id} - {safe_title}.txt"
                    srt_filename = f"{prefix}{video_id} - {safe_title}.srt"
                    # Relative path for storage
                    target_path = os.path.join(transcripts_dir, filename)
                    srt_target_path = os.path.join(srt_dir, srt_filename)

                    try:
                        saved_path = storage.write_text(target_path, youtube_transcript)
                        rprint(
                            f"Saved YouTube transcript ({language}): "
                            f"{format_clickable_path(saved_path)}"
                        )
                        # Update row with YouTube transcript info
                        if is_generated:
                            row[col_youtube] = saved_path
                        else:
                            row[col_human] = saved_path

                        # Save SRT
                        srt_content = format_as_srt(transcript_data)
                        saved_srt_path = storage.write_text(
                            srt_target_path, srt_content
                        )
                        rprint(
                            "Saved YouTube SRT: "
                            f"{format_clickable_path(saved_srt_path)}"
                        )
                        row[col_srt] = saved_srt_path
                    except Exception as e:
                        print(f"Error writing YouTube transcript/SRT: {e}")

            # Update character counts
            if youtube_transcript:
                row[f"Transcript characters from youtube{col_suffix}"] = len(
                    youtube_transcript
                )
            elif language != "en":
                # For non-English: get English transcript, then translate it
                en_transcript = ""
                en_path = row.get("Transcript File human generated") or row.get(
                    "Transcript File youtube generated"
                )
                if en_path and storage.exists(str(en_path)):
                    en_transcript = storage.read_text(str(en_path))
                    vprint(f"Loaded English transcript for {language} translation.")
                else:
                    # Fetch English transcript fresh
                    en_result = fetch_transcript(video_id, language="en")
                    if en_result:
                        en_transcript, en_is_generated, _ = en_result
                        vprint(
                            f"Fetched English transcript for {language} translation."
                        )
                        # Save English transcript if missing
                        if not row.get(
                            "Transcript File human generated"
                        ) and not row.get("Transcript File youtube generated"):
                            prefix = (
                                "youtube generated - "
                                if en_is_generated
                                else "human generated - "
                            )
                            filename = f"{prefix}{video_id} - {safe_title}.txt"
                            target_path = os.path.join(transcripts_dir, filename)
                            try:
                                saved_path = storage.write_text(
                                    target_path, en_transcript
                                )
                                rprint(
                                    f"Saved English transcript: "
                                    f"{format_clickable_path(saved_path)}"
                                )
                                if en_is_generated:
                                    row["Transcript File youtube generated"] = (
                                        saved_path
                                    )
                                else:
                                    row["Transcript File human generated"] = saved_path
                            except Exception as e:
                                print(f"Error writing English transcript: {e}")

                if en_transcript and translate_model:
                    # Translate English transcript to target language
                    col_translated = f"Transcript File translated{col_suffix}"
                    if row.get(col_translated) and storage.exists(
                        str(row[col_translated])
                    ):
                        youtube_transcript = storage.read_text(str(row[col_translated]))
                        vprint(f"Loaded existing translated transcript ({language}).")
                    else:
                        rprint(
                            f"Translating transcript to {language} "
                            f"using {translate_model}..."
                        )
                        youtube_transcript, _, _ = translate_text(
                            translate_model, en_transcript, language
                        )
                        filename = (
                            f"youtube translated {translate_model}-{language}"
                            f" - {video_id} - {safe_title}.txt"
                        )
                        target_path = os.path.join(transcripts_dir, filename)
                        try:
                            saved_path = storage.write_text(
                                target_path, youtube_transcript
                            )
                            rprint(
                                f"Saved translated transcript ({language}): "
                                f"{format_clickable_path(saved_path)}"
                            )
                            row[col_translated] = saved_path
                        except Exception as e:
                            print(f"Error writing translated transcript: {e}")

                        # Translate SRT
                        en_srt = ""
                        en_srt_path = row.get("SRT File youtube")
                        if en_srt_path and storage.exists(str(en_srt_path)):
                            en_srt = storage.read_text(str(en_srt_path))
                        if en_srt and not row.get(col_srt):
                            rprint(
                                f"Translating SRT to {language} "
                                f"using {translate_model}..."
                            )
                            translated_srt, _, _ = translate_text(
                                translate_model, en_srt, language
                            )
                            srt_filename = (
                                f"youtube translated {translate_model}-{language}"
                                f" - {video_id} - {safe_title}.srt"
                            )
                            srt_target_path = os.path.join(srt_dir, srt_filename)
                            try:
                                saved_srt_path = storage.write_text(
                                    srt_target_path, translated_srt
                                )
                                rprint(
                                    f"Saved translated SRT ({language}): "
                                    f"{format_clickable_path(saved_srt_path)}"
                                )
                                row[col_srt] = saved_srt_path
                                srt_content = translated_srt
                            except Exception as e:
                                print(f"Error writing translated SRT: {e}")
                elif en_transcript:
                    # No translation model — use English as-is
                    youtube_transcript = en_transcript
                    vprint(
                        f"Using English transcript as fallback for {language} "
                        "processing (no translation model set)."
                    )

                if youtube_transcript:
                    row[f"Transcript characters from youtube{col_suffix}"] = len(
                        youtube_transcript
                    )

            # --- AI Transcript Generation (if requested) ---
            ai_transcript = ""
            ai_srt_content = ""
            srt_transcript = ""

            if transcript_arg != "youtube":
                ai_col = f"Transcript File {transcript_arg} generated{col_suffix}"
                ai_srt_col = f"SRT File {transcript_arg}{col_suffix}"
                stt_cost_col = (
                    f"{normalize_model_name(transcript_arg)} STT cost{col_suffix} ($)"
                )

                # Check storage if not in row
                if not row.get(ai_col):
                    expected_ai_path = os.path.join(
                        transcripts_dir,
                        f"{transcript_arg} generated{lang_str} - "
                        f"{video_id} - {safe_title}.txt",
                    )
                    if storage.exists(expected_ai_path):
                        row[ai_col] = expected_ai_path

                # Check row for AI transcript
                if row.get(ai_col):
                    path = row[ai_col]
                    if path and storage.exists(str(path)):
                        ai_transcript = storage.read_text(str(path))

                # Load AI SRT if available
                if ai_transcript and row.get(ai_srt_col):
                    srt_path = row[ai_srt_col]
                    if srt_path and storage.exists(str(srt_path)):
                        ai_srt_content = storage.read_text(str(srt_path))
                elif ai_transcript:
                    # Try to find it on disk
                    expected_ai_srt_path = os.path.join(
                        srt_dir,
                        f"{transcript_arg} generated{lang_str} - "
                        f"{video_id} - {safe_title}.srt",
                    )
                    if storage.exists(expected_ai_srt_path):
                        ai_srt_content = storage.read_text(expected_ai_srt_path)
                        row[ai_srt_col] = expected_ai_srt_path

                # If no existing AI transcript, generate or translate it
                if not ai_transcript:
                    if language != "en" and translate_model:
                        # Translate English AI transcript instead of re-running STT
                        en_ai_col = f"Transcript File {transcript_arg} generated"
                        en_ai_path = row.get(en_ai_col)
                        en_ai_text = ""
                        if (
                            en_ai_path
                            and isinstance(en_ai_path, str)
                            and storage.exists(en_ai_path)
                        ):
                            en_ai_text = storage.read_text(en_ai_path)

                        if en_ai_text:
                            rprint(
                                f"Translating AI transcript to {language} "
                                f"using {translate_model}..."
                            )
                            ai_transcript, _, _ = translate_text(
                                translate_model, en_ai_text, language
                            )
                            filename = (
                                f"{transcript_arg} translated "
                                f"{translate_model}-{language}"
                                f" - {video_id} - {safe_title}.txt"
                            )
                            target_path = os.path.join(transcripts_dir, filename)
                            try:
                                saved_path = storage.write_text(
                                    target_path, ai_transcript
                                )
                                rprint(
                                    f"Saved translated AI transcript ({language}): "
                                    f"{format_clickable_path(saved_path)}"
                                )
                                row[ai_col] = saved_path
                            except Exception as e:
                                print(f"Error writing translated AI transcript: {e}")

                            # Translate AI SRT
                            en_ai_srt_col = f"SRT File {transcript_arg}"
                            en_ai_srt_path = row.get(en_ai_srt_col)
                            en_ai_srt = ""
                            if en_ai_srt_path and storage.exists(str(en_ai_srt_path)):
                                en_ai_srt = storage.read_text(str(en_ai_srt_path))
                            if en_ai_srt and not row.get(ai_srt_col):
                                rprint(
                                    f"Translating AI SRT to {language} "
                                    f"using {translate_model}..."
                                )
                                translated_ai_srt, _, _ = translate_text(
                                    translate_model, en_ai_srt, language
                                )
                                srt_filename = (
                                    f"{transcript_arg} translated "
                                    f"{translate_model}-{language}"
                                    f" - {video_id} - {safe_title}.srt"
                                )
                                srt_target_path = os.path.join(srt_dir, srt_filename)
                                try:
                                    saved_srt_path = storage.write_text(
                                        srt_target_path, translated_ai_srt
                                    )
                                    rprint(
                                        f"Saved translated AI SRT ({language}): "
                                        f"{format_clickable_path(saved_srt_path)}"
                                    )
                                    row[ai_srt_col] = saved_srt_path
                                    ai_srt_content = translated_ai_srt
                                except Exception as e:
                                    print(f"Error writing translated AI SRT: {e}")
                        else:
                            vprint(
                                f"English AI transcript not found for translation "
                                f"({transcript_arg})."
                            )
                    else:
                        # Run STT for English (or when no translate model)
                        audio_input_path = (
                            local_audio_path
                            if local_audio_path and os.path.exists(local_audio_path)
                            else None
                        )

                        if not audio_input_path and audio_file_path:
                            # Try to get it locally via storage abstraction
                            vprint(f"Retrieving audio file locally: {audio_file_path}")
                            audio_input_path = storage.get_local_file(
                                audio_file_path, download_dir=local_audio_dir
                            )

                        if not audio_input_path:
                            print(
                                "Error: Audio file not found for STT: "
                                f"{audio_file_path}"
                            )
                        else:
                            vprint(
                                f"Generating transcript using model: {transcript_arg} "
                                f"({language})..."
                            )

                            # Use the unified provider
                            provider = get_provider(transcript_arg)
                            if isinstance(provider, STTProvider):
                                ai_transcript, ai_srt_content, stt_in, stt_out = (
                                    provider.transcribe(
                                        audio_input_path,
                                        url,
                                        language=language,
                                        duration_seconds=video_duration_seconds,
                                        srt=True,
                                    )
                                )
                                # If the provider didn't return text in one go, try to
                                # extract it from SRT
                                if not ai_transcript and ai_srt_content:
                                    vprint("Extracting plain text from SRT locally...")
                                    # Strip SRT tags, timestamps, and sequence numbers
                                    # Pattern for timestamps:
                                    # 00:00:00,000 --> 00:00:00,000
                                    # Pattern for sequence numbers:
                                    # \d+ followed by newline
                                    # Pattern for HTML-like tags: <[^>]+>
                                    text_blocks = []
                                    lines = ai_srt_content.strip().splitlines()
                                    for line in lines:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        if re.match(r"^\d+$", line):
                                            continue
                                        if "-->" in line:
                                            continue
                                        # Remove tags
                                        line = re.sub(r"<[^>]+>", "", line)
                                        if line:
                                            text_blocks.append(line)
                                    ai_transcript = " ".join(text_blocks)

                                # If still empty, fallback to provider call without SRT
                                if not ai_transcript:
                                    vprint(
                                        "Fallback: Requesting text-only transcript "
                                        "from provider..."
                                    )
                                    ai_transcript, _, _, _ = provider.transcribe(
                                        audio_input_path,
                                        url,
                                        language=language,
                                        duration_seconds=video_duration_seconds,
                                        srt=False,
                                    )
                            else:
                                print(f"Error: {transcript_arg} does not support STT.")
                                continue

                            # Save AI transcript
                            prefix = f"{transcript_arg} generated{lang_str} - "
                            filename = f"{prefix}{video_id} - {safe_title}.txt"
                            srt_filename = f"{prefix}{video_id} - {safe_title}.srt"
                            target_path = os.path.join(transcripts_dir, filename)
                            srt_target_path = os.path.join(srt_dir, srt_filename)

                            try:
                                saved_path = storage.write_text(
                                    target_path, ai_transcript
                                )
                                rprint(
                                    "Saved AI transcript: "
                                    f"{format_clickable_path(saved_path)}"
                                )
                                row[ai_col] = saved_path

                                saved_srt_path = storage.write_text(
                                    srt_target_path, ai_srt_content
                                )
                                rprint(
                                    f"Saved AI SRT: "
                                    f"{format_clickable_path(saved_srt_path)}"
                                )
                                row[ai_srt_col] = saved_srt_path
                            except Exception as e:
                                print(f"Error writing AI transcript/SRT: {e}")

                        # Calculate STT Cost
                        if verbose:
                            input_price, output_price = get_model_pricing(
                                transcript_arg
                            )
                            if input_price is not None and output_price is not None:
                                stt_cost = (stt_in / 1_000_000) * input_price + (
                                    stt_out / 1_000_000
                                ) * output_price
                                row[stt_cost_col] = round(stt_cost, 2)
                                vprint(f"STT cost: ${row[stt_cost_col]:.2f}")

            # If AI transcript exists (either found or generated),
            # use it for summaries
            if ai_transcript:
                transcript = ai_transcript
                # Use SRT for Q&A if available
                srt_transcript = ai_srt_content if ai_srt_content else transcript
            else:
                transcript = youtube_transcript
                srt_transcript = srt_content if srt_content else transcript

            if transcript_arg != "youtube":
                row[f"Transcript characters from {transcript_arg}{col_suffix}"] = len(
                    ai_transcript
                )

            def run_from_video(model_name: str) -> None:
                """Generates outputs directly from the video (`--from-video`)."""
                if not from_video_modes:
                    return
                process_from_video(
                    row=row,
                    storage=storage,
                    model_name=model_name,
                    modes=from_video_modes,
                    video_id=video_id,
                    video_title=video_title,
                    safe_title=safe_title,
                    url=url,
                    language=language,
                    col_suffix=col_suffix,
                    lang_str=lang_str,
                    dirs={
                        "summaries": summaries_dir,
                        "summary_prompts": summary_prompts_dir,
                        "one_sentence": one_sentence_summaries_dir,
                        "speakers": speakers_dir,
                        "qa": qa_dir,
                        "video_context": video_context_dir,
                    },
                    verbose=verbose,
                    vprint=vprint,
                )

            def write_description(model_name: str) -> None:
                """Saves the chapters (from the SRT) to chapter-files/, then the
                plain-text YouTube description to descriptions-files/: a short
                description, the chapters, then the tags as hashtags."""
                summary_col_name = (
                    f"Summary Text {model_name} from {transcript_arg}{col_suffix}"
                )
                tags_col_name = f"Tags {transcript_arg} {model_name} model{col_suffix}"
                chapters_col = (
                    f"Chapters {model_name} from {transcript_arg}{col_suffix}"
                )
                chapters_file_col = (
                    f"Chapters File {model_name} from {transcript_arg}{col_suffix}"
                )
                description_col = (
                    f"Description Text {model_name} from {transcript_arg}{col_suffix}"
                )
                description_file_col = (
                    f"Description File {model_name} from {transcript_arg}{col_suffix}"
                )
                summary = row.get(summary_col_name)
                if not summary or is_error_response(summary):
                    return

                def add_cost(kind: str, tokens_in: int, tokens_out: int) -> None:
                    if not verbose:
                        return
                    input_price, output_price = get_model_pricing(model_name)
                    if input_price is None or output_price is None:
                        return
                    cost = (tokens_in / 1_000_000) * input_price + (
                        tokens_out / 1_000_000
                    ) * output_price
                    row[
                        f"{normalize_model_name(model_name)} "
                        f"{kind} cost from {transcript_arg}{col_suffix} ($)"
                    ] = round(cost, 2)
                    vprint(f"{kind.capitalize()} cost: ${cost:.2f}")

                # Chapters: reuse the saved ones, else generate and save them.
                chapters_text = row.get(chapters_col)
                if not isinstance(chapters_text, str) or not chapters_text.strip():
                    chapters_text = ""
                    if row.get(chapters_file_col):
                        try:
                            chapters_text = parse_chapters(
                                storage.read_text(str(row[chapters_file_col]))
                            )
                        except Exception as e:
                            print(f"Warning: Failed to read chapters file: {e}")
                if not chapters_text and chapters_dir:
                    rprint(
                        f"Generating chapters using model: {model_name} ({language})"
                    )
                    chapters_text, ch_input, ch_output = generate_chapters(
                        model_name, srt_transcript, language=language
                    )
                    add_cost("chapters", ch_input, ch_output)
                    if is_error_response(chapters_text) or not chapters_text:
                        rprint(
                            "[yellow]Warning: Chapter generation failed "
                            f"({chapters_text or 'fewer than 3 chapters'}); "
                            "the description has no chapters.[/yellow]"
                        )
                        chapters_text = ""
                    else:
                        chapters_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"chapters (from {transcript_arg}){lang_str}.txt"
                        )
                        target_path = os.path.join(chapters_dir, chapters_filename)
                        try:
                            chapters_full_path = storage.write_text(
                                target_path, chapters_text
                            )
                            rprint(
                                "Saved chapters: "
                                f"{format_clickable_path(chapters_full_path)}"
                            )
                            row[chapters_file_col] = chapters_full_path
                        except Exception as e:
                            print(f"Error writing chapters: {e}")
                if chapters_text:
                    row[chapters_col] = chapters_text

                if not descriptions_dir or row.get(description_file_col):
                    return
                rprint(f"Generating description using model: {model_name} ({language})")
                intro, d_input, d_output = generate_description_text(
                    model_name, summary, language=language
                )
                add_cost("description", d_input, d_output)
                if is_error_response(intro):
                    rprint(
                        f"[yellow]Warning: Description generation failed "
                        f"({intro}); skipping file save.[/yellow]"
                    )
                    return
                tags_val = row.get(tags_col_name)
                description_text = build_description(
                    intro,
                    chapters_text,
                    tags_val
                    if isinstance(tags_val, str) and not is_error_response(tags_val)
                    else "",
                )
                row[description_col] = description_text
                description_filename = (
                    f"{model_name} - {video_id} - {safe_title} - "
                    f"description (from {transcript_arg}){lang_str}.txt"
                )
                target_path = os.path.join(descriptions_dir, description_filename)
                try:
                    description_full_path = storage.write_text(
                        target_path, description_text
                    )
                    rprint(
                        "Saved description: "
                        f"{format_clickable_path(description_full_path)}"
                    )
                    row[description_file_col] = description_full_path
                except Exception as e:
                    print(f"Error writing description: {e}")

            if not transcript:
                vprint(
                    f"No transcript (YouTube or AI) available for {video_id} "
                    f"({language}). Skipping transcript-based processing for this "
                    "language."
                )
                # The video itself can still be summarized without a transcript.
                for model_name in model_names:
                    run_from_video(model_name)
                continue

            # --- Post-processing ---
            if post_process_arg:
                pp_results = post_process_transcript(transcript, post_process_arg)
                row.update(pp_results)
                if pp_results:
                    vprint(f"Post-process results: {pp_results}")

            # Summarize for each requested model
            for model_name in model_names:
                # Outputs generated directly from the video. Run first: the
                # transcript-based steps below `continue` early when their
                # outputs already exist, which would otherwise skip this.
                run_from_video(model_name)

                summary_col_name = (
                    f"Summary Text {model_name} from {transcript_arg}{col_suffix}"
                )
                summary_file_col_name = (
                    f"Summary File {model_name} from {transcript_arg}{col_suffix}"
                )
                speakers_col_name = f"Speakers {model_name} from {transcript_arg}"
                speakers_file_col_name = (
                    f"Speakers File {model_name} from {transcript_arg}"
                )
                summary_cost_col_name = (
                    f"{normalize_model_name(model_name)} "
                    f"summary cost from {transcript_arg}{col_suffix} ($)"
                )
                speaker_cost_col_name = (
                    f"{normalize_model_name(model_name)} "
                    f"Speaker extraction cost from {transcript_arg} ($)"
                )
                summary_prompt_file_col_name = (
                    f"Summary Prompt Path {model_name} from "
                    f"{transcript_arg}{col_suffix}"
                )

                # Save the exact prompt sent to the LLM. Done before generation
                # so the prompt is recorded even if summarization later fails.
                if not row.get(summary_prompt_file_col_name):
                    summary_prompt_text = build_summary_prompt(
                        transcript, video_title, url, language=language
                    )
                    summary_prompt_filename = (
                        f"{model_name} - {video_id} - {safe_title} - "
                        f"summary-prompt (from {transcript_arg}){lang_str}.txt"
                    )
                    target_path = os.path.join(
                        summary_prompts_dir, summary_prompt_filename
                    )
                    try:
                        saved_prompt_path = storage.write_text(
                            target_path, summary_prompt_text
                        )
                        rprint(
                            "Saved summary prompt: "
                            f"{format_clickable_path(saved_prompt_path)}"
                        )
                        row[summary_prompt_file_col_name] = saved_prompt_path
                    except Exception as e:
                        print(f"Error writing summary prompt: {e}")

                # Speaker Extraction
                speakers_text = ""
                speakers_input = 0
                speakers_output = 0
                # Check disk for speakers file
                if not row.get(speakers_file_col_name):
                    speakers_filename = (
                        f"{model_name} - {video_id} - {safe_title} - "
                        f"speakers (from {transcript_arg}).txt"
                    )
                    expected_path = os.path.join(speakers_dir, speakers_filename)
                    if storage.exists(expected_path):
                        row[speakers_file_col_name] = expected_path

                # Load speakers from file/row
                if row.get(speakers_file_col_name):
                    path = row[speakers_file_col_name]
                    if path and storage.exists(str(path)):
                        speakers_text = storage.read_text(str(path))
                        row[speakers_col_name] = speakers_text
                elif row.get(speakers_col_name) and not is_error_response(
                    row.get(speakers_col_name)
                ):
                    speakers_text = row[speakers_col_name]

                if not speakers_text or is_error_response(speakers_text):
                    # For speaker extraction, try to use English transcript if available
                    speaker_source_transcript = transcript
                    if language != "en":
                        en_path = row.get("Transcript File human generated") or row.get(
                            "Transcript File youtube generated"
                        )
                        if en_path and storage.exists(str(en_path)):
                            speaker_source_transcript = storage.read_text(str(en_path))
                            vprint(
                                "Using English transcript for speaker extraction "
                                f"({model_name})."
                            )
                        rprint(f"Extracting speakers using model: {model_name}")

                    speakers_text, speakers_input, speakers_output = extract_speakers(
                        model_name, speaker_source_transcript
                    )
                    row[speakers_col_name] = speakers_text
                    if (
                        speakers_text.strip() == "nan"
                        or speakers_text.strip() == 'float("nan")'
                    ):
                        row[speakers_col_name] = float("nan")

                    # Save Speakers File
                    if is_error_response(speakers_text):
                        rprint(
                            f"[yellow]Warning: Speaker extraction failed "
                            f"({speakers_text}); skipping file save.[/yellow]"
                        )
                        row[speakers_file_col_name] = ""
                    elif speakers_text and not isinstance(
                        row[speakers_col_name], float
                    ):
                        speakers_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"speakers (from {transcript_arg}).txt"
                        )
                        target_path = os.path.join(speakers_dir, speakers_filename)
                        try:
                            saved_path = storage.write_text(target_path, speakers_text)
                            rprint(
                                f"Saved speakers: {format_clickable_path(saved_path)}"
                            )
                            row[speakers_file_col_name] = saved_path
                        except Exception as e:
                            print(f"Error writing speakers file: {e}")

                    # Calculate Speaker Cost immediately
                    if verbose:
                        input_price, output_price = get_model_pricing(model_name)
                        if input_price is not None and output_price is not None:
                            speaker_cost = (
                                speakers_input / 1_000_000
                            ) * input_price + (
                                speakers_output / 1_000_000
                            ) * output_price
                            row[speaker_cost_col_name] = round(speaker_cost, 2)
                            vprint(f"Speaker extraction cost: ${speaker_cost:.2f}")

                # QA Generation
                qa_col_name = f"QA Text {model_name} from {transcript_arg}{col_suffix}"
                qa_file_col_name = (
                    f"QA File {model_name} from {transcript_arg}{col_suffix}"
                )
                qa_cost_col_name = (
                    f"{normalize_model_name(model_name)} QA cost from "
                    f"{transcript_arg}{col_suffix} ($)"
                )

                # Determine the best transcript for QA (prefer SRT for timestamps)
                qa_transcript_to_use = srt_transcript if srt_transcript else transcript

                # Check disk for QA file
                if not row.get(qa_file_col_name):
                    qa_filename = (
                        f"{model_name} - {video_id} - {safe_title} - "
                        f"qa (from {transcript_arg}){lang_str}.md"
                    )
                    expected_path = os.path.join(qa_dir, qa_filename)
                    if storage.exists(expected_path):
                        row[qa_file_col_name] = expected_path

                # Load QA from file/row
                if row.get(qa_file_col_name):
                    path = row[qa_file_col_name]
                    if path and storage.exists(str(path)):
                        row[qa_col_name] = storage.read_text(str(path))

                if not row.get(qa_col_name) or is_error_response(row.get(qa_col_name)):
                    rprint(f"Generating Q&A using model: {model_name} ({language})")

                    qa_text, qa_input, qa_output = generate_qa(
                        model_name,
                        qa_transcript_to_use,
                        speakers_text,
                        url,
                        language=language,
                        timing_reference=srt_content
                        if transcript_arg != "youtube"
                        else None,
                    )
                    row[qa_col_name] = qa_text

                    if qa_text.strip() == "nan" or qa_text.strip() == 'float("nan")':
                        row[qa_col_name] = float("nan")

                    # Save QA File
                    if is_error_response(qa_text):
                        rprint(
                            f"[yellow]Warning: Q&A generation failed "
                            f"({qa_text}); skipping file save.[/yellow]"
                        )
                        row[qa_file_col_name] = ""
                    elif qa_text and not isinstance(row[qa_col_name], float):
                        qa_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"qa (from {transcript_arg}){lang_str}.md"
                        )
                        target_path = os.path.join(qa_dir, qa_filename)
                        try:
                            saved_path = storage.write_text(target_path, qa_text)
                            rprint(f"Saved Q&A: {format_clickable_path(saved_path)}")
                            row[qa_file_col_name] = saved_path
                        except Exception as e:
                            print(f"Error writing Q&A file: {e}")

                    # Calculate QA Cost
                    if verbose:
                        input_price, output_price = get_model_pricing(model_name)
                        if input_price is not None and output_price is not None:
                            # QA input tokens include the speaker text provided in
                            # the prompt
                            qa_cost = (qa_input / 1_000_000) * input_price + (
                                qa_output / 1_000_000
                            ) * output_price
                            qa_cost = round(qa_cost, 2)
                            row[qa_cost_col_name] = qa_cost
                            vprint(f"Q&A cost: ${qa_cost:.2f}")

                # Check if we already have it in the row (from existing_row
                # or just loaded)
                if (
                    summary_col_name in row
                    and row[summary_col_name]
                    and not is_error_response(row[summary_col_name])
                ):
                    # Check if cost is missing and backfill if possible
                    if (
                        summary_cost_col_name not in row
                        or row[summary_cost_col_name] is None
                        or (
                            isinstance(row[summary_cost_col_name], float)
                            and row[summary_cost_col_name] != row[summary_cost_col_name]
                        )
                    ):  # Check for NaN
                        if verbose:
                            vprint(f"Backfilling cost for model: {model_name}")
                            input_price, output_price = get_model_pricing(model_name)
                            if input_price is not None and output_price is not None:
                                # Estimate tokens: ~4 chars per token
                                est_input_tokens = len(transcript) / 4
                                est_output_tokens = len(row[summary_col_name]) / 4
                                summary_cost = (
                                    est_input_tokens / 1_000_000
                                ) * input_price + (
                                    est_output_tokens / 1_000_000
                                ) * output_price

                                # Add speaker cost if we just generated them or
                                # can backfill it
                                if speakers_input > 0 or speakers_output > 0:
                                    s_cost = (
                                        speakers_input / 1_000_000
                                    ) * input_price + (
                                        speakers_output / 1_000_000
                                    ) * output_price
                                    summary_cost += s_cost
                                elif speaker_cost_col_name in row and not (
                                    isinstance(row[speaker_cost_col_name], float)
                                    and row[speaker_cost_col_name]
                                    != row[speaker_cost_col_name]
                                ):
                                    summary_cost += row[speaker_cost_col_name]

                                summary_cost = round(summary_cost, 2)
                                row[summary_cost_col_name] = summary_cost
                                vprint(f"Estimated summary cost: ${summary_cost:.2f}")
                    elif speakers_input > 0 or speakers_output > 0:
                        # Cost exists, but we generated speakers. Add that cost.
                        if verbose:
                            input_price, output_price = get_model_pricing(model_name)
                            if input_price is not None and output_price is not None:
                                s_cost = (speakers_input / 1_000_000) * input_price + (
                                    speakers_output / 1_000_000
                                ) * output_price
                                current_cost = row[summary_cost_col_name]
                                row[summary_cost_col_name] = round(
                                    current_cost + s_cost, 2
                                )
                                vprint(
                                    "Updated cost with speakers: "
                                    f"${row[summary_cost_col_name]:.2f}"
                                )
                    write_description(model_name)
                    continue

                # Check disk for summary file
                if not row.get(summary_file_col_name):
                    summary_filename = (
                        f"{model_name} - {video_id} - {safe_title} - "
                        f"summary (from {transcript_arg}){lang_str}.md"
                    )
                    expected_path = os.path.join(summaries_dir, summary_filename)
                    if storage.exists(expected_path):
                        row[summary_file_col_name] = expected_path

                # Load Summary from file/row
                if row.get(summary_file_col_name):
                    path = row[summary_file_col_name]
                    if path:
                        # We try to read.
                        try:
                            row[summary_col_name] = storage.read_text(str(path))
                        except Exception as e:
                            print(f"Warning: Failed to read summary file {path}: {e}")

                if not row.get(summary_col_name) or is_error_response(
                    row.get(summary_col_name)
                ):
                    rprint(f"Summarizing using model: {model_name} ({language})")

                    summary_text, input_tokens, output_tokens = generate_summary(
                        model_name, transcript, video_title, url, language=language
                    )

                    if verbose:
                        input_price, output_price = get_model_pricing(model_name)
                        if input_price is not None and output_price is not None:
                            # Add speaker tokens
                            total_input = input_tokens + speakers_input
                            total_output = output_tokens + speakers_output

                            summary_cost = (total_input / 1_000_000) * input_price + (
                                total_output / 1_000_000
                            ) * output_price
                            summary_cost = round(summary_cost, 2)
                            vprint(f"Summary cost: ${summary_cost:.2f}")
                            row[summary_cost_col_name] = summary_cost

                    summary_full_path = ""
                    if is_error_response(summary_text):
                        rprint(
                            f"[yellow]Warning: Summary generation failed "
                            f"({summary_text}); skipping file save.[/yellow]"
                        )
                    elif summaries_dir and summary_text:
                        summary_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"summary (from {transcript_arg}){lang_str}.md"
                        )
                        target_path = os.path.join(summaries_dir, summary_filename)
                        try:
                            summary_full_path = storage.write_text(
                                target_path, summary_text
                            )
                            rprint(
                                "Saved summary: "
                                f"{format_clickable_path(summary_full_path)}"
                            )
                        except Exception as e:
                            print(f"Error writing summary: {e}")

                    row[summary_file_col_name] = summary_full_path
                    row[summary_col_name] = summary_text

                # One Sentence Summary Generation
                one_sentence_col_name = (
                    f"One Sentence Summary {model_name} from "
                    f"{transcript_arg}{col_suffix}"
                )
                one_sentence_cost_col_name = (
                    f"{normalize_model_name(model_name)} one sentence summary cost "
                    f"from {transcript_arg}{col_suffix} ($)"
                )

                if (
                    row.get(summary_col_name)
                    and not is_error_response(row.get(summary_col_name))
                    and (
                        not row.get(one_sentence_col_name)
                        or is_error_response(row.get(one_sentence_col_name))
                    )
                ):
                    vprint(
                        f"Generating one sentence summary using model: {model_name} "
                        f"({language})"
                    )
                    (
                        one_sentence_text,
                        os_input,
                        os_output,
                    ) = generate_one_sentence_summary(
                        model_name, row[summary_col_name], language=language
                    )
                    row[one_sentence_col_name] = one_sentence_text

                    # Save One Sentence Summary File
                    if is_error_response(one_sentence_text):
                        rprint(
                            f"[yellow]Warning: One sentence summary failed "
                            f"({one_sentence_text}); skipping file save.[/yellow]"
                        )
                    elif one_sentence_text and one_sentence_summaries_dir:
                        os_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"one-sentence-summary (from {transcript_arg}){lang_str}.md"
                        )
                        target_path = os.path.join(
                            one_sentence_summaries_dir, os_filename
                        )
                        try:
                            os_full_path = storage.write_text(
                                target_path, one_sentence_text
                            )
                            rprint(
                                "Saved one sentence summary: "
                                f"{format_clickable_path(os_full_path)}"
                            )
                            os_col = (
                                f"One Sentence Summary File {model_name} from "
                                f"{transcript_arg}{col_suffix}"
                            )
                            row[os_col] = os_full_path
                        except Exception as e:
                            print(f"Error writing one sentence summary: {e}")

                    # Cost
                    if verbose:
                        input_price, output_price = get_model_pricing(model_name)
                        if input_price is not None and output_price is not None:
                            cost = (os_input / 1_000_000) * input_price + (
                                os_output / 1_000_000
                            ) * output_price
                            cost = round(cost, 2)
                            row[one_sentence_cost_col_name] = cost
                            vprint(f"One sentence summary cost: ${cost:.2f}")

                # Tag Generation
                tags_col_name = f"Tags {transcript_arg} {model_name} model{col_suffix}"
                tags_cost_col_name = (
                    f"{normalize_model_name(model_name)} "
                    f"tags cost from {transcript_arg}{col_suffix} ($)"
                )

                if (
                    (
                        not row.get(tags_col_name)
                        or is_error_response(row.get(tags_col_name))
                    )
                    and row.get(summary_col_name)
                    and not is_error_response(row.get(summary_col_name))
                ):
                    summary_for_tags = row[summary_col_name]
                    rprint(f"Generating tags using model: {model_name} ({language})")

                    tags_text, tags_input, tags_output = generate_tags(
                        model_name, summary_for_tags, language=language
                    )

                    if is_error_response(tags_text):
                        row[tags_col_name] = tags_text
                        rprint(
                            f"[yellow]Warning: Tag generation failed "
                            f"({tags_text}); skipping file save.[/yellow]"
                        )
                    else:
                        # Ensure no more than 5 tags
                        tag_list = [
                            t.strip() for t in tags_text.split(",") if t.strip()
                        ]
                        if len(tag_list) > 5:
                            tag_list = tag_list[:5]
                        row[tags_col_name] = ", ".join(tag_list)

                        if (
                            tags_text.strip() == "nan"
                            or tags_text.strip() == 'float("nan")'
                        ):
                            row[tags_col_name] = float("nan")

                    # Calculate Tags Cost
                    if verbose:
                        input_price, output_price = get_model_pricing(model_name)
                        if input_price is not None and output_price is not None:
                            tags_cost = (tags_input / 1_000_000) * input_price + (
                                tags_output / 1_000_000
                            ) * output_price
                            row[tags_cost_col_name] = tags_cost
                            vprint(f"Tags cost: ${tags_cost:.2f}")

                    # Save Tags File
                    if (
                        row.get(tags_col_name)
                        and not is_error_response(row.get(tags_col_name))
                        and tags_dir
                    ):
                        tags_val = row[tags_col_name]
                        if isinstance(tags_val, str) and tags_val != 'float("nan")':
                            tags_filename = (
                                f"{model_name} - {video_id} - {safe_title} - "
                                f"tags (from {transcript_arg}){lang_str}.txt"
                            )
                            target_path = os.path.join(tags_dir, tags_filename)
                            try:
                                tags_full_path = storage.write_text(
                                    target_path, tags_val
                                )
                                rprint(
                                    "Saved tags: "
                                    f"{format_clickable_path(tags_full_path)}"
                                )
                                tags_file_col = (
                                    f"Tags File {transcript_arg} {model_name} "
                                    f"model{col_suffix}"
                                )
                                row[tags_file_col] = tags_full_path
                            except Exception as e:
                                print(f"Error writing tags: {e}")

                write_description(model_name)

                # --- Secondary Speaker Extraction from YouTube (if applicable) ---
                yt_speakers_text = 'float("nan")'
                yt_speakers_input = 0
                yt_speakers_output = 0

                if (
                    transcript_arg != "youtube"
                    and youtube_transcript
                    and not no_youtube_summary
                ):
                    yt_speakers_col_name = f"Speakers {model_name} from youtube"
                    yt_speakers_file_col_name = (
                        f"Speakers File {model_name} from youtube"
                    )
                    yt_speaker_cost_col_name = (
                        f"{normalize_model_name(model_name)} "
                        f"Speaker extraction cost from youtube ($)"
                    )

                    # Check disk for YT speakers file
                    if not row.get(yt_speakers_file_col_name):
                        yt_speakers_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"speakers (from youtube).txt"
                        )
                        expected_path = os.path.join(speakers_dir, yt_speakers_filename)
                        if storage.exists(expected_path):
                            row[yt_speakers_file_col_name] = expected_path

                    # Load YT speakers from file/row
                    if row.get(yt_speakers_file_col_name):
                        path = row[yt_speakers_file_col_name]
                        if path and storage.exists(str(path)):
                            try:
                                yt_speakers_text = storage.read_text(str(path))
                                row[yt_speakers_col_name] = yt_speakers_text
                            except Exception as e:
                                vprint(
                                    "Warning: Failed to read YouTube speakers file "
                                    f"{path}: {e}"
                                )

                    elif row.get(yt_speakers_col_name) and not is_error_response(
                        row.get(yt_speakers_col_name)
                    ):
                        yt_speakers_text = row[yt_speakers_col_name]

                    # Generate if missing (checking specifically if it is 'float("nan")'
                    # default or actual text)
                    if (
                        yt_speakers_text == 'float("nan")'
                        or is_error_response(yt_speakers_text)
                    ) and (
                        not row.get(yt_speakers_col_name)
                        or is_error_response(row.get(yt_speakers_col_name))
                    ):
                        # Try to use English transcript for YT speaker extraction
                        # if available
                        yt_speaker_source_transcript = youtube_transcript
                        if language != "en":
                            en_path = row.get(
                                "Transcript File human generated"
                            ) or row.get("Transcript File youtube generated")
                            if en_path:
                                if storage.exists(str(en_path)):
                                    yt_speaker_source_transcript = storage.read_text(
                                        str(en_path)
                                    )
                                    vprint(
                                        "Using English transcript for YouTube "
                                        f"speaker extraction ({model_name})."
                                    )
                                elif os.path.exists(str(en_path)):
                                    with open(str(en_path), "r", encoding="utf-8") as f:
                                        yt_speaker_source_transcript = f.read()
                                        vprint(
                                            "Using English transcript for YouTube "
                                            f"speaker extraction ({model_name})."
                                        )
                        rprint(
                            f"Extracting speakers using model: {model_name} "
                            "(Source: YouTube Transcript)"
                        )
                        (
                            yt_speakers_text,
                            yt_speakers_input,
                            yt_speakers_output,
                        ) = extract_speakers(model_name, yt_speaker_source_transcript)

                        row[yt_speakers_col_name] = yt_speakers_text
                        if (
                            yt_speakers_text.strip() == "nan"
                            or yt_speakers_text.strip() == 'float("nan")'
                        ):
                            row[yt_speakers_col_name] = float("nan")

                        # Save YouTube Speakers File
                        if is_error_response(yt_speakers_text):
                            rprint(
                                f"[yellow]Warning: YouTube speaker extraction failed "
                                f"({yt_speakers_text}); skipping file save.[/yellow]"
                            )
                            row[yt_speakers_file_col_name] = ""
                        elif yt_speakers_text and not isinstance(
                            row[yt_speakers_col_name], float
                        ):
                            yt_speakers_filename = (
                                f"{model_name} - {video_id} - {safe_title} - "
                                f"speakers (from youtube).txt"
                            )
                            target_path = os.path.join(
                                speakers_dir, yt_speakers_filename
                            )

                            try:
                                saved_path = storage.write_text(
                                    target_path, yt_speakers_text
                                )
                                rprint(
                                    f"Saved YouTube speakers: {yt_speakers_filename}"
                                )
                                row[yt_speakers_file_col_name] = saved_path
                            except Exception as e:
                                print(f"Error writing YouTube speakers file: {e}")

                        # Calculate YouTube Speaker Cost
                        if verbose:
                            input_price, output_price = get_model_pricing(model_name)
                            if input_price is not None and output_price is not None:
                                yt_speaker_cost = (
                                    yt_speakers_input / 1_000_000
                                ) * input_price + (
                                    yt_speakers_output / 1_000_000
                                ) * output_price
                                yt_speaker_cost = round(yt_speaker_cost, 2)
                                row[yt_speaker_cost_col_name] = yt_speaker_cost
                                vprint(
                                    "YouTube Speaker extraction cost: "
                                    f"${yt_speaker_cost:.2f}"
                                )

                # --- Secondary Q&A from YouTube (if applicable) ---
                if (
                    transcript_arg != "youtube"
                    and youtube_transcript
                    and not no_youtube_summary
                ):
                    yt_qa_col_name = f"QA Text {model_name} from youtube{col_suffix}"
                    yt_qa_file_col_name = (
                        f"QA File {model_name} from youtube{col_suffix}"
                    )
                    yt_qa_cost_col_name = (
                        f"{normalize_model_name(model_name)} QA cost from "
                        f"youtube{col_suffix} ($)"
                    )

                    # Check disk for YT QA file
                    if not row.get(yt_qa_file_col_name):
                        qa_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"qa (from youtube){lang_str}.md"
                        )
                        expected_path = os.path.join(qa_dir, qa_filename)
                        if storage.exists(expected_path):
                            row[yt_qa_file_col_name] = expected_path

                    # Load YT QA from file/row
                    if row.get(yt_qa_file_col_name):
                        path = row[yt_qa_file_col_name]
                        if path:
                            try:
                                row[yt_qa_col_name] = storage.read_text(str(path))
                            except Exception as e:
                                vprint(
                                    "Warning: Failed to read YouTube Q&A file "
                                    f"{path}: {e}"
                                )
                    if not row.get(yt_qa_col_name) or is_error_response(
                        row.get(yt_qa_col_name)
                    ):
                        rprint(
                            f"Generating Q&A using model: {model_name} "
                            "(Source: YouTube Transcript)"
                        )
                        yt_qa_text, yt_qa_in, yt_qa_out = generate_qa(
                            model_name,
                            youtube_transcript,
                            yt_speakers_text,
                            url,
                            language=language,
                        )

                        row[yt_qa_col_name] = yt_qa_text
                        if (
                            yt_qa_text.strip() == "nan"
                            or yt_qa_text.strip() == 'float("nan")'
                        ):
                            row[yt_qa_col_name] = float("nan")

                        if verbose:
                            input_price, output_price = get_model_pricing(model_name)
                            if input_price is not None and output_price is not None:
                                # Pure QA cost
                                cost = (yt_qa_in / 1_000_000) * input_price + (
                                    yt_qa_out / 1_000_000
                                ) * output_price
                                yt_qa_cost = round(cost, 2)
                                vprint(f"YouTube Q&A cost: ${yt_qa_cost:.2f}")
                                row[yt_qa_cost_col_name] = yt_qa_cost

                        yt_qa_full_path = ""
                        if is_error_response(yt_qa_text):
                            rprint(
                                f"[yellow]Warning: YouTube Q&A generation failed "
                                f"({yt_qa_text}); skipping file save.[/yellow]"
                            )
                        elif row[yt_qa_col_name] and not isinstance(
                            row[yt_qa_col_name], float
                        ):
                            qa_filename = (
                                f"{model_name} - {video_id} - {safe_title} - "
                                f"qa (from youtube){lang_str}.md"
                            )
                            target_path = os.path.join(qa_dir, qa_filename)

                            try:
                                yt_qa_full_path = storage.write_text(
                                    target_path, row[yt_qa_col_name]
                                )
                                rprint(
                                    f"Saved YouTube Q&A: "
                                    f"{format_clickable_path(yt_qa_full_path)}"
                                )
                            except Exception as e:
                                print(f"Error writing YouTube Q&A: {e}")

                        row[yt_qa_file_col_name] = yt_qa_full_path

                # --- Secondary Summary from YouTube (if applicable) ---
                if (
                    transcript_arg != "youtube"
                    and youtube_transcript
                    and not no_youtube_summary
                ):
                    yt_sum_col_name = (
                        f"Summary Text {model_name} from youtube{col_suffix}"
                    )
                    yt_sum_file_col_name = (
                        f"Summary File {model_name} from youtube{col_suffix}"
                    )
                    yt_sum_cost_col_name = (
                        f"{normalize_model_name(model_name)} summary cost from "
                        f"youtube{col_suffix} ($)"
                    )
                    yt_sum_prompt_file_col_name = (
                        f"Summary Prompt Path {model_name} from youtube{col_suffix}"
                    )

                    # Save the exact prompt sent to the LLM before generating.
                    if not row.get(yt_sum_prompt_file_col_name):
                        yt_summary_prompt_text = build_summary_prompt(
                            youtube_transcript, video_title, url, language=language
                        )
                        summary_prompt_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"summary-prompt (from youtube){lang_str}.txt"
                        )
                        target_path = os.path.join(
                            summary_prompts_dir, summary_prompt_filename
                        )
                        try:
                            saved_prompt_path = storage.write_text(
                                target_path, yt_summary_prompt_text
                            )
                            rprint(
                                "Saved summary prompt: "
                                f"{format_clickable_path(saved_prompt_path)}"
                            )
                            row[yt_sum_prompt_file_col_name] = saved_prompt_path
                        except Exception as e:
                            print(f"Error writing summary prompt: {e}")

                    # Check disk for YT Summary file
                    if not row.get(yt_sum_file_col_name):
                        summary_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"summary (from youtube){lang_str}.md"
                        )
                        expected_path = os.path.join(summaries_dir, summary_filename)
                        if storage.exists(expected_path):
                            row[yt_sum_file_col_name] = expected_path

                    # Load YT Summary from file/row
                    if row.get(yt_sum_file_col_name):
                        path = row[yt_sum_file_col_name]
                        if path:
                            try:
                                row[yt_sum_col_name] = storage.read_text(str(path))
                            except Exception as e:
                                vprint(
                                    "Warning: Failed to read YouTube summary file "
                                    f"{path}: {e}"
                                )

                    if not row.get(yt_sum_col_name) or is_error_response(
                        row.get(yt_sum_col_name)
                    ):
                        rprint(
                            f"Generating summary using model: {model_name} "
                            "(Source: YouTube Transcript)"
                        )
                        (
                            yt_summary_text,
                            yt_input_tokens,
                            yt_output_tokens,
                        ) = generate_summary(
                            model_name,
                            youtube_transcript,
                            video_title,
                            url,
                            language=language,
                        )

                        if verbose:
                            input_price, output_price = get_model_pricing(model_name)
                            if input_price is not None and output_price is not None:
                                # We don't include speaker tokens here
                                # as we didn't extract
                                # speakers from the YouTube transcript
                                # specifically for this
                                # summary.
                                # If we wanted to be precise,
                                # we'd need to extract speakers
                                # from YT transcript too.
                                # For now, just the summary cost.
                                cost = (yt_input_tokens / 1_000_000) * input_price + (
                                    yt_output_tokens / 1_000_000
                                ) * output_price
                                yt_summary_cost = round(cost, 2)
                                vprint(f"YouTube Summary cost: ${yt_summary_cost:.2f}")
                                row[yt_sum_cost_col_name] = yt_summary_cost

                        yt_summary_full_path = ""
                        if is_error_response(yt_summary_text):
                            rprint(
                                f"[yellow]Warning: YouTube summary generation failed "
                                f"({yt_summary_text}); skipping file save.[/yellow]"
                            )
                        elif summaries_dir and yt_summary_text:
                            summary_filename = (
                                f"{model_name} - {video_id} - {safe_title} - "
                                f"summary (from youtube){lang_str}.md"
                            )
                            target_path = os.path.join(summaries_dir, summary_filename)

                            try:
                                yt_summary_full_path = storage.write_text(
                                    target_path, yt_summary_text
                                )
                                rprint(
                                    f"Saved YouTube summary: "
                                    f"{format_clickable_path(yt_summary_full_path)}"
                                )
                            except Exception as e:
                                print(f"Error writing YouTube summary: {e}")

                        row[yt_sum_file_col_name] = yt_summary_full_path
                        row[yt_sum_col_name] = yt_summary_text

                    # One Sentence Summary for YouTube Summary
                    yt_one_sentence_col_name = (
                        f"One Sentence Summary {model_name} from youtube{col_suffix}"
                    )
                    yt_one_sentence_cost_col_name = (
                        f"{normalize_model_name(model_name)} one sentence summary "
                        f"cost from youtube{col_suffix} ($)"
                    )

                    if (
                        row.get(yt_sum_col_name)
                        and not is_error_response(row.get(yt_sum_col_name))
                        and (
                            not row.get(yt_one_sentence_col_name)
                            or is_error_response(row.get(yt_one_sentence_col_name))
                        )
                    ):
                        rprint(
                            "Generating one sentence summary using model: "
                            f"{model_name} (Source: YouTube Transcript)"
                        )
                        (
                            yt_one_sentence_text,
                            yt_os_input,
                            yt_os_output,
                        ) = generate_one_sentence_summary(
                            model_name, row[yt_sum_col_name], language=language
                        )
                        row[yt_one_sentence_col_name] = yt_one_sentence_text

                        # Cost
                        if verbose:
                            input_price, output_price = get_model_pricing(model_name)
                            if input_price is not None and output_price is not None:
                                cost = (yt_os_input / 1_000_000) * input_price + (
                                    yt_os_output / 1_000_000
                                ) * output_price
                                cost = round(cost, 2)
                                row[yt_one_sentence_cost_col_name] = cost
                                vprint(
                                    f"YouTube one sentence summary cost: ${cost:.2f}"
                                )

                        # Save YT One Sentence Summary File
                        if is_error_response(yt_one_sentence_text):
                            rprint(
                                "[yellow]Warning: YouTube one sentence summary failed "
                                f"({yt_one_sentence_text}); "
                                "skipping file save.[/yellow]"
                            )
                        elif yt_one_sentence_text and one_sentence_summaries_dir:
                            os_filename = (
                                f"{model_name} - {video_id} - {safe_title} - "
                                f"one-sentence-summary (from youtube){lang_str}.md"
                            )
                            target_path = os.path.join(
                                one_sentence_summaries_dir, os_filename
                            )
                            try:
                                os_full_path = storage.write_text(
                                    target_path, yt_one_sentence_text
                                )
                                rprint(
                                    "Saved YouTube one sentence summary: "
                                    f"{format_clickable_path(os_full_path)}"
                                )
                                os_col = (
                                    f"One Sentence Summary File {model_name} from "
                                    f"youtube{col_suffix}"
                                )
                                row[os_col] = os_full_path
                            except Exception as e:
                                print(
                                    f"Error writing YouTube one sentence summary: {e}"
                                )

            # Infographic Generation
            if infographic_models:
                summary_targets = []

                # Target ALL summaries in the row (both existing and newly created)
                # This includes normal summaries and "from youtube" summaries
                for k in list(row.keys()):
                    if (
                        k.startswith("Summary Text ")
                        and row[k]
                        and not is_error_response(row[k])
                    ):
                        # Check language
                        if language != "en" and not k.endswith(f" ({language})"):
                            continue
                        if language == "en" and k.endswith(")"):
                            # Skip other languages
                            continue

                        m_name = k[len("Summary Text ") :]
                        # m_name might be "gemini-2.0-flash" or
                        # "gemini-2.0-flash from youtube"
                        # or "gemini-2.0-flash (es)" or
                        # "gemini-2.0-flash from youtube (es)"

                        summary_targets.append((k, m_name, row[k]))

                for (sum_col, m_name, s_text), infographic_model in product(
                    summary_targets, infographic_models
                ):
                    if not s_text or is_error_response(s_text):
                        continue

                    info_col = f"Summary Infographic File {m_name} {infographic_model}"

                    alt_text_col = (
                        f"Summary Infographic Alt Text {m_name} {infographic_model}"
                    )
                    alt_text_file_col = (
                        f"Infographic Alt Text Path {m_name} {infographic_model}"
                    )
                    prompt_file_col = (
                        f"Infographic Prompt Path {m_name} {infographic_model}"
                    )

                    safe_title = re.sub(r"[\\/*?:\"<>|]", "_", video_title).replace(
                        "\n", " "
                    )
                    safe_title = safe_title.replace("\r", "")

                    # Save the exact prompt sent to the image model. Done before
                    # generation so the prompt is recorded even if image
                    # generation later fails.
                    if not row.get(prompt_file_col):
                        prompt_text = build_infographic_prompt(
                            s_text, video_title, language=language
                        )
                        prompt_filename = (
                            f"{m_name} - {infographic_model} - {video_id} - "
                            f"{safe_title} - infographic-prompt.txt"
                        )
                        prompt_target_path = os.path.join(
                            infographic_prompts_dir, prompt_filename
                        )
                        try:
                            saved_prompt_path = storage.write_text(
                                prompt_target_path, prompt_text
                            )
                            rprint(
                                "Saved infographic prompt: "
                                f"{format_clickable_path(saved_prompt_path)}"
                            )
                            row[prompt_file_col] = saved_prompt_path
                        except Exception as e:
                            print(f"Error writing infographic prompt: {e}")

                    # 1. Check if both already exist in row
                    if row.get(info_col) and row.get(alt_text_col):
                        continue

                    # 2. Check disk for infographic (saved with the extension of
                    # the image format the model returned)
                    infographic_stem = os.path.join(
                        infographics_dir,
                        f"{m_name} - {infographic_model} - {video_id} - "
                        f"{safe_title} - infographic",
                    )
                    expected_path = next(
                        (
                            infographic_stem + ext
                            for ext in (".png", ".jpg", ".webp")
                            if storage.exists(infographic_stem + ext)
                        ),
                        None,
                    )

                    image_bytes = None
                    if expected_path:
                        row[info_col] = expected_path
                        # If we need alt text, we need the bytes
                        if not row.get(alt_text_col):
                            vprint(
                                "Loading existing infographic for alt text: "
                                f"{format_clickable_path(expected_path)}"
                            )
                            image_bytes = storage.read_bytes(expected_path)
                    else:
                        # 3. Generate infographic if it doesn't exist on disk
                        summary_file_path = row.get(f"Summary File {m_name}", "")
                        summary_filename = (
                            storage.get_name(summary_file_path)
                            if summary_file_path
                            else "unknown file"
                        )
                        rprint(
                            f"Generating infographic using model {infographic_model} "
                            f"from {summary_filename}"
                        )
                        image_bytes, input_tokens, output_tokens = generate_infographic(
                            infographic_model, s_text, video_title, language=language
                        )
                        if image_bytes:
                            try:
                                saved_path = storage.write_bytes(
                                    infographic_stem
                                    + image_file_extension(image_bytes),
                                    image_bytes,
                                )
                                rprint(
                                    "Saved infographic: "
                                    f"{format_clickable_path(saved_path)}"
                                )
                                row[info_col] = saved_path

                                # Calculate Infographic Cost
                                if verbose:
                                    input_price, output_price = get_model_pricing(
                                        infographic_model
                                    )
                                    if (
                                        input_price is not None
                                        and output_price is not None
                                    ):
                                        cost = (
                                            input_tokens / 1_000_000
                                        ) * input_price + (
                                            output_tokens / 1_000_000
                                        ) * output_price
                                        cost = round(cost, 2)
                                        cost_col = (
                                            f"Summary Infographic Cost {m_name} "
                                            f"{infographic_model} ($)"
                                        )
                                        row[cost_col] = cost
                                        vprint(f"Infographic cost: ${cost:.2f}")

                            except Exception as e:
                                print(f"Error writing infographic: {e}")

                    # 4. Alt Text Generation
                    if image_bytes and not row.get(alt_text_col):
                        alt_text_model = (
                            alt_text_model_arg or model_name
                        )  # model_name is the current summary model
                        rprint(
                            "Generating multimodal alt text using model: "
                            f"{alt_text_model}"
                        )
                        # Use the unified provider
                        provider = get_provider(alt_text_model)
                        if isinstance(provider, MultimodalProvider):
                            alt_text, at_input, at_output = provider.generate_alt_text(
                                image_bytes, language=language
                            )
                        else:
                            alt_text = (
                                f"Error: {alt_text_model} does not support "
                                "multimodal alt text."
                            )
                            at_input, at_output = 0, 0

                        row[alt_text_col] = alt_text

                        # Save Alt Text File
                        if alt_text and not is_error_response(alt_text):
                            alt_text_filename = (
                                f"{m_name} - {infographic_model} - {video_id} - "
                                f"{safe_title} - alt-text.md"
                            )
                            target_path = os.path.join(alt_text_dir, alt_text_filename)
                            try:
                                saved_path = storage.write_text(target_path, alt_text)
                                rprint(
                                    "Saved alt text: "
                                    f"{format_clickable_path(saved_path)}"
                                )
                                row[alt_text_file_col] = saved_path
                            except Exception as e:
                                print(f"Error writing alt text: {e}")

                        # Cost
                        if verbose:
                            input_price, output_price = get_model_pricing(
                                alt_text_model
                            )
                            if input_price is not None and output_price is not None:
                                at_cost = (at_input / 1_000_000) * input_price + (
                                    at_output / 1_000_000
                                ) * output_price
                                at_cost = round(at_cost, 2)
                                at_cost_col = (
                                    "Summary Infographic Alt Text Cost "
                                    f"{m_name} {infographic_model} ($)"
                                )
                                row[at_cost_col] = at_cost
                                vprint(f"Alt text cost: ${at_cost:.2f}")

        # --- Corrected Captions ---
        if corrected_captions_model:
            corrected = process_corrected_captions(
                row=row,
                storage=storage,
                model_name=corrected_captions_model,
                source=corrected_captions_source,
                transcript_arg=transcript_arg,
                video_id=video_id,
                video_title=video_title,
                safe_title=safe_title,
                url=url,
                dirs={
                    "srt": srt_dir,
                    "transcript": transcripts_dir,
                    "vtt": vtt_dir,
                    "video_context": video_context_dir,
                },
                verbose=verbose,
                vprint=vprint,
            )
            if corrected:
                original_path = os.path.join(
                    video_dir, f"{video_id} - {safe_title} - original video.mp4"
                )
                temp_video_dir = os.path.join(local_temp_dir, "video-files")
                downloaded: list[str | None] = []

                def get_video() -> str | None:
                    """The video, downloaded once (or the saved original)."""
                    if not downloaded:
                        local = (
                            storage.get_local_file(
                                original_path, download_dir=temp_video_dir
                            )
                            if storage.exists(original_path)
                            else None
                        )
                        if not local:
                            rprint(f"Downloading video {video_id}...")
                            local = download_video(video_id, temp_video_dir)
                        downloaded.append(local)
                    return downloaded[0]

                save_original_video(
                    row=row, storage=storage, get_video=get_video, path=original_path
                )
                if downloaded and not downloaded[0]:
                    rprint(
                        "[yellow]Couldn't download the video, so skipped the audio "
                        "description, the captioned and described video and the "
                        "player page.[/yellow]"
                    )
                else:
                    descriptions = process_audio_description(
                        row=row,
                        storage=storage,
                        model_name=corrected_captions_model,
                        captions=corrected,
                        get_video=get_video,
                        tts_arg=tts_arg,
                        video_id=video_id,
                        video_title=video_title,
                        safe_title=safe_title,
                        dirs={
                            "vtt": vtt_dir,
                            "descriptive_transcripts": descriptive_transcripts_dir,
                        },
                        duration_seconds=video_duration_seconds,
                        verbose=verbose,
                        vprint=vprint,
                    )
                    ad_model = corrected_captions_model
                    descriptions_vtt = row.get(f"Descriptions VTT File {ad_model}")
                    process_accessible_video(
                        row=row,
                        storage=storage,
                        captions_path=corrected.path,
                        model_name=corrected_captions_model,
                        source=corrected.source,
                        descriptions=descriptions,
                        descriptions_vtt=descriptions_vtt,
                        tts_arg=tts_arg,
                        get_video=get_video,
                        video_id=video_id,
                        safe_title=safe_title,
                        video_dir=video_dir,
                        duration_seconds=video_duration_seconds,
                        vprint=vprint,
                    )
                    # player.html plays the saved video; embed.html streams
                    # it from YouTube and needs no other files, for pasting
                    # into a site (e.g. Google Sites' Embed code).
                    for page, youtube_id, column in (
                        ("player", None, "Player Page"),
                        ("embed", video_id, "Embed Page"),
                    ):
                        write_player_page(
                            row=row,
                            storage=storage,
                            path=os.path.join(
                                vtt_dir,
                                f"{corrected_captions_model} - {video_id} - "
                                f"{safe_title} - {page}.html",
                            ),
                            title=video_title,
                            video_path=original_path,
                            captions_path=row.get(
                                f"Captions VTT File {corrected_captions_model} "
                                f"from {corrected.source}",
                                "",
                            ),
                            descriptions_path=row.get(
                                f"Extended Descriptions VTT File {ad_model}"
                            ),
                            transcript_path=row.get(
                                f"Descriptive Transcript File {ad_model}"
                            ),
                            youtube_id=youtube_id,
                            column=column,
                        )
                # Downloads are large; don't keep them across a playlist (the
                # saved original stays in video-files/).
                for path in downloaded:
                    if path and os.path.abspath(path).startswith(
                        os.path.abspath(temp_video_dir)
                    ):
                        os.remove(path)

        if not verbose:
            oss = next(
                (
                    v
                    for k, v in row.items()
                    if "One Sentence Summary" in k and isinstance(v, str)
                ),
                None,
            )
            if oss:
                rprint(f"Summary: {oss}")
            else:
                trans = next(
                    (
                        v
                        for k, v in row.items()
                        if "Transcript File" in k and isinstance(v, str)
                    ),
                    None,
                )
                if trans:
                    rprint(f"Transcript: {format_clickable_path(trans)}")

        if outfile.lower() in ("none", "n"):
            rprint("\n[bold green]Results (Not Saved):[/bold green]")
            for key, value in row.items():
                if value and not str(value).lower() == "nan":
                    rprint(f"[bold]{key}:[/bold] {value}")

        rows.append(row)

        # Save progress after each video
        try:
            # Create a DataFrame from the rows processed so far
            current_rows_df = pl.DataFrame(rows)

            # Combine with existing data
            if existing_df is not None:
                # Identify URLs processed in this session
                processed_urls = current_rows_df["URL"].to_list()
                # Keep rows from existing_df that haven't been re-processed
                existing_remaining = existing_df.filter(
                    ~pl.col("URL").is_in(processed_urls)
                )
                current_save_df = pl.concat(
                    [existing_remaining, current_rows_df], how="diagonal"
                )
            else:
                current_save_df = current_rows_df

            if "Data Published" in current_save_df.columns:
                current_save_df = current_save_df.sort(
                    "Data Published", descending=True
                )

            current_save_df = reorder_columns(current_save_df)
            storage.save_dataframe(current_save_df, outfile_path)
            vprint(f"Progress saved to {outfile}")
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")

        time.sleep(1)
        print()

    final_df = None

    if rows:
        new_df = pl.DataFrame(rows)
        if existing_df is not None:
            processed_urls = new_df["URL"].to_list()
            existing_remaining = existing_df.filter(
                ~pl.col("URL").is_in(processed_urls)
            )
            final_df = pl.concat([existing_remaining, new_df], how="diagonal")
        else:
            final_df = new_df
    elif existing_df is not None:
        final_df = existing_df

    if final_df is not None and not final_df.is_empty():
        should_save = bool(rows)

        if "Data Published" in final_df.columns:
            final_df = final_df.sort("Data Published", descending=True)

        # Intermediate save
        if should_save:
            temp_df = reorder_columns(final_df)
            # We use the same path, updating the sheet
            intermediate_path = storage.save_dataframe(temp_df, outfile_path)
            vprint(f"Intermediate save (pre-TTS/Video): {intermediate_path}")

        if tts_arg:
            rprint("Checking for TTS generation...")
            # TTS will scan all columns, so it should pick up the new
            # language columns too
            final_df = process_tts(
                final_df, tts_arg, storage, base_dir, languages=languages
            )
            should_save = True

        if combine_info_audio:
            rprint("Checking for Video generation...")
            final_df = process_videos(final_df, storage, base_dir)
            should_save = True

        if should_save:
            final_df = reorder_columns(final_df)
            saved_path = storage.save_dataframe(final_df, outfile_path)
            rprint(
                f"Successfully wrote {len(rows)} new rows to storage. "
                f"Total rows: {len(final_df)} at {format_clickable_path(saved_path)}"
            )
        else:
            vprint("No new data to gather or all videos already processed.")
    else:
        vprint("No new data to gather or all videos already processed.")

    # Cleanup local temp dir. Another run in the same directory may remove it
    # first, so a missing directory isn't an error.
    import shutil

    shutil.rmtree(local_temp_dir, ignore_errors=True)

    if isinstance(storage, MemoryStorage):
        return storage
    return None


if __name__ == "__main__":
    main()
