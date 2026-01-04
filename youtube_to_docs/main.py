import argparse
import logging
import os
import re
import time

import polars as pl
from rich import print as rprint
from rich_argparse import RichHelpFormatter

from youtube_to_docs.infographic import generate_infographic
from youtube_to_docs.llms import (
    extract_speakers,
    generate_one_sentence_summary,
    generate_qa,
    generate_summary,
    generate_transcript,
    get_model_pricing,
)
from youtube_to_docs.models import MODEL_SUITES
from youtube_to_docs.storage import GoogleDriveStorage, LocalStorage, M365Storage
from youtube_to_docs.transcript import (
    extract_audio,
    fetch_transcript,
    get_video_details,
    get_youtube_service,
    resolve_video_ids,
)
from youtube_to_docs.tts import process_tts
from youtube_to_docs.utils import (
    format_clickable_path,
    normalize_model_name,
    reorder_columns,
)
from youtube_to_docs.video import process_videos


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


def main(args_list: list[str] | None = None) -> None:
    # Define styles for the help output
    RichHelpFormatter.styles["argparse.args"] = "cyan italic"
    RichHelpFormatter.styles["argparse.groups"] = "bold yellow"
    RichHelpFormatter.styles["argparse.metavar"] = "bold magenta"

    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    parser.add_argument(
        "video_id",
        nargs="?",
        default="atmGAHYpf_c",
        help=(
            "Can be one of: \n"
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
            "`sharepoint` or `s` to store to Microsoft SharePoint."
        ),
    )
    parser.add_argument(
        "-t",
        "--transcript",
        default="youtube",
        help=(
            "The transcript source to use. \n"
            "Can be `youtube` (default) to fetch existing YouTube transcripts, \n"
            "or an AI model name (e.g. `gemini-3-flash-preview`) to perform STT on "
            "extracted audio."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help=(
            "The LLM to use for speaker extraction, Q&A generation, and "
            "summarization.\n"
            "Can be one of: \n"
            "Gemini model (e.g., `gemini-3-flash-preview`)\n"
            "GCP Vertex model (prefixed with `vertex-`; e.g. "
            "`vertex-claude-haiku-4-5@20251001`)\n"
            "AWS Bedrock model (prefixed with `bedrock-`; e.g. "
            "`bedrock-claude-haiku-4-5-20251001-v1`\n"
            "`bedrock-nova-2-lite-v1`)\n"
            "Azure Foundry model (prefix with `foundry-`; e.g. "
            "`foundry-gpt-5-mini`)\n"
            "Can also be a comma-separated list of models (e.g. "
            "`gemini-3-flash-preview,bedrock-claude-haiku-4-5-20251001-v1`).\n"
            "Defaults to `None`."
        ),
    )
    parser.add_argument(
        "--tts",
        default=None,
        help=(
            "The TTS model and voice to use. "
            "Format: `{model}-{voice}` e.g. `gemini-2.5-flash-preview-tts-Kore` \n"
            "or `gemini-2.5-pro-preview-tts-Kore`"
        ),
    )
    parser.add_argument(
        "-i",
        "--infographic",
        default=None,
        help=(
            "The image model to use for generating an infographic. "
            "e.g. `gemini-2.5-flash-image` or `gemini-3-pro-image-preview`"
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
        "-l",
        "--language",
        default="en",
        help=("The target language (e.g. `es`, `fr`, `en`). Default is `en`."),
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
            "`gemini-flash`: summarization (`gemini-3-flash-preview`), "
            "TTS (`gemini-2.5-flash-preview-tts-Kore`), "
            "and Infographic (`gemini-2.5-flash-image`). \n"
            "`gemini-pro`: summarization (`gemini-3-pro-preview`), "
            "TTS (`gemini-2.5-pro-preview-tts-Kore`), "
            "and Infographic (`gemini-3-pro-image-preview`). \n"
            "`gemini-flash-pro-image`: summarization (`gemini-3-flash-preview`), "
            "TTS (`gemini-2.5-flash-preview-tts-Kore`), "
            "and Infographic (`gemini-3-pro-image-preview`). \n"
            "Also sets `--no-youtube-summary`."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output.",
    )

    args = parser.parse_args(args_list)

    verbose = args.verbose

    if not verbose:
        # Suppress "Both GOOGLE_API_KEY and GEMINI_API_KEY are set" warning
        # from google-genai library unless verbose is enabled.
        logging.getLogger("google_genai._api_client").setLevel(logging.ERROR)

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
        if args.transcript == "youtube":
            args.transcript = suite["transcript"]
        args.no_youtube_summary = True

    transcript_arg = args.transcript
    video_id_input = args.video_id
    outfile = args.outfile
    model_names_arg = args.model
    tts_arg = args.tts
    infographic_arg = args.infographic
    no_youtube_summary = args.no_youtube_summary
    language_arg = args.language

    combine_info_audio = args.combine_infographic_audio
    model_names = model_names_arg.split(",") if model_names_arg else []
    languages = language_arg.split(",") if language_arg else ["en"]

    youtube_service = get_youtube_service()

    video_ids = resolve_video_ids(video_id_input, youtube_service)

    # Setup Output Directories
    # Setup Storage
    if outfile in ("sharepoint", "s"):
        vprint(f"Using SharePoint storage. Output: {outfile}")
        storage = M365Storage()
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

    # Local temp dir for processing (Audio/TTS require local files)
    local_temp_dir = "temp_processing_artifacts"
    local_audio_dir = os.path.join(local_temp_dir, "audio-files")
    os.makedirs(local_audio_dir, exist_ok=True)

    storage.ensure_directory(transcripts_dir)
    storage.ensure_directory(summaries_dir)
    storage.ensure_directory(infographics_dir)
    storage.ensure_directory(speakers_dir)
    storage.ensure_directory(qa_dir)
    storage.ensure_directory(audio_dir)
    storage.ensure_directory(video_dir)

    # Load existing CSV if it exists
    existing_df = storage.load_dataframe(outfile_path)
    if existing_df is not None:
        vprint(f"Loaded existing data from {outfile} ({len(existing_df)} rows)")
    else:
        vprint(f"No existing data found at {outfile}. Starting fresh.")

    vprint(f"Processing {len(video_ids)} videos.")
    vprint(f"Processing Videos: {video_ids}")
    vprint(f"Saving to: {outfile}")
    if model_names:
        vprint(f"Summarizing using models: {model_names}")
    vprint(f"Target Languages: {languages}")

    rows = []

    for i, video_id in enumerate(video_ids, 1):
        url = f"https://www.youtube.com/watch?v={video_id}"
        vprint(f"Processing Video ID: {video_id}")

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
            description = row.get("Description", "")
            publishedAt = row.get("Data Published", "")
            channelTitle = row.get("Channel", "")
            tags = row.get("Tags", "")
            video_duration = row.get("Duration", "")

        display_title = video_title if video_title else video_id
        print(f"(Video {i} of {len(video_ids)}) Video Title: {display_title}")

        safe_title = re.sub(r'[\\/*?:"><>|]', "_", video_title).replace("\n", " ")
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
                    pass

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
                vprint(f"Extracting audio for {transcript_arg}...")
                local_audio_path = extract_audio(video_id, local_audio_dir)
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
            vprint(f"--- Processing Language: {language} ---")
            col_suffix = f" ({language})" if language != "en" else ""
            lang_str = f" ({language})" if language != "en" else ""

            col_youtube = f"Transcript File youtube generated{col_suffix}"
            col_human = f"Transcript File human generated{col_suffix}"

            # --- YouTube Transcript Fetching ---
            youtube_transcript = ""
            is_generated = False

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

            # If no existing transcript, fetch from YouTube
            if not youtube_transcript:
                result = fetch_transcript(video_id, language=language)
                if result:
                    youtube_transcript, is_generated = result
                    prefix = (
                        f"youtube generated{lang_str} - "
                        if is_generated
                        else f"human generated{lang_str} - "
                    )
                    filename = f"{prefix}{video_id} - {safe_title}.txt"
                    # Relative path for storage
                    target_path = os.path.join(transcripts_dir, filename)

                    try:
                        saved_path = storage.write_text(target_path, youtube_transcript)
                        vprint(
                            f"Saved YouTube transcript ({language}): "
                            f"{format_clickable_path(saved_path)}"
                        )

                        # Update row with YouTube transcript info
                        if is_generated:
                            row[col_youtube] = saved_path
                        else:
                            row[col_human] = saved_path
                    except Exception as e:
                        print(f"Error writing YouTube transcript: {e}")

            # Update character counts
            if youtube_transcript:
                row[f"Transcript characters from youtube{col_suffix}"] = len(
                    youtube_transcript
                )
            elif language != "en":
                # Fallback to English transcript if available
                en_path = row.get("Transcript File human generated") or row.get(
                    "Transcript File youtube generated"
                )
                if en_path and storage.exists(str(en_path)):
                    youtube_transcript = storage.read_text(str(en_path))
                    vprint(
                        "Using existing English transcript as fallback for "
                        f"{language} processing."
                    )
                else:
                    # Try fetching English fresh
                    en_result = fetch_transcript(video_id, language="en")
                    if en_result:
                        youtube_transcript, en_is_generated = en_result
                        vprint(
                            f"Fetched English transcript as fallback for {language} "
                            "processing."
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
                                    target_path, youtube_transcript
                                )
                                vprint(
                                    f"Saved fallback English transcript: "
                                    f"{format_clickable_path(saved_path)}"
                                )
                                if en_is_generated:
                                    row["Transcript File youtube generated"] = (
                                        saved_path
                                    )
                                else:
                                    row["Transcript File human generated"] = saved_path
                            except Exception as e:
                                print(f"Error writing fallback English transcript: {e}")

                if youtube_transcript:
                    row[f"Transcript characters from youtube{col_suffix}"] = len(
                        youtube_transcript
                    )

            # --- AI Transcript Generation (if requested) ---
            ai_transcript = ""
            stt_cost = float("nan")
            transcript = youtube_transcript  # Default to YouTube transcript

            if transcript_arg != "youtube":
                ai_col = f"Transcript File {transcript_arg} generated{col_suffix}"
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

                # If no existing AI transcript, generate it
                if not ai_transcript:
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
                        print(f"Error: Audio file not found for STT: {audio_file_path}")
                    else:
                        vprint(
                            f"Generating transcript using model: {transcript_arg} "
                            f"({language})..."
                        )
                        ai_transcript, stt_in, stt_out = generate_transcript(
                            transcript_arg, audio_input_path, url, language=language
                        )

                        # Save AI transcript
                        prefix = f"{transcript_arg} generated{lang_str} - "
                        filename = f"{prefix}{video_id} - {safe_title}.txt"
                        target_path = os.path.join(transcripts_dir, filename)

                        try:
                            saved_path = storage.write_text(target_path, ai_transcript)
                            vprint(
                                "Saved AI transcript: "
                                f"{format_clickable_path(saved_path)}"
                            )
                            row[ai_col] = saved_path
                        except Exception as e:
                            print(f"Error writing AI transcript: {e}")

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

                row[f"Transcript characters from {transcript_arg}{col_suffix}"] = len(
                    ai_transcript
                )

            if not transcript:
                vprint(
                    f"No transcript (YouTube or AI) available for {video_id} "
                    f"({language}). Skipping further processing for this language."
                )
                continue

            # Summarize for each requested model
            for model_name in model_names:
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

                # Speaker Extraction
                speakers_text = ""
                speakers_input = 0
                speakers_output = 0
                speaker_cost = float("nan")

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
                elif row.get(speakers_col_name):
                    speakers_text = row[speakers_col_name]

                if not speakers_text:
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
                    vprint(f"Extracting speakers using model: {model_name}")
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
                    if speakers_text and not isinstance(row[speakers_col_name], float):
                        speakers_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"speakers (from {transcript_arg}).txt"
                        )
                        target_path = os.path.join(speakers_dir, speakers_filename)
                        try:
                            saved_path = storage.write_text(target_path, speakers_text)
                            vprint(
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

                if not row.get(qa_col_name):
                    vprint(f"Generating Q&A using model: {model_name} ({language})")
                    qa_text, qa_input, qa_output = generate_qa(
                        model_name, transcript, speakers_text, language=language
                    )
                    row[qa_col_name] = qa_text

                    if qa_text.strip() == "nan" or qa_text.strip() == 'float("nan")':
                        row[qa_col_name] = float("nan")

                    # Save QA File
                    if qa_text and not isinstance(row[qa_col_name], float):
                        qa_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"qa (from {transcript_arg}){lang_str}.md"
                        )
                        target_path = os.path.join(qa_dir, qa_filename)
                        try:
                            saved_path = storage.write_text(target_path, qa_text)
                            vprint(f"Saved Q&A: {format_clickable_path(saved_path)}")
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
                if summary_col_name in row and row[summary_col_name]:
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

                if not row.get(summary_col_name):
                    vprint(f"Summarizing using model: {model_name} ({language})")
                    summary_text, input_tokens, output_tokens = generate_summary(
                        model_name, transcript, video_title, url, language=language
                    )

                    summary_cost = float("nan")
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
                    if summaries_dir and summary_text:
                        summary_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"summary (from {transcript_arg}){lang_str}.md"
                        )
                        target_path = os.path.join(summaries_dir, summary_filename)
                        try:
                            summary_full_path = storage.write_text(
                                target_path, summary_text
                            )
                            vprint(
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

                if row.get(summary_col_name) and not row.get(one_sentence_col_name):
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

                    elif row.get(yt_speakers_col_name):
                        yt_speakers_text = row[yt_speakers_col_name]

                    # Generate if missing (checking specifically if it is 'float("nan")'
                    # default or actual text)
                    if yt_speakers_text == 'float("nan")' and (
                        not row.get(yt_speakers_col_name)
                    ):
                        # Try to use English transcript for YT speaker extraction
                        # if available
                        yt_speaker_source_transcript = youtube_transcript
                        if language != "en":
                            en_path = row.get(
                                "Transcript File human generated"
                            ) or row.get("Transcript File youtube generated")
                            if en_path and os.path.exists(str(en_path)):
                                with open(str(en_path), "r", encoding="utf-8") as f:
                                    yt_speaker_source_transcript = f.read()
                                    vprint(
                                        "Using English transcript for YouTube speaker "
                                        f"extraction ({model_name})."
                                    )
                        vprint(
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
                        if yt_speakers_text and not isinstance(
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
                                vprint(
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
                    if not row.get(yt_qa_col_name):
                        vprint(
                            f"Generating Q&A using model: {model_name} "
                            "(Source: YouTube Transcript)"
                        )
                        yt_qa_text, yt_qa_in, yt_qa_out = generate_qa(
                            model_name,
                            youtube_transcript,
                            yt_speakers_text,
                            language=language,
                        )

                        row[yt_qa_col_name] = yt_qa_text
                        if (
                            yt_qa_text.strip() == "nan"
                            or yt_qa_text.strip() == 'float("nan")'
                        ):
                            row[yt_qa_col_name] = float("nan")

                        yt_qa_cost = float("nan")
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
                        if row[yt_qa_col_name] and not isinstance(
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
                                vprint(
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

                    if not row.get(yt_sum_col_name):
                        vprint(
                            f"Summarizing using model: {model_name} "
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

                        yt_summary_cost = float("nan")
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
                        if summaries_dir and yt_summary_text:
                            summary_filename = (
                                f"{model_name} - {video_id} - {safe_title} - "
                                f"summary (from youtube){lang_str}.md"
                            )
                            target_path = os.path.join(summaries_dir, summary_filename)

                            try:
                                yt_summary_full_path = storage.write_text(
                                    target_path, yt_summary_text
                                )
                                vprint(
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

                    if row.get(yt_sum_col_name) and not row.get(
                        yt_one_sentence_col_name
                    ):
                        vprint(
                            "Generating one sentence summary using "
                            f"model: {model_name} "
                            "(Source: YouTube Transcript)"
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

            # Infographic Generation
            if infographic_arg:
                summary_targets = []

                # Target ALL summaries in the row (both existing and newly created)
                # This includes normal summaries and "from youtube" summaries
                for k in list(row.keys()):
                    if k.startswith("Summary Text ") and row[k]:
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

                for sum_col, m_name, s_text in summary_targets:
                    if not s_text:
                        continue

                    info_col = f"Summary Infographic File {m_name} {infographic_arg}"

                    # Check if already exists in row
                    if info_col in row and row[info_col]:
                        continue

                    # Check disk
                    safe_title = re.sub(r"[\\/*?:\"<>|]", "_", video_title).replace(
                        "\n", " "
                    )
                    safe_title = safe_title.replace("\r", "")

                    # Infographic filename should include language if m_name
                    # includes it. m_name already includes (es) if
                    # applicable because we sliced it from k.
                    # So the filename will automatically include (es).
                    infographic_filename = (
                        f"{m_name} - {infographic_arg} - {video_id} - "
                        f"{safe_title} - infographic.png"
                    )
                    expected_path = os.path.join(infographics_dir, infographic_filename)
                    if storage.exists(expected_path):
                        row[info_col] = expected_path
                        continue

                    summary_file_path = row.get(f"Summary File {m_name}", "")
                    summary_filename = (
                        os.path.basename(summary_file_path)
                        if summary_file_path
                        else "unknown file"
                    )
                    vprint(
                        f"Generating infographic using model {infographic_arg} "
                        f"from {summary_filename}"
                    )
                    image_bytes, input_tokens, output_tokens = generate_infographic(
                        infographic_arg, s_text, video_title, language=language
                    )
                    if image_bytes:
                        try:
                            saved_path = storage.write_bytes(expected_path, image_bytes)
                            vprint(
                                "Saved infographic: "
                                f"{format_clickable_path(saved_path)}"
                            )
                            row[info_col] = saved_path

                            # Calculate Infographic Cost
                            if verbose:
                                input_price, output_price = get_model_pricing(
                                    infographic_arg
                                )
                                if input_price is not None and output_price is not None:
                                    cost = (input_tokens / 1_000_000) * input_price + (
                                        output_tokens / 1_000_000
                                    ) * output_price
                                    cost = round(cost, 2)
                                    cost_col = (
                                        f"Summary Infographic Cost {m_name} "
                                        f"{infographic_arg} ($)"
                                    )
                                    row[cost_col] = cost
                                    vprint(f"Infographic cost: ${cost:.2f}")

                        except Exception as e:
                            print(f"Error writing infographic: {e}")

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
            vprint("Checking for TTS generation...")
            # TTS will scan all columns, so it should pick up the new
            # language columns too
            final_df = process_tts(
                final_df, tts_arg, storage, base_dir, languages=languages
            )
            should_save = True

        if combine_info_audio:
            vprint("Checking for Video generation...")
            final_df = process_videos(final_df, storage, base_dir)
            should_save = True

        if should_save:
            final_df = reorder_columns(final_df)
            saved_path = storage.save_dataframe(final_df, outfile_path)
            rprint(
                f"Successfully wrote {len(final_df)} rows to "
                f"{format_clickable_path(saved_path)}"
            )
        else:
            vprint("No new data to gather or all videos already processed.")
    else:
        vprint("No new data to gather or all videos already processed.")

    # Cleanup local temp dir
    if os.path.exists(local_temp_dir):
        import shutil

        shutil.rmtree(local_temp_dir)


if __name__ == "__main__":
    main()
