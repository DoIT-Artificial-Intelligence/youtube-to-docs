# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "google-auth>=2.45.0",
#     "google-genai>=1.56.0",
#     "google-api-python-client>=2.187.0",
#     "isodate>=0.7.2",
#     "openai>=1.56.0",
#     "polars>=1.36.1",
#     "requests>=2.32.5",
#     "youtube-transcript-api>=1.2.3",
#     "yt-dlp>=2025.2.19",
#     "static-ffmpeg>=2.5"
# ///
import argparse
import os
import re
import time

import polars as pl

from youtube_to_docs.infographic import generate_infographic
from youtube_to_docs.llms import (
    extract_speakers,
    generate_qa,
    generate_summary,
    generate_transcript,
    get_model_pricing,
    normalize_model_name,
)
from youtube_to_docs.transcript import (
    extract_audio,
    fetch_transcript,
    get_video_details,
    get_youtube_service,
    resolve_video_ids,
)
from youtube_to_docs.tts import process_tts


def reorder_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Reorder columns according to the specified logical structure."""
    cols = df.columns
    base_order = [
        "URL",
        "Title",
        "Description",
        "Data Published",
        "Channel",
        "Tags",
        "Duration",
        "Transcript characters from youtube",
        "Audio File",
    ]

    # Filter base_order to only include columns that actually exist
    final_order = [c for c in base_order if c in cols]

    # Add other Transcript characters columns
    other_transcript_chars = [
        c
        for c in cols
        if c.startswith("Transcript characters from ") and c not in final_order
    ]
    final_order.extend(sorted(other_transcript_chars))

    # Add Transcript File columns
    transcript_files = [c for c in cols if c.startswith("Transcript File ")]
    final_order.extend(sorted(transcript_files))

    # Add STT Cost columns
    stt_costs = [c for c in cols if " STT cost" in c]
    final_order.extend(sorted(stt_costs))

    # Add Summary File columns
    summary_files = [
        c for c in cols if c.startswith("Summary File ") and "from youtube" not in c
    ]
    final_order.extend(sorted(summary_files))

    # Add Summary Infographic File columns
    infographic_files = [c for c in cols if c.startswith("Summary Infographic File ")]
    final_order.extend(sorted(infographic_files))

    # Add Summary Infographic Cost columns
    infographic_costs = [c for c in cols if c.startswith("Summary Infographic Cost ")]
    final_order.extend(sorted(infographic_costs))

    # Add Audio File columns (from TTS)
    audio_files = [c for c in cols if c.startswith("Summary Audio File ")]
    final_order.extend(sorted(audio_files))

    # Add Video File columns
    video_files = [c for c in cols if c == "Video File"]
    final_order.extend(video_files)

    # Add QA File columns
    qa_files = [c for c in cols if c.startswith("QA File ")]
    final_order.extend(sorted(qa_files))

    # Add Speakers columns
    speakers = [
        c
        for c in cols
        if c.startswith("Speakers ") and not c.startswith("Speakers File ")
    ]
    final_order.extend(sorted(speakers))

    # Add Speakers File columns
    speakers_files = [c for c in cols if c.startswith("Speakers File ")]
    final_order.extend(sorted(speakers_files))

    # Add Speaker Extraction Cost columns
    speaker_costs = [c for c in cols if " Speaker extraction cost " in c]
    final_order.extend(sorted(speaker_costs))

    # Add Summary Text columns
    summary_texts = [
        c for c in cols if c.startswith("Summary Text ") and "from youtube" not in c
    ]
    final_order.extend(sorted(summary_texts))

    # Add Summary Cost columns
    summary_costs = [
        c for c in cols if " summary cost " in c and "from youtube" not in c
    ]
    final_order.extend(sorted(summary_costs))

    # Add YouTube Summary Text columns (secondary)
    yt_summary_texts = [c for c in cols if "Summary Text" in c and "from youtube" in c]
    final_order.extend(sorted(yt_summary_texts))

    # Add YouTube Summary File columns (secondary)
    yt_summary_files = [c for c in cols if "Summary File" in c and "from youtube" in c]
    final_order.extend(sorted(yt_summary_files))

    # Add YouTube Summary Cost columns (secondary)
    yt_summary_costs = [c for c in cols if "summary cost" in c and "from youtube" in c]
    final_order.extend(sorted(yt_summary_costs))

    # Add QA Text columns
    qa_texts = [c for c in cols if c.startswith("QA Text ")]
    final_order.extend(sorted(qa_texts))

    # Add QA Cost columns
    qa_costs = [c for c in cols if " QA cost " in c]
    final_order.extend(sorted(qa_costs))

    # Add any remaining columns that weren't caught
    remaining = [c for c in cols if c not in final_order]
    final_order.extend(remaining)

    return df.select(final_order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video_id",
        nargs="?",
        default="atmGAHYpf_c",
        help=(
            "Can be one of: \n"
            "A Video ID e.g. 'atmGAHYpf_c'\n"
            "Playlist ID (starts with PL e.g. 'PL8ZxoInteClyHaiReuOHpv6Z4SPrXtYtW')\n"
            "Channel Handle (starts with @ e.g. '@mga-hgo1740')\n"
            "Comma-separated list of Video IDs. (e.g. 'KuPc06JgI_A,GalhDyf3F8g')"
        ),
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="youtube-to-docs-artifacts/youtube-docs.csv",
        help=("Can be one of: \nLocal file path to save the output CSV file."),
    )
    parser.add_argument(
        "-t",
        "--transcript",
        default="youtube",
        help=(
            "The transcript source to use. \n"
            "Can be 'youtube' (default) to fetch existing YouTube transcripts, \n"
            "or an AI model name (e.g. 'gemini-3-flash-preview') to perform STT on "
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
            "Gemini model (e.g., 'gemini-3-flash-preview')\n"
            "GCP Vertex model (prefixed with 'vertex-'; e.g. "
            "'vertex-claude-haiku-4-5@20251001')\n"
            "AWS Bedrock model (prefixed with 'bedrock-'; e.g. "
            "'bedrock-claude-haiku-4-5-20251001-v1\n"
            "'bedrock-nova-2-lite-v1')\n"
            "Azure Foundry model (prefix with 'foundry-'; e.g. "
            "'foundry-gpt-5-mini)'\n"
            "Can also be a comma-separated list of models (e.g. "
            "'gemini-3-flash-preview,bedrock-claude-haiku-4-5-20251001-v1').\n"
            "Defaults to None."
        ),
    )
    parser.add_argument(
        "--tts",
        default=None,
        help=(
            "The TTS model and voice to use. "
            "Format: {model}-{voice} e.g. 'gemini-2.5-flash-preview-tts-Kore' \n"
            "or 'gemini-2.5-pro-preview-tts'"
        ),
    )
    parser.add_argument(
        "-i",
        "--infographic",
        default=None,
        help=(
            "The image model to use for generating an infographic. "
            "e.g. 'gemini-2.5-flash-image' or 'gemini-3-pro-image-preview'"
        ),
    )
    parser.add_argument(
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
        help=("The target language (e.g. 'es', 'fr', 'en'). Default is 'en'."),
    )

    args = parser.parse_args()
    transcript_arg = args.transcript
    video_id_input = args.video_id
    outfile = args.outfile
    model_names_arg = args.model
    tts_arg = args.tts
    infographic_arg = args.infographic
    no_youtube_summary = args.no_youtube_summary
    language_arg = args.language

    model_names = model_names_arg.split(",") if model_names_arg else []
    languages = language_arg.split(",") if language_arg else ["en"]

    youtube_service = get_youtube_service()

    video_ids = resolve_video_ids(video_id_input, youtube_service)

    # Setup Output Directories
    output_dir = os.path.dirname(outfile)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    base_dir = output_dir if output_dir else "."
    transcripts_dir = os.path.join(base_dir, "transcript-files")
    summaries_dir = os.path.join(base_dir, "summary-files")
    infographics_dir = os.path.join(base_dir, "infographic-files")
    speakers_dir = os.path.join(base_dir, "speaker-extraction-files")
    qa_dir = os.path.join(base_dir, "qa-files")
    audio_dir = os.path.join(base_dir, "audio-files")
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(infographics_dir, exist_ok=True)
    os.makedirs(speakers_dir, exist_ok=True)
    os.makedirs(qa_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    # Load existing CSV if it exists
    existing_df = None
    if os.path.exists(outfile):
        try:
            existing_df = pl.read_csv(outfile)
            print(f"Loaded existing data from {outfile} ({len(existing_df)} rows)")
        except Exception as e:
            print(f"Warning: Could not read existing CSV {outfile}: {e}")

    print(f"Processing {len(video_ids)} videos.")
    print(f"Processing Videos: {video_ids}")
    print(f"Saving to: {outfile}")
    if model_names:
        print(f"Summarizing using models: {model_names}")
    print(f"Target Languages: {languages}")

    rows = []

    for video_id in video_ids:
        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"Processing Video ID: {video_id}")

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

        print(f"Video Title: {video_title}")

        safe_title = re.sub(r'[\\/*?:"><>|]', "_", video_title).replace("\n", " ")
        safe_title = safe_title.replace("\r", "")

        # Audio Extraction (needed if AI transcript is requested,
        # regardless of language)
        # Check disk first if missing in row
        if transcript_arg != "youtube":
            if not row.get("Audio File") or not str(row.get("Audio File")).endswith(
                ".m4a"
            ):
                expected_audio = os.path.abspath(
                    os.path.join(audio_dir, f"{video_id}.m4a")
                )
                if os.path.exists(expected_audio):
                    print(f"Found existing audio file: {expected_audio}")
                    row["Audio File"] = expected_audio

        audio_file_path = row.get("Audio File", "")
        if transcript_arg != "youtube" and (
            not audio_file_path or not audio_file_path.endswith(".m4a")
        ):
            print(f"Extracting audio for STT using {transcript_arg}...")
            audio_file_path = extract_audio(video_id, audio_dir)
            if audio_file_path:
                print(f"Audio saved to: {audio_file_path}")
                row["Audio File"] = audio_file_path

        # --- Language Dependent Logic ---
        for language in languages:
            print(f"--- Processing Language: {language} ---")
            col_suffix = f" ({language})" if language != "en" else ""
            lang_str = f" ({language})" if language != "en" else ""

            col_youtube = f"Transcript File youtube generated{col_suffix}"
            col_human = f"Transcript File human generated{col_suffix}"

            # --- YouTube Transcript Fetching ---
            youtube_transcript = ""
            youtube_transcript_full_path = ""
            is_generated = False

            # Check disk if not in row
            if not row.get(col_youtube) and not row.get(col_human):
                gen_path = os.path.abspath(
                    os.path.join(
                        transcripts_dir,
                        f"youtube generated{lang_str} - {video_id} - {safe_title}.txt",
                    )
                )
                human_path = os.path.abspath(
                    os.path.join(
                        transcripts_dir,
                        f"human generated{lang_str} - {video_id} - {safe_title}.txt",
                    )
                )
                if os.path.exists(human_path):
                    row[col_human] = human_path
                elif os.path.exists(gen_path):
                    row[col_youtube] = gen_path

            # Load from row (which might have just been updated from disk)
            if row.get(col_youtube):
                path = row[col_youtube]
                if path and os.path.exists(str(path)):
                    with open(str(path), "r", encoding="utf-8") as f:
                        youtube_transcript = f.read()
                    youtube_transcript_full_path = str(path)
                    is_generated = True
            elif row.get(col_human):
                path = row[col_human]
                if path and os.path.exists(str(path)):
                    with open(str(path), "r", encoding="utf-8") as f:
                        youtube_transcript = f.read()
                    youtube_transcript_full_path = str(path)
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
                    youtube_transcript_full_path = os.path.abspath(
                        os.path.join(transcripts_dir, filename)
                    )
                    try:
                        with open(
                            youtube_transcript_full_path, "w", encoding="utf-8"
                        ) as f:
                            f.write(youtube_transcript)
                        print(f"Saved YouTube transcript ({language}): {filename}")
                    except OSError as e:
                        print(f"Error writing YouTube transcript: {e}")

                    # Update row with YouTube transcript info
                    if is_generated:
                        row[col_youtube] = youtube_transcript_full_path
                    else:
                        row[col_human] = youtube_transcript_full_path

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
                if en_path and os.path.exists(str(en_path)):
                    with open(str(en_path), "r", encoding="utf-8") as f:
                        youtube_transcript = f.read()
                    print(
                        "Using existing English transcript as fallback for "
                        f"{language} processing."
                    )
                else:
                    # Try fetching English fresh
                    en_result = fetch_transcript(video_id, language="en")
                    if en_result:
                        youtube_transcript, en_is_generated = en_result
                        print(
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
                            en_full_path = os.path.abspath(
                                os.path.join(transcripts_dir, filename)
                            )
                            try:
                                with open(en_full_path, "w", encoding="utf-8") as f:
                                    f.write(youtube_transcript)
                                print(f"Saved fallback English transcript: {filename}")
                                if en_is_generated:
                                    row["Transcript File youtube generated"] = (
                                        en_full_path
                                    )
                                else:
                                    row["Transcript File human generated"] = (
                                        en_full_path
                                    )
                            except OSError as e:
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

                # Check disk if not in row
                if not row.get(ai_col):
                    expected_ai_path = os.path.abspath(
                        os.path.join(
                            transcripts_dir,
                            f"{transcript_arg} generated{lang_str} - "
                            f"{video_id} - {safe_title}.txt",
                        )
                    )
                    if os.path.exists(expected_ai_path):
                        row[ai_col] = expected_ai_path

                # Check row for AI transcript
                if row.get(ai_col):
                    path = row[ai_col]
                    if path and os.path.exists(str(path)):
                        with open(str(path), "r", encoding="utf-8") as f:
                            ai_transcript = f.read()

                # If no existing AI transcript, generate it
                if not ai_transcript:
                    if not audio_file_path or not os.path.exists(audio_file_path):
                        print(f"Error: Audio file not found for STT: {audio_file_path}")
                    else:
                        print(
                            f"Generating transcript using model: {transcript_arg} "
                            f"({language})..."
                        )
                        ai_transcript, stt_in, stt_out = generate_transcript(
                            transcript_arg, audio_file_path, url, language=language
                        )
                        # Save AI transcript
                        prefix = f"{transcript_arg} generated{lang_str} - "
                        filename = f"{prefix}{video_id} - {safe_title}.txt"
                        ai_transcript_full_path = os.path.abspath(
                            os.path.join(transcripts_dir, filename)
                        )
                        try:
                            with open(
                                ai_transcript_full_path, "w", encoding="utf-8"
                            ) as f:
                                f.write(ai_transcript)
                            print(f"Saved AI transcript: {filename}")
                            row[ai_col] = ai_transcript_full_path
                        except OSError as e:
                            print(f"Error writing AI transcript: {e}")

                        # Calculate STT Cost
                        input_price, output_price = get_model_pricing(transcript_arg)
                        if input_price is not None and output_price is not None:
                            stt_cost = (stt_in / 1_000_000) * input_price + (
                                stt_out / 1_000_000
                            ) * output_price
                            row[stt_cost_col] = round(stt_cost, 2)
                            print(f"STT cost: ${row[stt_cost_col]:.2f}")

                # If AI transcript exists (either found or generated),
                # use it for summaries
                if ai_transcript:
                    transcript = ai_transcript

                row[f"Transcript characters from {transcript_arg}{col_suffix}"] = len(
                    ai_transcript
                )

            if not transcript:
                print(
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
                    expected_path = os.path.abspath(
                        os.path.join(speakers_dir, speakers_filename)
                    )
                    if os.path.exists(expected_path):
                        row[speakers_file_col_name] = expected_path

                # Load speakers from file/row
                if row.get(speakers_file_col_name):
                    path = row[speakers_file_col_name]
                    if path and os.path.exists(str(path)):
                        with open(str(path), "r", encoding="utf-8") as f:
                            speakers_text = f.read()
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
                        if en_path and os.path.exists(str(en_path)):
                            with open(str(en_path), "r", encoding="utf-8") as f:
                                speaker_source_transcript = f.read()
                                print(
                                    "Using English transcript for speaker extraction "
                                    f"({model_name})."
                                )
                    print(f"Extracting speakers using model: {model_name}")
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
                        speakers_full_path = os.path.abspath(
                            os.path.join(speakers_dir, speakers_filename)
                        )
                        try:
                            with open(speakers_full_path, "w", encoding="utf-8") as f:
                                f.write(speakers_text)
                            print(f"Saved speakers: {speakers_filename}")
                            row[speakers_file_col_name] = speakers_full_path
                        except OSError as e:
                            print(f"Error writing speakers file: {e}")

                    # Calculate Speaker Cost immediately
                    input_price, output_price = get_model_pricing(model_name)
                    if input_price is not None and output_price is not None:
                        speaker_cost = (speakers_input / 1_000_000) * input_price + (
                            speakers_output / 1_000_000
                        ) * output_price
                        row[speaker_cost_col_name] = round(speaker_cost, 2)
                        print(f"Speaker extraction cost: ${speaker_cost:.2f}")

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
                    expected_path = os.path.abspath(os.path.join(qa_dir, qa_filename))
                    if os.path.exists(expected_path):
                        row[qa_file_col_name] = expected_path

                # Load QA from file/row
                if row.get(qa_file_col_name):
                    path = row[qa_file_col_name]
                    if path and os.path.exists(str(path)):
                        with open(str(path), "r", encoding="utf-8") as f:
                            row[qa_col_name] = f.read()

                if not row.get(qa_col_name):
                    print(f"Generating Q&A using model: {model_name} ({language})")
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
                        qa_full_path = os.path.abspath(
                            os.path.join(qa_dir, qa_filename)
                        )
                        try:
                            with open(qa_full_path, "w", encoding="utf-8") as f:
                                f.write(qa_text)
                            print(f"Saved Q&A: {qa_filename}")
                            row[qa_file_col_name] = qa_full_path
                        except OSError as e:
                            print(f"Error writing Q&A file: {e}")

                    # Calculate QA Cost
                    input_price, output_price = get_model_pricing(model_name)
                    if input_price is not None and output_price is not None:
                        # QA input tokens include the speaker text provided in
                        # the prompt
                        qa_cost = (qa_input / 1_000_000) * input_price + (
                            qa_output / 1_000_000
                        ) * output_price
                        qa_cost = round(qa_cost, 2)
                        row[qa_cost_col_name] = qa_cost
                        print(f"Q&A cost: ${qa_cost:.2f}")

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
                        print(f"Backfilling cost for model: {model_name}")
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
                                s_cost = (speakers_input / 1_000_000) * input_price + (
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
                            print(f"Estimated summary cost: ${summary_cost:.2f}")
                    elif speakers_input > 0 or speakers_output > 0:
                        # Cost exists, but we generated speakers. Add that cost.
                        input_price, output_price = get_model_pricing(model_name)
                        if input_price is not None and output_price is not None:
                            s_cost = (speakers_input / 1_000_000) * input_price + (
                                speakers_output / 1_000_000
                            ) * output_price
                            current_cost = row[summary_cost_col_name]
                            row[summary_cost_col_name] = round(current_cost + s_cost, 2)
                            print(
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
                    expected_path = os.path.abspath(
                        os.path.join(summaries_dir, summary_filename)
                    )
                    if os.path.exists(expected_path):
                        row[summary_file_col_name] = expected_path

                # Load Summary from file/row
                if row.get(summary_file_col_name):
                    path = row[summary_file_col_name]
                    if path and os.path.exists(str(path)):
                        with open(str(path), "r", encoding="utf-8") as f:
                            row[summary_col_name] = f.read()

                if not row.get(summary_col_name):
                    print(f"Summarizing using model: {model_name} ({language})")
                    summary_text, input_tokens, output_tokens = generate_summary(
                        model_name, transcript, video_title, url, language=language
                    )

                    summary_cost = float("nan")
                    input_price, output_price = get_model_pricing(model_name)
                    if input_price is not None and output_price is not None:
                        # Add speaker tokens
                        total_input = input_tokens + speakers_input
                        total_output = output_tokens + speakers_output

                        summary_cost = (total_input / 1_000_000) * input_price + (
                            total_output / 1_000_000
                        ) * output_price
                        summary_cost = round(summary_cost, 2)
                        print(f"Summary cost: ${summary_cost:.2f}")

                    summary_full_path = ""
                    if summaries_dir and summary_text:
                        summary_filename = (
                            f"{model_name} - {video_id} - {safe_title} - "
                            f"summary (from {transcript_arg}){lang_str}.md"
                        )
                        summary_full_path = os.path.abspath(
                            os.path.join(summaries_dir, summary_filename)
                        )
                        try:
                            with open(summary_full_path, "w", encoding="utf-8") as f:
                                f.write(summary_text)
                            print(f"Saved summary: {summary_filename}")
                        except OSError as e:
                            print(f"Error writing summary: {e}")

                    row[summary_file_col_name] = summary_full_path
                    row[summary_col_name] = summary_text
                    row[summary_cost_col_name] = summary_cost

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
                        expected_path = os.path.abspath(
                            os.path.join(speakers_dir, yt_speakers_filename)
                        )
                        if os.path.exists(expected_path):
                            row[yt_speakers_file_col_name] = expected_path

                    # Load YT speakers from file/row
                    if row.get(yt_speakers_file_col_name):
                        path = row[yt_speakers_file_col_name]
                        if path and os.path.exists(str(path)):
                            with open(str(path), "r", encoding="utf-8") as f:
                                yt_speakers_text = f.read()
                            row[yt_speakers_col_name] = yt_speakers_text
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
                                    print(
                                        "Using English transcript for YouTube speaker "
                                        f"extraction ({model_name})."
                                    )
                        print(
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
                            yt_speakers_full_path = os.path.abspath(
                                os.path.join(speakers_dir, yt_speakers_filename)
                            )
                            try:
                                with open(
                                    yt_speakers_full_path, "w", encoding="utf-8"
                                ) as f:
                                    f.write(yt_speakers_text)
                                print(f"Saved YouTube speakers: {yt_speakers_filename}")
                                row[yt_speakers_file_col_name] = yt_speakers_full_path
                            except OSError as e:
                                print(f"Error writing YouTube speakers file: {e}")

                        # Calculate YouTube Speaker Cost
                        input_price, output_price = get_model_pricing(model_name)
                        if input_price is not None and output_price is not None:
                            yt_speaker_cost = (
                                yt_speakers_input / 1_000_000
                            ) * input_price + (
                                yt_speakers_output / 1_000_000
                            ) * output_price
                            yt_speaker_cost = round(yt_speaker_cost, 2)
                            row[yt_speaker_cost_col_name] = yt_speaker_cost
                            print(
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
                        expected_path = os.path.abspath(
                            os.path.join(qa_dir, qa_filename)
                        )
                        if os.path.exists(expected_path):
                            row[yt_qa_file_col_name] = expected_path

                    # Load YT QA from file/row
                    if row.get(yt_qa_file_col_name):
                        path = row[yt_qa_file_col_name]
                        if path and os.path.exists(str(path)):
                            with open(str(path), "r", encoding="utf-8") as f:
                                row[yt_qa_col_name] = f.read()

                    if not row.get(yt_qa_col_name):
                        print(
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
                        input_price, output_price = get_model_pricing(model_name)
                        if input_price is not None and output_price is not None:
                            # Pure QA cost
                            cost = (yt_qa_in / 1_000_000) * input_price + (
                                yt_qa_out / 1_000_000
                            ) * output_price
                            yt_qa_cost = round(cost, 2)
                            print(f"YouTube Q&A cost: ${yt_qa_cost:.2f}")

                        yt_qa_full_path = ""
                        if row[yt_qa_col_name] and not isinstance(
                            row[yt_qa_col_name], float
                        ):
                            qa_filename = (
                                f"{model_name} - {video_id} - {safe_title} - "
                                f"qa (from youtube){lang_str}.md"
                            )
                            yt_qa_full_path = os.path.abspath(
                                os.path.join(qa_dir, qa_filename)
                            )
                            try:
                                with open(yt_qa_full_path, "w", encoding="utf-8") as f:
                                    f.write(row[yt_qa_col_name])
                                print(f"Saved YouTube Q&A: {qa_filename}")
                            except OSError as e:
                                print(f"Error writing YouTube Q&A: {e}")

                        row[yt_qa_file_col_name] = yt_qa_full_path
                        row[yt_qa_cost_col_name] = yt_qa_cost

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
                        expected_path = os.path.abspath(
                            os.path.join(summaries_dir, summary_filename)
                        )
                        if os.path.exists(expected_path):
                            row[yt_sum_file_col_name] = expected_path

                    # Load YT Summary from file/row
                    if row.get(yt_sum_file_col_name):
                        path = row[yt_sum_file_col_name]
                        if path and os.path.exists(str(path)):
                            with open(str(path), "r", encoding="utf-8") as f:
                                row[yt_sum_col_name] = f.read()

                    if not row.get(yt_sum_col_name):
                        print(
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
                        input_price, output_price = get_model_pricing(model_name)
                        if input_price is not None and output_price is not None:
                            # We don't include speaker tokens here as we didn't extract
                            # speakers from the YouTube transcript specifically for this
                            # summary.
                            # If we wanted to be precise, we'd need to extract speakers
                            # from YT transcript too.
                            # For now, just the summary cost.
                            cost = (yt_input_tokens / 1_000_000) * input_price + (
                                yt_output_tokens / 1_000_000
                            ) * output_price
                            yt_summary_cost = round(cost, 2)
                            print(f"YouTube Summary cost: ${yt_summary_cost:.2f}")

                        yt_summary_full_path = ""
                        if summaries_dir and yt_summary_text:
                            summary_filename = (
                                f"{model_name} - {video_id} - {safe_title} - "
                                f"summary (from youtube){lang_str}.md"
                            )
                            yt_summary_full_path = os.path.abspath(
                                os.path.join(summaries_dir, summary_filename)
                            )
                            try:
                                with open(
                                    yt_summary_full_path, "w", encoding="utf-8"
                                ) as f:
                                    f.write(yt_summary_text)
                                print(f"Saved YouTube summary: {summary_filename}")
                            except OSError as e:
                                print(f"Error writing YouTube summary: {e}")

                        row[yt_sum_file_col_name] = yt_summary_full_path
                        row[yt_sum_col_name] = yt_summary_text
                        row[yt_sum_cost_col_name] = yt_summary_cost

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
                    expected_path = os.path.abspath(
                        os.path.join(infographics_dir, infographic_filename)
                    )
                    if os.path.exists(expected_path):
                        row[info_col] = expected_path
                        continue

                    summary_file_path = row.get(f"Summary File {m_name}", "")
                    summary_filename = (
                        os.path.basename(summary_file_path)
                        if summary_file_path
                        else "unknown file"
                    )
                    print(
                        f"Generating infographic using model {infographic_arg} "
                        f"from {summary_filename}"
                    )
                    image_bytes, input_tokens, output_tokens = generate_infographic(
                        infographic_arg, s_text, video_title, language=language
                    )
                    if image_bytes:
                        infographic_full_path = expected_path
                        try:
                            with open(infographic_full_path, "wb") as f:
                                f.write(image_bytes)
                            print(f"Saved infographic: {infographic_filename}")
                            row[info_col] = infographic_full_path

                            # Calculate Infographic Cost
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
                                print(f"Infographic cost: ${cost:.2f}")

                        except OSError as e:
                            print(f"Error writing infographic: {e}")

        rows.append(row)
        time.sleep(1)

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

        if tts_arg:
            print("Checking for TTS generation...")
            # TTS will scan all columns, so it should pick up the new
            # language columns too
            final_df = process_tts(final_df, tts_arg, base_dir, languages=languages)
            should_save = True

        if should_save:
            final_df = reorder_columns(final_df)
            final_df.write_csv(outfile)
            print(f"Successfully wrote {len(final_df)} rows to {outfile}")
        else:
            print("No new data to gather or all videos already processed.")
    else:
        print("No new data to gather or all videos already processed.")


if __name__ == "__main__":
    main()
