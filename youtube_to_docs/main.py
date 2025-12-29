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
#     "youtube-transcript-api>=1.2.3"
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
    get_model_pricing,
    normalize_model_name,
)
from youtube_to_docs.transcript import (
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
        "Transcript characters",
    ]

    # Filter base_order to only include columns that actually exist
    final_order = [c for c in base_order if c in cols]

    # Add Transcript File columns
    transcript_files = [c for c in cols if c.startswith("Transcript File ")]
    final_order.extend(sorted(transcript_files))

    # Add Summary File columns
    summary_files = [c for c in cols if c.startswith("Summary File ")]
    final_order.extend(sorted(summary_files))

    # Add Summary Infographic File columns
    infographic_files = [c for c in cols if c.startswith("Summary Infographic File ")]
    final_order.extend(sorted(infographic_files))

    # Add Audio File columns (from TTS)
    audio_files = [c for c in cols if c.startswith("Summary Audio File ")]
    final_order.extend(sorted(audio_files))

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
    speaker_costs = [c for c in cols if c.endswith(" Speaker extraction cost ($)")]
    final_order.extend(sorted(speaker_costs))

    # Add Summary Text columns
    summary_texts = [c for c in cols if c.startswith("Summary Text ")]
    final_order.extend(sorted(summary_texts))

    # Add Summary Cost columns
    summary_costs = [c for c in cols if c.endswith(" summary cost ($)")]
    final_order.extend(sorted(summary_costs))

    # Add QA Text columns
    qa_texts = [c for c in cols if c.startswith("QA Text ")]
    final_order.extend(sorted(qa_texts))

    # Add QA Cost columns
    qa_costs = [c for c in cols if c.endswith(" QA cost ($)")]
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
        default="youtube-docs.csv",
        help=("Can be one of: \nLocal file path to save the output CSV file."),
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
            "'bedrock-claude-haiku-4-5-20251001-v1'\n"
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
            "Format: {model}-{voice} e.g. 'gemini-2.5-flash-preview-tts-Kore'"
        ),
    )
    parser.add_argument(
        "--infographic",
        default=None,
        help=(
            "The image model to use for generating an infographic. "
            "e.g. 'gemini-2.5-flash-image'"
        ),
    )

    args = parser.parse_args()
    video_id_input = args.video_id
    outfile = args.outfile
    model_names_arg = args.model
    tts_arg = args.tts
    infographic_arg = args.infographic

    model_names = model_names_arg.split(",") if model_names_arg else []

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
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(infographics_dir, exist_ok=True)
    os.makedirs(speakers_dir, exist_ok=True)
    os.makedirs(qa_dir, exist_ok=True)

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

        # Determine if we need to process this video
        needs_details = existing_row is None
        needs_transcript = existing_row is None

        # Check needs for each model
        models_needing_summary = []
        for model_name in model_names:
            s_col = f"Summary Text {model_name}"
            if (
                existing_row is None
                or s_col not in existing_row
                or not existing_row[s_col]
            ):
                models_needing_summary.append(model_name)

        # Check needs for infographic (any summary text column without
        # corresponding infographic)
        # This is a bit complex to pre-calculate perfectly without the row
        # constructed, but we can rely on the loop later.
        # Generally if we are adding new summaries, we likely need
        # infographics for them.

        missing_costs = False
        for model_name in model_names:
            summary_cost_col_name = (
                f"{normalize_model_name(model_name)} summary cost ($)"
            )
            if existing_row:
                if (
                    summary_cost_col_name not in existing_row
                    or existing_row[summary_cost_col_name] is None
                    or (
                        isinstance(existing_row[summary_cost_col_name], float)
                        and existing_row[summary_cost_col_name]
                        != existing_row[summary_cost_col_name]
                    )
                ):
                    missing_costs = True
                    break

        if (
            not needs_details
            and not needs_transcript
            and not models_needing_summary
            and not infographic_arg  # No infographic arg, don't care about missing
            and not missing_costs
        ):
            # If infographic arg IS present, we should check if we are missing
            # any infographics for existing summaries.
            missing_infographics = False
            if infographic_arg and existing_row:
                for k in existing_row.keys():
                    if k.startswith("Summary Text ") and existing_row[k]:
                        m_name = k[len("Summary Text ") :]
                        info_col = (
                            f"Summary Infographic File {m_name} {infographic_arg}"
                        )
                        if info_col not in existing_row or not existing_row[info_col]:
                            missing_infographics = True
                            break

            if not missing_infographics and not missing_costs:
                print(
                    f"Skipping {video_id}: already exists in table with metadata, "
                    "summaries, and infographics."
                )
                continue

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
        else:
            assert existing_row is not None
            video_title = existing_row["Title"]
            description = existing_row["Description"]
            publishedAt = existing_row["Data Published"]
            channelTitle = existing_row["Channel"]
            tags = existing_row["Tags"]
            video_duration = existing_row["Duration"]

        print(f"Video Title: {video_title}")

        # Fetch/Save Transcript
        transcript = ""
        transcript_full_path = ""
        is_generated = False

        # Check existing row for transcript
        if existing_row:
            col_youtube = "Transcript File youtube generated"
            col_human = "Transcript File human generated"
            found_col = None

            if col_youtube in existing_row and existing_row[col_youtube]:
                found_col = col_youtube
                is_generated = True
            elif col_human in existing_row and existing_row[col_human]:
                found_col = col_human
                is_generated = False

            if found_col:
                transcript_full_path = existing_row[found_col]
                if os.path.exists(transcript_full_path):
                    print(
                        f"Reading existing transcript from file: {transcript_full_path}"
                    )
                    with open(transcript_full_path, "r", encoding="utf-8") as f:
                        transcript = f.read()

        if not transcript:
            result = fetch_transcript(video_id)
            if not result:
                continue
            transcript, is_generated = result

            # Save Transcript
            safe_title = re.sub(r'[\\/*?:"<>|]', "_", video_title).replace("\n", " ")
            safe_title = safe_title.replace("\r", "")
            prefix = "youtube generated - " if is_generated else "human generated - "
            transcript_filename = f"{prefix}{video_id} - {safe_title}.txt"
            transcript_full_path = os.path.abspath(
                os.path.join(transcripts_dir, transcript_filename)
            )
            try:
                with open(transcript_full_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
                print(f"Saved transcript: {transcript_filename}")
            except OSError as e:
                print(f"Error writing transcript: {e}")

        # Prepare row data base
        row = existing_row.copy() if existing_row else {}

        if is_generated:
            transcript_col_name = "Transcript File youtube generated"
        else:
            transcript_col_name = "Transcript File human generated"

        row.update(
            {
                "URL": url,
                "Title": video_title,
                "Description": description,
                "Data Published": publishedAt,
                "Channel": channelTitle,
                "Tags": tags,
                "Duration": video_duration,
                "Transcript characters": len(transcript),
                transcript_col_name: transcript_full_path,
            }
        )

        # Summarize for each requested model
        for model_name in model_names:
            summary_col_name = f"Summary Text {model_name}"
            summary_file_col_name = f"Summary File {model_name}"
            speakers_col_name = f"Speakers {model_name}"
            speakers_file_col_name = f"Speakers File {model_name}"
            summary_cost_col_name = (
                f"{normalize_model_name(model_name)} summary cost ($)"
            )
            speaker_cost_col_name = (
                f"{normalize_model_name(model_name)} Speaker extraction cost ($)"
            )

            # Speaker Extraction
            speakers_text = ""
            speakers_input = 0
            speakers_output = 0
            speaker_cost = float("nan")

            if speakers_col_name in row and row[speakers_col_name]:
                speakers_text = row[speakers_col_name]
                # If we have text but no cost, we can't easily recalculate exact tokens
                # without re-running counting, but we can estimate or leave as NaN.
                # If we are backfilling cost, we will handle it below.
            else:
                print(f"Extracting speakers using model: {model_name}")
                speakers_text, speakers_input, speakers_output = extract_speakers(
                    model_name, transcript
                )
                row[speakers_col_name] = speakers_text
                if (
                    speakers_text.strip() == "nan"
                    or speakers_text.strip() == 'float("nan")'
                ):
                    row[speakers_col_name] = float("nan")

                # Save Speakers File
                if speakers_text and not isinstance(row[speakers_col_name], float):
                    safe_title = re.sub(r'[\\/*?:"<>|]', "_", video_title).replace(
                        "\n", " "
                    )
                    safe_title = safe_title.replace("\r", "")
                    speakers_filename = (
                        f"{model_name} - {video_id} - {safe_title} - speakers.txt"
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
                    speaker_cost = round(speaker_cost, 2)
                    row[speaker_cost_col_name] = speaker_cost
                    print(f"Speaker extraction cost: ${speaker_cost:.2f}")

            # QA Generation
            qa_col_name = f"QA Text {model_name}"
            qa_file_col_name = f"QA File {model_name}"
            qa_cost_col_name = f"{normalize_model_name(model_name)} QA cost ($)"

            if qa_col_name not in row or not row[qa_col_name]:
                print(f"Generating Q&A using model: {model_name}")
                qa_text, qa_input, qa_output = generate_qa(
                    model_name, transcript, speakers_text
                )
                row[qa_col_name] = qa_text

                if qa_text.strip() == "nan" or qa_text.strip() == 'float("nan")':
                    row[qa_col_name] = float("nan")

                # Save QA File
                if qa_text and not isinstance(row[qa_col_name], float):
                    safe_title = re.sub(r'[\\/*?:"<>|]', "_", video_title).replace(
                        "\n", " "
                    )
                    safe_title = safe_title.replace("\r", "")
                    qa_filename = f"{model_name} - {video_id} - {safe_title} - qa.md"
                    qa_full_path = os.path.abspath(os.path.join(qa_dir, qa_filename))
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
                    # Add speaker tokens to cost as they are part of input
                    # speakers_input is for extraction, not for QA prompt input
                    # But we passed speakers_text to QA prompt, so we should count it.
                    # The generate_qa return values (qa_input) should include it if
                    # the API reports it correctly.

                    qa_cost = (qa_input / 1_000_000) * input_price + (
                        qa_output / 1_000_000
                    ) * output_price
                    qa_cost = round(qa_cost, 2)
                    row[qa_cost_col_name] = qa_cost
                    print(f"Q&A cost: ${qa_cost:.2f}")

            # Check if we already have it in the row (from existing_row)
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
                        summary_cost = (est_input_tokens / 1_000_000) * input_price + (
                            est_output_tokens / 1_000_000
                        ) * output_price

                        # Add speaker cost if we just generated them or can backfill it
                        # If we just generated it, speakers_input > 0
                        if speakers_input > 0 or speakers_output > 0:
                            s_cost = (speakers_input / 1_000_000) * input_price + (
                                speakers_output / 1_000_000
                            ) * output_price
                            summary_cost += s_cost
                        elif speaker_cost_col_name in row and not (
                            isinstance(row[speaker_cost_col_name], float)
                            and row[speaker_cost_col_name] != row[speaker_cost_col_name]
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

            print(f"Summarizing using model: {model_name}")
            summary_text, input_tokens, output_tokens = generate_summary(
                model_name, transcript, video_title, url
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

                # If we calculated speaker cost earlier, make sure it's in the row
                # (it was added above)

            summary_full_path = ""
            if summaries_dir and summary_text:
                safe_title = re.sub(r"[\\/*?:\"<>|]", "_", video_title).replace(
                    "\n", " "
                )
                safe_title = safe_title.replace("\r", "")
                summary_filename = (
                    f"{model_name} - {video_id} - {safe_title} - summary.md"
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

        # Infographic Generation
        if infographic_arg:
            summary_targets = []

            # Target ALL summaries in the row (both existing and newly created)
            for k in list(row.keys()):
                if k.startswith("Summary Text ") and row[k]:
                    m_name = k[len("Summary Text ") :]
                    summary_targets.append((k, m_name, row[k]))

            for sum_col, m_name, s_text in summary_targets:
                if not s_text:
                    continue

                info_col = f"Summary Infographic File {m_name} {infographic_arg}"

                # Check if already exists
                if info_col in row and row[info_col]:
                    continue

                print(f"Generating infographic for model: {m_name}")
                image_bytes = generate_infographic(infographic_arg, s_text, video_title)
                if image_bytes:
                    safe_title = re.sub(r"[\\/*?:\"<>|]", "_", video_title).replace(
                        "\n", " "
                    )
                    safe_title = safe_title.replace("\r", "")
                    infographic_filename = (
                        f"{m_name} - {infographic_arg} - {video_id} - "
                        f"{safe_title} - infographic.png"
                    )
                    infographic_full_path = os.path.abspath(
                        os.path.join(infographics_dir, infographic_filename)
                    )
                    try:
                        with open(infographic_full_path, "wb") as f:
                            f.write(image_bytes)
                        print(f"Saved infographic: {infographic_filename}")
                        row[info_col] = infographic_full_path
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
            final_df = process_tts(final_df, tts_arg, base_dir)
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
