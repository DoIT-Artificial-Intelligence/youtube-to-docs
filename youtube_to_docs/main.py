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
#
# Run as:
# uv run https://raw.githubusercontent.com/DoIT-Artifical-Intelligence/youtube-to-docs/refs/heads/main/youtube_to_docs/main.py --model gemini-3-flash-preview  # noqa
# To test locally run one of:
# uv run youtube-to-docs --model gemini-3-flash-preview
# uv run python -m youtube_to_docs.main --model gemini-3-flash-preview
# uv run youtube-to-docs --model vertex-claude-haiku-4-5@20251001
# uv run youtube-to-docs --model bedrock-claude-haiku-4-5-20251001-v1
# uv run youtube-to-docs --model bedrock-nova-2-lite-v1
# uv run youtube-to-docs --model bedrock-claude-haiku-4-5-20251001
# uv run youtube-to-docs --model foundry-gpt-5-mini


import argparse
import os
import re
import time
from typing import List, Optional

import polars as pl

from youtube_to_docs.llms import generate_summary
from youtube_to_docs.transcript import (
    fetch_transcript,
    get_video_details,
    get_youtube_service,
    resolve_video_ids,
)


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
            "The LLM to use for summarization. Can be one of: \n"
            "Gemini model (e.g., 'gemini-3-flash-preview')\n"
            "GCP Vertex model (prefixed with 'vertex-'). e.g. "
            "vertex-claude-haiku-4-5@20251001\n"
            "AWS Bedrock model (prefixed with 'bedrock-'). e.g. "
            "bedrock-claude-haiku-4-5-20251001-v1\n"
            "Azure Foundry model (prefix with 'foundry-). e.g. 'foundry-gpt-5-mini'\n"
            "Defaults to None."
        ),
    )

    args = parser.parse_args()
    video_id_input: str = args.video_id
    outfile: str = args.outfile
    model_name: Optional[str] = args.model

    youtube_service = get_youtube_service()

    video_ids = resolve_video_ids(video_id_input, youtube_service)

    # Setup Output Directories
    transcripts_dir: Optional[str] = None
    summaries_dir: Optional[str] = None
    output_dir = os.path.dirname(outfile)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    base_dir = output_dir if output_dir else "."
    transcripts_dir = os.path.join(base_dir, "transcript-files")
    summaries_dir = os.path.join(base_dir, "summary-files")
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    # Load existing CSV if it exists
    existing_df: Optional[pl.DataFrame] = None
    if os.path.exists(outfile):
        try:
            existing_df = pl.read_csv(outfile)
            print(f"Loaded existing data from {outfile} ({len(existing_df)} rows)")
        except Exception as e:
            print(f"Warning: Could not read existing CSV {outfile}: {e}")

    print(f"Processing {len(video_ids)} videos.")
    print(f"Processing Videos: {video_ids}")
    print(f"Saving to: {outfile}")
    if model_name:
        print(f"Summarizing using model: {model_name}")

    rows: List[dict] = []
    summary_col_name = f"Summary Text {model_name}" if model_name else "Summary Text"
    summary_file_col_name = (
        f"Summary File {model_name}" if model_name else "Summary File"
    )

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
        needs_summary = model_name is not None and (
            existing_row is None
            or summary_col_name not in existing_row
            or not existing_row[summary_col_name]
        )

        if not needs_details and not needs_transcript and not needs_summary:
            print(
                f"Skipping {video_id}: already exists in table with metadata "
                "and summary."
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
            video_title = existing_row["Title"]  # type: ignore
            description = existing_row["Description"]  # type: ignore
            publishedAt = existing_row["Data Published"]  # type: ignore
            channelTitle = existing_row["Channel"]  # type: ignore
            tags = existing_row["Tags"]  # type: ignore
            video_duration = existing_row["Duration"]  # type: ignore

        print(f"Video Title: {video_title}")

        # Fetch/Save Transcript
        transcript = ""
        transcript_full_path = ""
        if (
            existing_row
            and "Transcript File youtube generated" in existing_row
            and existing_row["Transcript File youtube generated"]
        ):
            transcript_full_path = existing_row["Transcript File youtube generated"]
            if os.path.exists(transcript_full_path):
                print(f"Reading existing transcript from file: {transcript_full_path}")
                with open(transcript_full_path, "r", encoding="utf-8") as f:
                    transcript = f.read()

        if not transcript:
            result = fetch_transcript(video_id)
            if not result:
                continue
            transcript, is_generated = result

            # Save Transcript
            safe_title = (
                re.sub(r'[\\/*?:"<>|]', "_", video_title)
                .replace("\n", " ")
                .replace("\r", "")
            )
            prefix = "youtube generated - " if is_generated else ""
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

        # Summarize
        summary_text = existing_row.get(summary_col_name, "") if existing_row else ""
        summary_full_path = (
            existing_row.get(summary_file_col_name, "") if existing_row else ""
        )

        if needs_summary:
            print(f"Summarizing using model: {model_name}")
            summary_text = generate_summary(model_name, transcript, video_title, url)

            if summaries_dir and summary_text:
                safe_title = (
                    re.sub(r'[\\/*?:"<>|]', "_", video_title)
                    .replace("\n", " ")
                    .replace("\r", "")
                )
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

        row = existing_row.copy() if existing_row else {}
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
                "Transcript File youtube generated": transcript_full_path,
                summary_file_col_name: summary_full_path,
                summary_col_name: summary_text,
            }
        )
        rows.append(row)
        time.sleep(1)

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

        if "Data Published" in final_df.columns:
            final_df = final_df.sort("Data Published", descending=True)

        final_df.write_csv(outfile)
        print(f"Successfully wrote {len(final_df)} rows to {outfile}")
    else:
        print("No new data to gather or all videos already processed.")


if __name__ == "__main__":
    main()
