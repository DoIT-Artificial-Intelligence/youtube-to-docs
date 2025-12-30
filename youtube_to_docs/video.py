import argparse
import os
import subprocess

import polars as pl
from static_ffmpeg import run

from youtube_to_docs.main import reorder_columns


def create_video(image_path: str, audio_path: str, output_path: str) -> bool:
    """Creates an MP4 video from an image and an audio file using ffmpeg."""
    # Use static_ffmpeg to ensure ffmpeg is available
    ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()

    command = [
        ffmpeg_path,
        "-y",  # Overwrite output file if it exists
        "-loop",
        "1",
        "-i",
        image_path,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        output_path,
    ]

    try:
        # Redirect stdout and stderr to devnull to keep output clean
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        return False


def process_videos(csv_path: str) -> None:
    """Processes the CSV to create videos from infographics and audio files."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    try:
        df = pl.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return

    # Setup Video Directory
    base_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else "."
    video_dir = os.path.join(base_dir, "video-files")
    os.makedirs(video_dir, exist_ok=True)

    # Identify relevant columns
    info_cols = [c for c in df.columns if c.startswith("Summary Infographic File ")]
    audio_cols = [c for c in df.columns if c.startswith("Summary Audio File ")]

    if not info_cols or not audio_cols:
        print("Required columns (infographic and audio) not found in CSV.")
        return

    video_files = []

    for row in df.iter_rows(named=True):
        infographics = [
            row[c]
            for c in info_cols
            if row[c] and isinstance(row[c], str) and os.path.exists(row[c])
        ]
        audios = [
            row[c]
            for c in audio_cols
            if row[c] and isinstance(row[c], str) and os.path.exists(row[c])
        ]

        if len(infographics) == 1 and len(audios) == 1:
            info_path = infographics[0]
            audio_path = audios[0]

            # Generate output filename based on audio filename
            audio_basename = os.path.basename(audio_path)
            video_filename = os.path.splitext(audio_basename)[0] + ".mp4"
            video_path = os.path.abspath(os.path.join(video_dir, video_filename))

            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                print(f"Video already exists: {video_filename}")
                video_files.append(video_path)
            else:
                print(f"Creating video: {video_filename}")
                if create_video(info_path, audio_path, video_path):
                    print(f"Successfully created: {video_filename}")
                    video_files.append(video_path)
                else:
                    video_files.append(None)
        else:
            if len(infographics) > 1 or len(audios) > 1:
                print(
                    f"Skipping row for {row.get('Title', 'Unknown')}: "
                    f"Multiple infographics ({len(infographics)}) or "
                    f"audios ({len(audios)}) found."
                )
            video_files.append(None)

    # Add back to the dataframe
    if "Video File" in df.columns:
        # Merge with existing Video File column if it exists
        df = df.with_columns(
            pl.when(pl.col("Video File").is_null())
            .then(pl.Series(video_files))
            .otherwise(pl.col("Video File"))
            .alias("Video File")
        )
    else:
        df = df.with_columns(pl.Series(name="Video File", values=video_files))

    # Save CSV
    df = reorder_columns(df)
    df.write_csv(csv_path)
    print(f"Updated {csv_path} with Video File column.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine infographics and audio into MP4 videos."
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="youtube-to-docs-artifacts/youtube-docs.csv",
        help="Path to the CSV file to process.",
    )

    args = parser.parse_args()
    process_videos(args.outfile)


if __name__ == "__main__":
    main()
