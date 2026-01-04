import os
import subprocess
import tempfile

import polars as pl
from rich import print as rprint

from youtube_to_docs.storage import Storage
from youtube_to_docs.utils import format_clickable_path


def create_video(image_path: str, audio_path: str, output_path: str) -> bool:
    """Creates an MP4 video from an image and an audio file using ffmpeg."""
    # Use static_ffmpeg to ensure ffmpeg is available
    from static_ffmpeg import run

    try:
        ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
    except Exception as e:
        print(f"Error fetching ffmpeg: {e}")
        return False

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


def process_videos(
    df: pl.DataFrame, storage: Storage, base_dir: str = "."
) -> pl.DataFrame:
    """Processes the DataFrame to create videos from infographics and audio files."""

    # Setup Video Directory in Storage
    video_dir = os.path.join(base_dir, "video-files")
    storage.ensure_directory(video_dir)

    # Identify relevant columns
    info_cols = [c for c in df.columns if c.startswith("Summary Infographic File ")]
    audio_cols = [c for c in df.columns if c.startswith("Summary Audio File ")]

    if not info_cols or not audio_cols:
        print("Required columns (infographic and audio) not found in CSV.")
        return df

    video_files = []

    # Create a temporary directory for local processing (download/ffmpeg)
    with tempfile.TemporaryDirectory() as temp_dir:
        rprint(f"Using temporary directory for video processing: {temp_dir}")

        for row in df.iter_rows(named=True):
            # Find valid infographics and audios for this row
            infographics = []
            for c in info_cols:
                path = row.get(c)
                if path and isinstance(path, str) and storage.exists(path):
                    infographics.append(path)

            audios = []
            for c in audio_cols:
                path = row.get(c)
                if path and isinstance(path, str) and storage.exists(path):
                    audios.append(path)

            if len(infographics) == 1 and len(audios) == 1:
                info_path_remote = infographics[0]
                audio_path_remote = audios[0]

                # Determine output filename
                # Use audio filename as base, but ensure it ends in .mp4
                # We need to handle if audio_path_remote is a URL or path
                if audio_path_remote.startswith("http"):
                    # Try to extract video ID from URL
                    video_id = None
                    if "URL" in row and row["URL"]:
                        import re

                        match = re.search(r"v=([a-zA-Z0-9_-]+)", row["URL"])
                        if match:
                            video_id = match.group(1)

                    if video_id:
                        video_filename = f"{video_id}.mp4"
                    elif "Title" in row and row["Title"]:
                        safe_title = "".join(
                            [c if c.isalnum() else "_" for c in row["Title"]]
                        )
                        video_filename = f"{safe_title}.mp4"
                    else:
                        import uuid

                        video_filename = f"video_{uuid.uuid4()}.mp4"
                else:
                    audio_basename = os.path.basename(audio_path_remote)
                    video_filename = os.path.splitext(audio_basename)[0] + ".mp4"

                target_video_path = os.path.join(video_dir, video_filename)

                # Check if video already exists in storage
                if storage.exists(target_video_path):
                    # If we can get a full path/link, use it
                    if hasattr(storage, "get_full_path"):
                        video_files.append(storage.get_full_path(target_video_path))
                    else:
                        video_files.append(target_video_path)
                    rprint(f"Video already exists: {video_filename}")
                    continue

                rprint(f"Creating video: {video_filename}")

                # Download files to temp dir
                local_info_path = os.path.join(temp_dir, "input_image.png")
                # Preserve extension or default to .m4a if unknown
                ext = os.path.splitext(audio_path_remote)[1] or ".m4a"
                local_audio_path = os.path.join(temp_dir, f"input_audio{ext}")
                local_video_path = os.path.join(temp_dir, "output_video.mp4")

                try:
                    # Download Infographic
                    info_bytes = storage.read_bytes(info_path_remote)
                    with open(local_info_path, "wb") as f:
                        f.write(info_bytes)

                    # Download Audio
                    audio_bytes = storage.read_bytes(audio_path_remote)
                    with open(local_audio_path, "wb") as f:
                        f.write(audio_bytes)

                    # Create Video
                    if create_video(
                        local_info_path, local_audio_path, local_video_path
                    ):
                        # Upload Video
                        # We can use storage.upload_file if we have a path,
                        # but storage.upload_file expects a local path.
                        uploaded_link = storage.upload_file(
                            local_video_path,
                            target_video_path,
                            content_type="video/mp4",
                        )
                        rprint(
                            f"Successfully created and uploaded: "
                            f"{format_clickable_path(uploaded_link)}"
                        )
                        video_files.append(uploaded_link)
                    else:
                        video_files.append(None)
                except Exception as e:
                    print(f"Error processing video for row: {e}")
                    video_files.append(None)

            else:
                if len(infographics) > 1 or len(audios) > 1:
                    print(
                        f"Skipping row for {row.get('Title', 'Unknown')}: "
                        f"Multiple infographics ({len(infographics)}) or "
                        f"audios ({len(audios)}) found. Ambiguous."
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

    return df
