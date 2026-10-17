import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

import polars as pl
from rich import print as rprint

from youtube_to_docs.storage import Storage
from youtube_to_docs.transcript import download_video
from youtube_to_docs.utils import format_clickable_path

# Open captions: white sans-serif text on an 80% opaque black box, which keeps
# well above the WCAG 4.5:1 contrast minimum over any picture, at the bottom
# centre. Sizes are on libass's 288-line canvas, so FontSize 18 is about 1/16
# of the frame height whatever the resolution.
OPEN_CAPTION_STYLE = ",".join(
    [
        "FontName=Arial",
        "FontSize=18",
        "PrimaryColour=&H00FFFFFF",
        "OutlineColour=&H33000000",
        "BackColour=&H33000000",
        "BorderStyle=3",
        "Outline=2",
        "Shadow=0",
        "MarginV=18",
        "MarginL=30",
        "MarginR=30",
    ]
)


def create_video(image_path: str, audio_path: str, output_path: str) -> bool:
    """Creates an MP4 video from an image and an audio file using ffmpeg."""
    # Use static_ffmpeg to ensure ffmpeg is available
    try:
        from static_ffmpeg import run
    except ImportError as e:
        raise ImportError(
            "Missing dependencies for audio/video processing. "
            'Please run with: uvx "youtube-to-docs[all]"'
        ) from e

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


def _extract_lang_from_col(col_name: str) -> str:
    """Extracts a 2-letter language code from a column name, or 'en' if none found."""
    match = re.search(r"\(([a-z]{2})\)", col_name)
    return match.group(1) if match else "en"


def process_videos(
    df: pl.DataFrame, storage: Storage, base_dir: str = "."
) -> pl.DataFrame:
    """Processes the DataFrame to create videos from infographics and audio files.

    Supports multiple language pairs: groups infographic and audio columns by
    language suffix (e.g. `(es)`), creates one video per matched language pair,
    and stores results in `Video File` (English) or `Video File (es)` (translated).
    """

    # Setup Video Directory in Storage
    video_dir = os.path.join(base_dir, "video-files")
    storage.ensure_directory(video_dir)

    # Identify relevant columns
    info_cols = [c for c in df.columns if c.startswith("Summary Infographic File ")]
    audio_cols = [c for c in df.columns if c.startswith("Summary Audio File ")]

    if not info_cols or not audio_cols:
        print("Required columns (infographic and audio) not found in CSV.")
        return df

    # Group columns by language suffix
    info_by_lang: dict[str, list[str]] = defaultdict(list)
    for c in info_cols:
        info_by_lang[_extract_lang_from_col(c)].append(c)

    audio_by_lang: dict[str, list[str]] = defaultdict(list)
    for c in audio_cols:
        audio_by_lang[_extract_lang_from_col(c)].append(c)

    # Find languages that have both infographic and audio columns
    paired_langs = sorted(set(info_by_lang.keys()) & set(audio_by_lang.keys()))

    if not paired_langs:
        print("No matching infographic/audio column pairs found.")
        return df

    # video_results[lang] = list of (path or None) per row
    video_results: dict[str, list] = {lang: [] for lang in paired_langs}

    with tempfile.TemporaryDirectory() as temp_dir:
        rprint(f"Using temporary directory for video processing: {temp_dir}")

        for row in df.iter_rows(named=True):
            for lang in paired_langs:
                # Find the first valid infographic for this language
                infographic = None
                for c in info_by_lang[lang]:
                    path = row.get(c)
                    if path and isinstance(path, str) and storage.exists(path):
                        infographic = path
                        break

                # Find the first valid audio for this language
                audio = None
                for c in audio_by_lang[lang]:
                    path = row.get(c)
                    if path and isinstance(path, str) and storage.exists(path):
                        audio = path
                        break

                if not infographic or not audio:
                    video_results[lang].append(None)
                    continue

                # Determine output filename
                lang_suffix = f" ({lang})" if lang != "en" else ""
                if audio.startswith("http"):
                    video_id = None
                    if "URL" in row and row["URL"]:
                        match = re.search(r"v=([a-zA-Z0-9_-]+)", row["URL"])
                        if match:
                            video_id = match.group(1)

                    if video_id:
                        video_filename = f"{video_id}{lang_suffix}.mp4"
                    elif "Title" in row and row["Title"]:
                        safe_title = "".join(
                            [c if c.isalnum() else "_" for c in row["Title"]]
                        )
                        video_filename = f"{safe_title}{lang_suffix}.mp4"
                    else:
                        import uuid

                        video_filename = f"video_{uuid.uuid4()}{lang_suffix}.mp4"
                else:
                    audio_basename = os.path.basename(audio)
                    video_filename = os.path.splitext(audio_basename)[0] + ".mp4"

                target_video_path = os.path.join(video_dir, video_filename)

                # Check if video already exists in storage
                if storage.exists(target_video_path):
                    if hasattr(storage, "get_full_path"):
                        video_results[lang].append(
                            storage.get_full_path(target_video_path)
                        )
                    else:
                        video_results[lang].append(target_video_path)
                    rprint(f"Video already exists: {video_filename}")
                    continue

                rprint(f"Creating video: {video_filename}")

                info_ext = os.path.splitext(infographic)[1] or ".png"
                local_info_path = os.path.join(
                    temp_dir, f"input_image_{lang}{info_ext}"
                )
                ext = os.path.splitext(audio)[1] or ".m4a"
                local_audio_path = os.path.join(temp_dir, f"input_audio_{lang}{ext}")
                local_video_path = os.path.join(temp_dir, f"output_video_{lang}.mp4")

                try:
                    info_bytes = storage.read_bytes(infographic)
                    with open(local_info_path, "wb") as f:
                        f.write(info_bytes)

                    audio_bytes = storage.read_bytes(audio)
                    with open(local_audio_path, "wb") as f:
                        f.write(audio_bytes)

                    if create_video(
                        local_info_path, local_audio_path, local_video_path
                    ):
                        uploaded_link = storage.upload_file(
                            local_video_path,
                            target_video_path,
                            content_type="video/mp4",
                        )
                        rprint(
                            f"Successfully created and uploaded: "
                            f"{format_clickable_path(uploaded_link)}"
                        )
                        video_results[lang].append(uploaded_link)
                    else:
                        video_results[lang].append(None)
                except Exception as e:
                    print(f"Error processing video for row: {e}")
                    video_results[lang].append(None)

    # Add video columns to DataFrame
    for lang in paired_langs:
        col_name = "Video File" if lang == "en" else f"Video File ({lang})"
        if col_name in df.columns:
            df = df.with_columns(
                pl.when(pl.col(col_name).is_null())
                .then(pl.Series(video_results[lang]))
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )
        else:
            df = df.with_columns(pl.Series(name=col_name, values=video_results[lang]))

    return df


def burn_captions(
    video_path: str,
    srt_path: str,
    output_path: str,
    duration_seconds: Optional[float] = None,
) -> bool:
    """
    Re-encodes a video with the captions in `srt_path` drawn onto the picture
    (open captions), keeping the original audio. Prints progress when the
    duration is known. Returns True on success.
    """
    try:
        from static_ffmpeg import run
    except ImportError as e:
        raise ImportError(
            "Missing dependencies for audio/video processing. "
            'Please run with: uvx "youtube-to-docs[all]"'
        ) from e

    try:
        ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
    except Exception as e:
        print(f"Error fetching ffmpeg: {e}")
        return False

    with tempfile.TemporaryDirectory() as work:
        # The subtitles filter parses its argument as filter syntax, so the SRT
        # gets a plain relative name in ffmpeg's working directory.
        shutil.copyfile(srt_path, os.path.join(work, "captions.srt"))
        command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-progress",
            "pipe:1",
            "-y",
            "-i",
            os.path.abspath(video_path),
            "-vf",
            f"subtitles=captions.srt:force_style='{OPEN_CAPTION_STYLE}'",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            os.path.abspath(output_path),
        ]
        log_path = os.path.join(work, "ffmpeg.log")
        try:
            with open(log_path, "w") as log:
                process = subprocess.Popen(
                    command, cwd=work, stdout=subprocess.PIPE, stderr=log, text=True
                )
                reported = 0
                for line in process.stdout or []:
                    key, _, value = line.strip().partition("=")
                    if not duration_seconds or key not in (
                        "out_time_us",
                        "out_time_ms",
                    ):
                        continue
                    try:
                        done = int(value) / 1_000_000 / duration_seconds
                    except ValueError:
                        continue
                    if done >= (reported + 1) / 10 and reported < 9:
                        reported = min(int(done * 10), 9)
                        rprint(f"  Burning captions: {reported * 10}%")
                process.wait()
            if process.returncode != 0:
                with open(log_path) as log:
                    print(f"Error burning captions: {log.read().strip()[-500:]}")
                return False
            return True
        except OSError as e:
            print(f"Error burning captions: {e}")
            return False


def process_captioned_video(
    *,
    row: Dict[str, Any],
    storage: Storage,
    captions_path: str,
    model_name: str,
    source: str,
    video_id: str,
    safe_title: str,
    video_dir: str,
    duration_seconds: Optional[float],
    vprint: Callable[..., None] = print,
) -> None:
    """
    Saves a copy of the video with the corrected captions burned in (`-ccv`)
    to `video_dir`. Skips if it already exists.
    """
    col = f"Captioned Video File {model_name} from {source}"
    target = os.path.join(
        video_dir,
        f"{model_name} - {video_id} - {safe_title} - captioned video "
        f"(from {source}).mp4",
    )
    existing = row.get(col)
    if isinstance(existing, str) and existing and storage.exists(existing):
        vprint(f"Captioned video already exists: {existing}")
        return
    if storage.exists(target):
        row[col] = storage.get_full_path(target)
        vprint(f"Captioned video already exists: {target}")
        return

    with tempfile.TemporaryDirectory() as work:
        srt = storage.get_local_file(captions_path, download_dir=work)
        if not srt:
            print(f"Error: corrected captions not found at {captions_path}")
            return
        rprint(f"Downloading video {video_id} for open captions...")
        video = download_video(video_id, os.path.join(work, "download"))
        if not video:
            return
        rprint("Burning corrected captions into the video (re-encoding)...")
        output = os.path.join(work, "captioned.mp4")
        if not burn_captions(video, srt, output, duration_seconds):
            return
        try:
            saved = storage.upload_file(output, target, content_type="video/mp4")
        except Exception as e:
            print(f"Error saving captioned video: {e}")
            return
    row[col] = saved
    rprint(f"Saved captioned video: {format_clickable_path(saved)}")
