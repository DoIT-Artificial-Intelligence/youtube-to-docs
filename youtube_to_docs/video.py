import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import polars as pl
from rich import print as rprint

from youtube_to_docs.captions import normalize_srt
from youtube_to_docs.storage import Storage
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
        # Fit 4K UHD so large infographics stay playable, keeping even sides.
        "-vf",
        "scale=w='min(3840,iw)':h='min(2160,ih)':force_original_aspect_ratio="
        "decrease:force_divisible_by=2",
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


def render_accessible_video(
    video_path: str,
    srt_path: str,
    output_path: str,
    duration_seconds: Optional[float] = None,
    narration_path: Optional[str] = None,
) -> bool:
    """
    Re-encodes a video with the captions in `srt_path` drawn onto the picture
    (open captions) and, if given, the audio-description narration mixed in,
    lowering the programme audio while it plays. Prints progress when the
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

    subtitles = f"subtitles=captions.srt:force_style='{OPEN_CAPTION_STYLE}'"
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
        ]
        if narration_path:
            command += [
                "-i",
                os.path.abspath(narration_path),
                "-filter_complex",
                f"[0:v]{subtitles}[video];"
                "[1:a]aresample=48000,aformat=channel_layouts=stereo,"
                "asplit=2[side][voice];"
                "[0:a]aresample=48000,aformat=channel_layouts=stereo[programme];"
                "[programme][side]sidechaincompress=threshold=0.02:ratio=8:"
                "attack=20:release=400[ducked];"
                "[ducked][voice]amix=inputs=2:duration=first:normalize=0[audio]",
                "-map",
                "[video]",
                "-map",
                "[audio]",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
            ]
        else:
            command += ["-vf", subtitles, "-c:a", "copy"]
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
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
                        rprint(f"  Rendering video: {reported * 10}%")
                process.wait()
            if process.returncode != 0:
                with open(log_path) as log:
                    print(f"Error rendering video: {log.read().strip()[-500:]}")
                return False
            return True
        except OSError as e:
            print(f"Error rendering video: {e}")
            return False


def save_original_video(
    *,
    row: Dict[str, Any],
    storage: Storage,
    get_video: Callable[[], Optional[str]],
    path: str,
) -> None:
    """Keeps the downloaded video in `video-files/` (the source for website
    players). Skips if it's already there."""
    col = "Original Video File"
    if storage.exists(path):
        row[col] = row.get(col) or storage.get_full_path(path)
        return
    video = get_video()
    if not video:
        return
    try:
        saved = storage.upload_file(video, path, content_type="video/mp4")
    except Exception as e:
        print(f"Error saving the original video: {e}")
        return
    row[col] = saved
    rprint(f"Saved original video: {format_clickable_path(saved)}")


def process_accessible_video(
    *,
    row: Dict[str, Any],
    storage: Storage,
    captions_path: str,
    model_name: str,
    source: str,
    descriptions: Optional[List[Any]],
    descriptions_vtt: Optional[str],
    tts_arg: Optional[str],
    get_video: Callable[[], Optional[str]],
    video_id: str,
    safe_title: str,
    video_dir: str,
    duration_seconds: Optional[float],
    vprint: Callable[..., None] = print,
) -> None:
    """
    Saves a copy of the video with the corrected captions burned in and the
    audio description narrated in the pauses (part of `-cc`). `descriptions`
    are the narrated standard descriptions from this run; otherwise they are
    read back from `descriptions_vtt` and narrated again. Skips if it exists.
    """
    from youtube_to_docs.audio_description import (
        DEFAULT_NARRATION_TTS,
        descriptions_from_webvtt,
        synthesize,
        write_narration_track,
    )

    col = f"Captioned And Described Video File {model_name} from {source}"
    target = os.path.join(
        video_dir,
        f"{model_name} - {video_id} - {safe_title} - captioned and described "
        f"video (from {source}).mp4",
    )
    existing = row.get(col)
    if isinstance(existing, str) and existing and storage.exists(existing):
        vprint(f"Captioned and described video already exists: {existing}")
        return
    if storage.exists(target):
        row[col] = storage.get_full_path(target)
        vprint(f"Captioned and described video already exists: {target}")
        return

    with tempfile.TemporaryDirectory() as work:
        srt = os.path.join(work, "captions.srt")
        try:
            content = storage.read_text(captions_path)
        except Exception as e:
            print(f"Error: corrected captions not found at {captions_path}: {e}")
            return
        with open(srt, "w", encoding="utf-8") as f:
            f.write(normalize_srt(content))
        video = get_video()
        if not video:
            return
        if (
            descriptions is None
            and descriptions_vtt
            and storage.exists(descriptions_vtt)
        ):
            descriptions = descriptions_from_webvtt(storage.read_text(descriptions_vtt))
            synthesize(descriptions, tts_arg or DEFAULT_NARRATION_TTS)
        narration = None
        if descriptions:
            narration = os.path.join(work, "narration.wav")
            if not write_narration_track(
                descriptions, duration_seconds or 0.0, narration
            ):
                narration = None
        rprint(
            "Rendering the video with captions"
            + (" and audio description" if narration else "")
            + " (re-encoding)..."
        )
        output = os.path.join(work, "accessible.mp4")
        if not render_accessible_video(video, srt, output, duration_seconds, narration):
            return
        try:
            saved = storage.upload_file(output, target, content_type="video/mp4")
        except Exception as e:
            print(f"Error saving the captioned and described video: {e}")
            return
    row[col] = saved
    rprint(f"Saved captioned and described video: {format_clickable_path(saved)}")


ABLE_PLAYER_URL = "https://cdn.jsdelivr.net/npm/ableplayer@5.0.0/build"
JQUERY_URL = "https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"

# Able Player 5.0 pauses the video for each description and only resumes it
# when the speech ends, so the video can stay paused: if the first
# description comes before the browser's voices have loaded (nothing is
# read), if the speech fails, or if Chrome drops the end of the speech.
ABLE_PLAYER_FIXES = """
    <script>
      (function () {
        const synth = window.speechSynthesis;
        if (!window.AblePlayer || !synth) return;
        const proto = AblePlayer.prototype;
        // Chrome loads its voices in the background: start before Play.
        synth.getVoices();
        synth.addEventListener("voiceschanged", () => synth.getVoices());
        // Chrome can drop the events of speech nothing else refers to.
        const utterances = new Set();
        const speak = synth.speak.bind(synth);
        synth.speak = (utterance) => {
          utterances.add(utterance);
          const done = () => utterances.delete(utterance);
          utterance.addEventListener("end", done);
          utterance.addEventListener("error", done);
          speak(utterance);
        };
        const players = new Set();
        const showDescription = proto.showDescription;
        proto.showDescription = function (now) {
          players.add(this);
          const reads = this.descReader !== "screenreader";
          if (reads && !this.speechEnabled && synth.getVoices().length) {
            // Speech is switched on by the first click, which can come
            // before the voices have loaded.
            this.speechEnabled = true;
            this.getBrowserVoices();
          }
          // Only pause for a description that will be read aloud.
          const pause = this.prefDescPause;
          if (reads && !this.speechEnabled) this.prefDescPause = 0;
          try {
            showDescription.call(this, now);
          } finally {
            this.prefDescPause = pause;
          }
        };
        // Able Player never clears this flag, so a later pause by the
        // viewer could be undone when a description ends.
        const playMedia = proto.playMedia;
        proto.playMedia = function () {
          this.pausedForDescription = false;
          return playMedia.apply(this, arguments);
        };
        const handlePlay = proto.handlePlay;
        proto.handlePlay = function (e) {
          if (!this.paused) this.pausedForDescription = false;
          return handlePlay.apply(this, arguments);
        };
        // Resume a video left paused for a description once the speech has
        // stopped (or after a minute, if Chrome never says it ended).
        setInterval(() => {
          for (const player of players) {
            const waiting = player.pausedForDescription && player.paused &&
              player.okToPlay !== false && !synth.paused;
            if (!waiting) {
              player.descWaitStart = null;
              continue;
            }
            player.descWaitStart = player.descWaitStart || Date.now();
            const waited = Date.now() - player.descWaitStart;
            const quiet = !synth.speaking && !synth.pending;
            if ((quiet && waited > 1500) || waited > 60000) {
              if (!quiet) synth.cancel();
              player.descWaitStart = null;
              player.playMedia();
            }
          }
        }, 500);
      })();
    </script>"""


def player_page_html(
    title: str,
    video_src: str,
    captions_src: str,
    descriptions_src: Optional[str],
    transcript_href: Optional[str],
    track_texts: Optional[Dict[str, str]] = None,
    youtube_id: Optional[str] = None,
) -> str:
    """
    A web page that plays the video in Able Player with the WebVTT captions
    and descriptions tracks, for checking them in a browser and as a template
    for a website. Captions and descriptions start on: Able Player reads the
    descriptions aloud and pauses the video while each one is read (extended
    description).

    `track_texts` (WebVTT text by track kind) are embedded so the page also
    works when opened straight from disk: browsers won't load .vtt files into
    a file:// page, so there the tracks use the embedded copies.

    With `youtube_id` the video streams from YouTube and the tracks always use
    the embedded copies, so the page needs no other files: it can be pasted
    into a site builder's embed box (e.g. Google Sites' Embed code).
    """
    name = html.escape(title or "Video")
    copies = ""
    if track_texts:
        # "<" is escaped so the text can't end the script element early.
        data = json.dumps(track_texts).replace("<", "\\u003c")
        # Browsers won't load .vtt files into a page opened from disk, and a
        # page pasted into a site has no files beside it.
        condition = "true" if youtube_id else 'location.protocol === "file:"'
        copies = f"""
      <script type="application/json" id="track-copies">{data}</script>
      <script>
        if ({condition}) {{
          const copies = JSON.parse(
            document.getElementById("track-copies").textContent);
          for (const track of document.querySelectorAll("#video1 track")) {{
            if (copies[track.kind]) {{
              track.src = URL.createObjectURL(
                new Blob([copies[track.kind]], {{type: "text/vtt"}}));
            }}
          }}
        }}
      </script>"""
    source = (
        f' data-youtube-id="{html.escape(youtube_id)}">'
        if youtube_id
        else f'>\n        <source type="video/mp4" src="{html.escape(video_src)}">'
    )
    descriptions = (
        f'\n        <track kind="descriptions" src="{html.escape(descriptions_src)}" '
        'srclang="en" label="English">'
        if descriptions_src
        else ""
    )
    transcript = (
        f'\n      <p><a href="{html.escape(transcript_href)}">Descriptive transcript'
        "</a> (the dialogue with descriptions of the visual information)</p>"
        if transcript_href
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{name}: accessible player</title>
    <link rel="stylesheet" href="{ABLE_PLAYER_URL}/ableplayer.min.css">
    <style>
      /* Keep the player and its controls on screen (16:9 video). */
      main {{ max-width: min(60rem, (100vh - 20rem) * 16 / 9); margin: 0 auto; }}
    </style>
  </head>
  <body>
    <main>
      <h1>{name}</h1>
      <p>Captions and audio descriptions are on: the slides and other visual
      information are described aloud, and the video pauses while each
      description is read. Select the Description button to turn them off, and
      the Transcript button to read along.</p>
      <video id="video1" data-able-player preload="metadata" playsinline
        data-desc-pause-default="on" data-state-captions="on"
        data-state-descriptions="on"{source}
        <track kind="captions" src="{html.escape(captions_src)}" srclang="en"
          label="English" default>{descriptions}
      </video>{copies}{transcript}
    </main>
    <script src="{JQUERY_URL}"></script>
    <script src="{ABLE_PLAYER_URL}/ableplayer.min.js"></script>{ABLE_PLAYER_FIXES}
  </body>
</html>
"""


def write_player_page(
    *,
    row: Dict[str, Any],
    storage: Storage,
    path: str,
    title: str,
    video_path: str,
    captions_path: str,
    descriptions_path: Optional[str],
    transcript_path: Optional[str],
    youtube_id: Optional[str] = None,
    column: str = "Player Page",
) -> None:
    """Writes the Able Player page (see player_page_html) with links relative
    to the page, so the folder can be served or published as it is, and
    copies of the tracks for opening it straight from disk. With `youtube_id`
    it writes the self-contained embed version, which only links the
    transcript if it has a full URL (e.g. a Google Doc)."""
    folder = os.path.dirname(path)

    def link(target: Optional[str]) -> Optional[str]:
        if not target:
            return None
        if target.startswith("http"):
            return target
        relative = os.path.relpath(target, folder).replace(os.sep, "/")
        return urllib.parse.quote(relative)

    track_texts = {}
    for kind, track in (
        ("captions", captions_path),
        ("descriptions", descriptions_path),
    ):
        try:
            if track and storage.exists(track):
                track_texts[kind] = storage.read_text(track)
        except Exception as e:
            print(f"Warning: could not embed the {kind} track in the player: {e}")
    page = player_page_html(
        title,
        link(video_path) or "",
        link(captions_path) or "",
        link(descriptions_path),
        (
            transcript_path
            if transcript_path and transcript_path.startswith("http")
            else None
        )
        if youtube_id
        else link(transcript_path),
        track_texts,
        youtube_id,
    )
    kind = "embed page" if youtube_id else "player page"
    try:
        saved = storage.write_text(path, page)
    except Exception as e:
        print(f"Error writing the {kind}: {e}")
        return
    row[column] = saved
    rprint(f"Saved {kind}: {format_clickable_path(saved)}")
