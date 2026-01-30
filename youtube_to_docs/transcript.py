"""Helpers for YouTube metadata, audio extraction, and transcript retrieval."""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import isodate
from googleapiclient.discovery import build
from youtube_transcript_api import (
    IpBlocked,
    NoTranscriptFound,
    TranscriptsDisabled,
    TranslationLanguageNotAvailable,
    VideoUnavailable,
    YouTubeTranscriptApi,
)


def extract_audio(
    video_id: str,
    output_dir: str,
    ext: str = "m4a",
) -> Optional[str]:
    """Extracts audio from a YouTube video using yt-dlp."""
    try:
        import static_ffmpeg
        import yt_dlp

        # Ensure ffmpeg is in path
        static_ffmpeg.add_paths()
    except ImportError as e:
        raise ImportError(
            "Missing dependencies for audio/video processing. "
            'Please run with: uvx "youtube-to-docs[all]"'
        ) from e

    url = f"https://www.youtube.com/watch?v={video_id}"
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts: Dict[str, Any] = {
        "format": f"bestaudio[ext={ext}]/bestaudio",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": ext,
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                # The filename returned by prepare_filename might not reflect
                # post-processing
                # So we construct it manually based on expectation
                filename = ydl.prepare_filename(info)
                base, _ = os.path.splitext(filename)
                final_path = os.path.abspath(f"{base}.{ext}")
                if os.path.exists(final_path):
                    return final_path
                # Fallback if yt-dlp didn't rename it as expected (rare)
                return os.path.abspath(filename)
    except Exception as e:
        print(f"Error extracting audio for {video_id}: {e}")
    return None


def get_youtube_service() -> Optional[Any]:
    """Builds and returns the YouTube Data API service."""
    try:
        api_key = os.environ["YOUTUBE_DATA_API_KEY"]
        return build("youtube", "v3", developerKey=api_key)
    except KeyError:
        print(
            "Warning: YOUTUBE_DATA_API_KEY not found. Playlist and Channel expansion "
            "will fail."
        )
        return None


def resolve_video_ids(video_id_input: str, youtube_service: Optional[Any]) -> List[str]:
    """
    Resolves the input (video ID, list, playlist, or channel handle)
    into a list of video IDs.
    """
    video_ids: List[str] = []

    # Handle full URLs
    if "youtube.com" in video_id_input or "youtu.be" in video_id_input:
        # Regex to capture the 11-character video ID from common URL formats
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_id_input)
        if match:
            video_id_input = match.group(1)
            print(f"Extracted video ID from URL: {video_id_input}")

    # Handle Channel Handles (e.g. @channelname)
    if video_id_input.startswith("@"):
        if not youtube_service:
            print("Error: YOUTUBE_DATA_API_KEY is required to resolve channel handles.")
            sys.exit(1)
        service = cast(Any, youtube_service)
        print(f"Resolving channel handle: {video_id_input}...")
        request = service.channels().list(
            part="contentDetails", forHandle=video_id_input
        )
        response = request.execute()
        if not response["items"]:
            print(f"Error: No channel found for handle {video_id_input}")
            sys.exit(1)
        # Get the 'uploads' playlist ID from the channel details
        video_id_input = response["items"][0]["contentDetails"]["relatedPlaylists"][
            "uploads"
        ]
        print(f"Found uploads playlist: {video_id_input}")

    # Single video (standard ID length is 11)
    if len(video_id_input) == 11 and "," not in video_id_input:
        video_ids = [video_id_input]
    # List of videos
    elif "," in video_id_input:
        video_ids = video_id_input.split(",")
    # Playlist (Standard 'PL' or Uploads 'UU')
    elif video_id_input.startswith("PL") or video_id_input.startswith("UU"):
        if not youtube_service:
            print("Error: YOUTUBE_DATA_API_KEY is required for playlists.")
            sys.exit(1)
        service = cast(Any, youtube_service)
        request = service.playlistItems().list(
            part="contentDetails", playlistId=video_id_input, maxResults=50
        )
        while request:
            response = request.execute()
            for item in response["items"]:
                video_ids.append(item["contentDetails"]["videoId"])
            request = service.playlistItems().list_next(request, response)

    return video_ids


def get_video_details(
    video_id: str, youtube_service: Optional[Any]
) -> Optional[Tuple[str, str, str, str, str, str, str, float]]:
    """
    Fetches video metadata from YouTube Data API.
    Returns a tuple of (video_title, description, publishedAt,
    channelTitle, tags, video_duration, url, video_duration_seconds).
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    if not youtube_service:
        return "", "", "", "", "", "", url, 0.0

    service = cast(Any, youtube_service)
    request = service.videos().list(part="snippet,contentDetails", id=video_id)
    response = request.execute()

    if response["items"]:
        snippet = response["items"][0]["snippet"]
        video_title: str = snippet["title"]
        description: str = snippet["description"]
        publishedAt: str = snippet["publishedAt"]
        channelTitle: str = snippet["channelTitle"]
        tags: str = ", ".join(snippet.get("tags", []))
        iso_duration: str = response["items"][0]["contentDetails"]["duration"]
        duration_obj = isodate.parse_duration(iso_duration)
        video_duration: str = str(duration_obj)
        video_duration_seconds: float = duration_obj.total_seconds()
        return (
            video_title,
            description,
            publishedAt,
            channelTitle,
            tags,
            video_duration,
            url,
            video_duration_seconds,
        )
    else:
        print(f"Warning: No details found for video ID {video_id}")
        return None


def fetch_transcript(
    video_id: str, language: str = "en"
) -> Optional[Tuple[str, bool, List[Dict[str, Any]]]]:
    """
    Fetches the transcript for a given video ID.
    Tries to find a transcript in the requested language.
    If not found, tries to translate an English transcript (or any available)
    to the requested language.
    Returns (text, is_generated).
    """
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
        transcript_obj = None

        # 1. Try exact match (manual)
        try:
            transcript_obj = transcript_list.find_manually_created_transcript(
                [language]
            )
        except Exception:
            pass

        # 2. Try exact match (generated)
        if not transcript_obj:
            try:
                transcript_obj = transcript_list.find_generated_transcript([language])
            except Exception:
                pass

        # 3. Try translating from English (manual)
        if not transcript_obj:
            try:
                en_transcript = transcript_list.find_manually_created_transcript(
                    ["en", "en-US", "en-GB"]
                )
                transcript_obj = en_transcript.translate(language)
            except Exception:
                pass

        # 4. Try translating from English (generated)
        if not transcript_obj:
            try:
                en_transcript = transcript_list.find_generated_transcript(
                    ["en", "en-US", "en-GB"]
                )
                transcript_obj = en_transcript.translate(language)
            except Exception:
                pass

        # 5. Try translating from ANY available
        if not transcript_obj:
            try:
                # Just take the first one
                first_transcript = next(iter(transcript_list))
                transcript_obj = first_transcript.translate(language)
            except Exception:
                pass

        if transcript_obj:
            transcript_data = transcript_obj.fetch()

            # Handle both dicts and objects
            # (some versions return FetchedTranscriptSnippet)
            def get_val(item, key):
                return item[key] if isinstance(item, dict) else getattr(item, key)

            transcript_text = " ".join([get_val(t, "text") for t in transcript_data])
            is_generated = bool(transcript_obj.is_generated)
            return (
                str(transcript_text),
                is_generated,
                cast(List[Dict[str, Any]], transcript_data),
            )

        return None

    except (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
        TranslationLanguageNotAvailable,
    ):
        print(
            f"Transcript not available for {video_id} in language '{language}' "
            "(or translation failed)."
        )
        return None
    except IpBlocked:
        print(
            f"Warning: YouTube returned an IP Blocked error for {video_id}. "
            "This might be due to rate limiting or the transcript isn't available."
        )
        return None
    except Exception as e:
        print(f"Error fetching transcript for {video_id}: {e}")
        return None


def format_as_srt(transcript_data: List[Any]) -> str:
    """Formats raw transcript data (list of dicts/objects) as an SRT string."""
    srt_output = []

    def get_val(item, key):
        return item[key] if isinstance(item, dict) else getattr(item, key)

    for i, entry in enumerate(transcript_data, 1):
        start = get_val(entry, "start")
        duration = get_val(entry, "duration")
        end = start + duration
        text = get_val(entry, "text")

        start_time = format_srt_timestamp(start)
        end_time = format_srt_timestamp(end)

        srt_output.append(f"{i}")
        srt_output.append(f"{start_time} --> {end_time}")
        srt_output.append(f"{text}\n")

    return "\n".join(srt_output)


def format_srt_timestamp(seconds: float) -> str:
    """Formats seconds into SRT timestamp format (HH:MM:SS,mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds * 1000) % 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"
