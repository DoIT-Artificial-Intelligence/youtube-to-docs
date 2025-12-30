import os
import sys
from typing import Any, List, Optional, Tuple, cast

import isodate
import static_ffmpeg
import yt_dlp
from googleapiclient.discovery import Resource, build
from youtube_transcript_api import (
    IpBlocked,
    NoTranscriptFound,
    TranscriptsDisabled,
    TranslationLanguageNotAvailable,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

# Ensure ffmpeg is in path
static_ffmpeg.add_paths()


def extract_audio(video_id: str, output_dir: str) -> Optional[str]:
    """Extracts audio from a YouTube video using yt-dlp."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio[ext=m4a]",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                # The filename returned by prepare_filename should have the
                # correct extension
                filename = ydl.prepare_filename(info)
                # Ensure .m4a extension
                base, _ = os.path.splitext(filename)
                return os.path.abspath(f"{base}.m4a")
    except Exception as e:
        print(f"Error extracting audio for {video_id}: {e}")
    return None


def get_youtube_service() -> Optional[Resource]:
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


def resolve_video_ids(
    video_id_input: str, youtube_service: Optional[Resource]
) -> List[str]:
    """
    Resolves the input (video ID, list, playlist, or channel handle)
    into a list of video IDs.
    """
    video_ids: List[str] = []

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
    video_id: str, youtube_service: Optional[Resource]
) -> Optional[Tuple[str, str, str, str, str, str, str]]:
    """
    Fetches video metadata from YouTube Data API.
    Returns a tuple of (video_title, description, publishedAt,
    channelTitle, tags, video_duration, url).
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    if not youtube_service:
        return "", "", "", "", "", "", url

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
        video_duration: str = str(isodate.parse_duration(iso_duration))
        return (
            video_title,
            description,
            publishedAt,
            channelTitle,
            tags,
            video_duration,
            url,
        )
    else:
        print(f"Warning: No details found for video ID {video_id}")
        return None


def fetch_transcript(video_id: str, language: str = "en") -> Optional[Tuple[str, bool]]:
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
            transcript = " ".join([t.text for t in transcript_data])
            is_generated = transcript_obj.is_generated
            if transcript_obj.translation_languages:
                # If it was translated, we consider it "generated" in a sense,
                # or at least not the original human source.
                # But strictly `is_generated` refers to ASR.
                # Let's keep it as is from the object, but if it's a translation
                # it might be good to know. For now, just return what the obj says.
                pass
            return transcript, is_generated

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
