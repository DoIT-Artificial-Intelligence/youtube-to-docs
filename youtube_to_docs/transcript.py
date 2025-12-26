import os
import sys
from typing import Any, List, Optional, Tuple, cast

import isodate
from googleapiclient.discovery import Resource, build
from youtube_transcript_api import YouTubeTranscriptApi

# Global instance for transcript API
ytt_api = YouTubeTranscriptApi()


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


def fetch_transcript(video_id: str) -> Optional[Tuple[str, bool]]:
    """Fetches the transcript for a given video ID and returns (text, is_generated)."""
    try:
        transcript_obj = ytt_api.fetch(video_id, languages=("en", "en-US"))
        is_generated = getattr(transcript_obj, "is_generated", False)
        transcript_data = transcript_obj.to_raw_data()
        transcript = " ".join([t["text"] for t in transcript_data])
        return transcript, is_generated
    except Exception as e:
        print(f"Error fetching transcript for {video_id}: {e}")
        return None
