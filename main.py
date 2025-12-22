# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "google-genai>=1.56.0",
#     "google-api-python-client>=2.187.0",
#     "isodate>=0.7.2",
#     "youtube-transcript-api>=1.2.3"
# ]
# ///
#
# Run as:
# uv run https://raw.githubusercontent.com/DoIT-Artifical-Intelligence/youtube-to-docs/refs/heads/main/main.py -- PL8ZxoInteClyHaiReuOHpv6Z4SPrXtYtW

import os
import argparse
from googleapiclient.discovery import build

try:
    YOUTUBE_DATA_API_KEY = os.environ["YOUTUBE_DATA_API_KEY"]
    youtube_service = build("youtube", "v3", developerKey=YOUTUBE_DATA_API_KEY)
except KeyError:
    YOUTUBE_DATA_API_KEY = None
    youtube_service = None
    print("Warning: YOUTUBE_DATA_API_KEY not found. Playlist expansion will fail.")

parser = argparse.ArgumentParser()
parser.add_argument(
    "video_id",
    nargs='?',
    default="KuPc06JgI_A",
    help="A Video ID e.g. 'KuPc06JgI_A', Playlist ID (starts with PL e.g. 'PL8ZxoInteClyHaiReuOHpv6Z4SPrXtYtW'), or comma-separated list of Video IDs. (e.g. 'KuPc06JgI_A,GalhDyf3F8g')"
)
args = parser.parse_args()
video_id = args.video_id

# Single video
if len(video_id) == 11:
    video_ids = [video_id]
# List of videos
if ",".in(video_id):
    video_ids = video_id.split(',')
# Playlist
if video_id[0:2] == "PL":
    video_ids = []
    request = service.playlistItems().list(
        part="contentDetails",
        playlistId=playlist_id,
        maxResults=50
    )
    while request:
        response = request.execute()
        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])
        request = service.playlistItems().list_next(request, response)

print(video_ids)