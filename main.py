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
# uv run https://raw.githubusercontent.com/DoIT-Artifical-Intelligence/youtube-to-docs/refs/heads/main/main.py --KuPc06JgI_A
import os
import argparse
from googleapiclient.discovery import build

# 1. Setup Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "video_id",
    nargs='?',
    default="KuPc06JgI_A",
    help="The YouTube Video ID (defaults to KuPc06JgI_A)")
args = parser.parse_args()

# Access the variable
print(f"Processing Video ID: {args.video_id}")

# 2. Setup YouTube Service
try:
    YOUTUBE_DATA_API_KEY = os.environ["YOUTUBE_DATA_API_KEY"]
    youtube_service = build("youtube", "v3", developerKey=YOUTUBE_DATA_API_KEY)
except KeyError:
    YOUTUBE_DATA_API_KEY = None
    youtube_service = None
    print("Warning: YOUTUBE_DATA_API_KEY not found in environment.")