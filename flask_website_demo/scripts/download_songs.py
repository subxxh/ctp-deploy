"""Helper script to download Youtube Videos or equivalent videos from soundcloud/spotify and convert to a .WAV file"""

import base64
import re
from pathlib import Path

import requests
from yt_dlp import YoutubeDL

# --- CONFIG ---
SPOTIFY_CLIENT_ID = "675c5ff0168c4157bbc6a419b0703647"
SPOTIFY_CLIENT_SECRET = "367277e14cc74c83a5e05ca8af63651c"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

FFMPEG_PATH = str(PROJECT_ROOT / "ffmpeg" / "bin")
DOWNLOADS_DIR = BASE_DIR / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

YDL_OPTS = {
    "format": "bestaudio/best",
    "ffmpeg_location": FFMPEG_PATH,
    "cookiesfrombrowser": ["firefox"],
    "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    "js_runtimes": {"node": {}},
    "quiet": True,
    "compat_opts": set(),
    "remote_components": {"ejs:github"},
    "outtmpl": str(DOWNLOADS_DIR / "%(title)s.%(ext)s"),
    }


# SPOTIFY METADATA
def get_spotify_metadata(track_url):
    credentials = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    encoded = base64.b64encode(credentials.encode()).decode()

    token_headers = {
        "Authorization": "Basic " + encoded,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        headers=token_headers,
    )
    data = r.json()

    if "access_token" not in data:
        return None

    token = data["access_token"]

    match = re.search(r"track/([A-Za-z0-9]+)", track_url)
    if not match:
        return None

    track_id = match.group(1)
    auth = {"Authorization": f"Bearer {token}"}

    track_data = requests.get(
        f"https://api.spotify.com/v1/tracks/{track_id}", headers=auth
    ).json()

    return {
        "title": track_data["name"],
        "author": track_data["artists"][0]["name"],
    }


# SOUNDCLOUD METADATA
def get_soundcloud_metadata(track_url):
    with YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(track_url, download=False)
    return {"title": info["title"], "author": info["uploader"]}


# YOUTUBE SEARCH DOWNLOAD
def youtube_search(query):
    with YoutubeDL({"quiet": True}) as ydl:
        results = ydl.extract_info(f"ytsearch1:{query}", download=False)
    return results["entries"][0]["webpage_url"]

def get_youtube_embed_url(url: str) -> str:
    """Convert a YouTube URL to an embed URL."""
    # Standard URL ?v=VIDEO_ID
    match = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]+)", url)
    if match:
        video_id = match.group(1)
        return f"https://www.youtube.com/embed/{video_id}"
    return url  

# MAIN DOWNLOAD LOGIC
def download_audio(url: str):
    """
    Download audio:
    - Direct YouTube URL
    - Spotify track → search on YouTube
    - SoundCloud → search on YouTube
    Returns:
        Path to WAV file
        Final YouTube URL used
    """

    # Spotify
    if "spotify.com" in url:
        meta = get_spotify_metadata(url)
        if not meta:
            return None, None

        search_query = f"{meta['title']} {meta['author']} audio"
        url = youtube_search(search_query)

    # SoundCloud
    if "soundcloud.com" in url:
        meta = get_soundcloud_metadata(url)
        search_query = f"{meta['title']} {meta['author']} audio"
        url = youtube_search(search_query)

    # YouTube / final audio download
    with YoutubeDL(YDL_OPTS) as ydl:
        info = ydl.extract_info(url, download=True)

    # Construct output path
    wav_path = DOWNLOADS_DIR / (info["title"] + ".wav")
    
    # Convert URL to embed URL
    embed_url = get_youtube_embed_url(info["webpage_url"])
    
    return wav_path, embed_url
