"""
Utility Module

Shared helper functions for the distributed video inference pipeline:
- Configuration loading
- Video hashing
- YouTube URL resolution and downloading
- URL pattern matching
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import re
import os
import time
import yaml
import logging
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


# --- Configuration Loading ---

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Config file not found at '%s', using defaults.", config_path)
        return {
            "backends": [{"url": "http://localhost:8000", "model": "yolov8n", "name": "Default"}],
            "buffer_size": 300,
            "target_fps": 30
        }


# --- Video Utilities ---

def generate_video_hash(video_path: str) -> str:
    """Generate a SHA-256 hash for a video file."""
    sha256_hash = hashlib.sha256()
    with open(video_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def is_url(path):
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(path) is not None


# --- YouTube Utilities ---

def get_youtube_stream_url(youtube_url):
    """
    Get the direct stream URL for a YouTube video using yt-dlp.
    Retries up to 3 times if it fails.
    """
    import subprocess

    command = [
        "yt-dlp",
        "-g",
        "-f", "best[ext=mp4]/best",
        youtube_url
    ]

    for attempt in range(3):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            stream_url = result.stdout.strip()
            if stream_url:
                return stream_url
        except subprocess.CalledProcessError as e:
            logger.warning("YouTube stream URL attempt %d failed: %s", attempt + 1, e)
            time.sleep(1)
            if attempt == 2:
                logger.error("Failed to get YouTube stream URL after 3 attempts.")
                return None
    return None


def download_youtube_video(youtube_url, output_dir):
    """
    Download a YouTube video using yt-dlp to a specified directory.
    Returns the path to the downloaded file.
    """
    import subprocess

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    command = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", output_template,
        "--no-playlist",
        youtube_url
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        downloaded_path = None
        for line in process.stdout:
            logger.debug("yt-dlp: %s", line.strip())
            if "Destination:" in line and "Merger" not in line:
                parts = line.split("Destination:")
                if len(parts) > 1:
                    downloaded_path = parts[1].strip()
            elif "Merging formats into" in line:
                parts = line.split("Merging formats into")
                if len(parts) > 1:
                    path = parts[1].strip()
                    if path.startswith('"') and path.endswith('"'):
                        path = path[1:-1]
                    downloaded_path = path
            elif "has already been downloaded" in line:
                parts = line.split("] ")
                if len(parts) > 1:
                    path = parts[1].split(" has already")[0].strip()
                    downloaded_path = path

        process.wait()

        if process.returncode == 0 and downloaded_path and os.path.exists(downloaded_path):
            return downloaded_path
        elif process.returncode == 0:
            try:
                cmd_filename = [
                    "yt-dlp",
                    "--get-filename",
                    "-o", output_template,
                    "--no-playlist",
                    youtube_url
                ]
                res = subprocess.run(cmd_filename, capture_output=True, text=True, check=True)
                clean_path = res.stdout.strip()
                if os.path.exists(clean_path):
                    return clean_path
            except Exception as e:
                logger.error("Error getting filename from yt-dlp: %s", e)
                return None

        return None

    except Exception as e:
        logger.error("Error downloading YouTube video: %s", e)
        return None
