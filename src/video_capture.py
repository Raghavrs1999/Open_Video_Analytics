"""
Video Capture Module

Thread-safe video capture with background frame reading.
Supports local files, RTSP streams, HTTP streams, and YouTube URLs.
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import cv2
import time
import queue
import logging
import threading
from .utils import is_url, get_youtube_stream_url

logger = logging.getLogger(__name__)


class VideoCapture:
    def __init__(self, source: str, target_fps: int = 30):
        self.source = source
        self.original_source = source

        # Handle YouTube URLs
        if is_url(str(source)) and ("youtube.com" in str(source) or "youtu.be" in str(source)):
            logger.info("Detected YouTube URL: %s", source)
            stream_url = get_youtube_stream_url(source)
            if stream_url:
                self.source = stream_url
            else:
                logger.warning("Could not get stream URL, will attempt direct open (likely fail)")

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        self.target_fps = target_fps
        self.frame_queue = queue.Queue(maxsize=60)
        self.running = False
        self.thread = None

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = source_fps if source_fps > 0 else target_fps

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Background thread to capture frames."""
        frame_id = 0

        while self.running:
            if self.frame_queue.full():
                time.sleep(0.01)
                continue

            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str) and self.source.startswith(("rtsp://", "http://", "https://")):
                    logger.warning("Stream disconnected, attempting reconnect...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.source)
                    continue
                else:
                    self.running = False
                    break

            try:
                self.frame_queue.put({
                    'frame_id': frame_id,
                    'frame': frame,
                    'timestamp': time.time()
                }, timeout=1.0)
                frame_id += 1
            except queue.Full:
                continue

        self.cap.release()

    def get_frame(self, timeout=0.1):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
