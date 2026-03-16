"""
RtspWriter — Push annotated frames to MediaMTX via ffmpeg stdin pipe.

The worker creates one RtspWriter per session.
Frames are written as raw BGR bytes via stdin to an ffmpeg process
that re-encodes them as H.264 and publishes RTSP to MediaMTX.

No GStreamer required — only ffmpeg in PATH (included in the worker Dockerfile).
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import logging
import subprocess
import shutil
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger("rtsp_writer")


class RtspWriter:
    """
    Opens an ffmpeg subprocess and pushes BGR frames to an RTSP URL.

    Args:
        rtsp_url  : Full RTSP URL to push to, e.g. rtsp://localhost:8554/session1
        width     : Frame width  (pixels)
        height    : Frame height (pixels)
        fps       : Target frame rate for the RTSP stream
        crf       : H.264 quality (0=lossless, 51=worst). Default 23 is broadcast quality.
        preset    : ffmpeg H.264 preset. 'ultrafast' for <5ms encoding latency.
        ffmpeg_bin: Path to ffmpeg binary (default: auto-detect in PATH)
    """

    def __init__(
        self,
        rtsp_url:   str,
        width:      int,
        height:     int,
        fps:        int   = 25,
        crf:        int   = 23,
        preset:     str   = "ultrafast",
        ffmpeg_bin: str   = "",
    ):
        self.rtsp_url   = rtsp_url
        self.width      = width
        self.height     = height
        self.fps        = fps
        self._proc: Optional[subprocess.Popen] = None
        self._lock      = threading.Lock()
        self._dropped   = 0
        self._written   = 0

        # Locate ffmpeg
        ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"

        cmd = [
            ffmpeg,
            "-loglevel",  "warning",

            # ── Input: raw BGR frames from stdin ──────────────────────────
            "-f",         "rawvideo",
            "-vcodec",    "rawvideo",
            "-pix_fmt",   "bgr24",
            "-s",         f"{width}x{height}",
            "-r",         str(fps),
            "-i",         "pipe:0",          # read from stdin

            # ── Encoding ─────────────────────────────────────────────────
            "-vcodec",    "libx264",
            "-pix_fmt",   "yuv420p",
            "-preset",    preset,
            "-tune",      "zerolatency",      # minimise encode latency
            "-crf",       str(crf),
            "-g",         str(fps * 2),       # keyframe every 2 seconds

            # ── Output: RTSP ───────────────────────────────────────────────
            "-f",         "rtsp",
            "-rtsp_transport", "tcp",
            rtsp_url,
        ]

        logger.info("Starting ffmpeg RTSP writer → %s", rtsp_url)
        logger.debug("ffmpeg command: %s", " ".join(cmd))

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            # Drain stderr in a background thread to prevent pipe blockage
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, daemon=True
            )
            self._stderr_thread.start()
            logger.info("ffmpeg RTSP writer started (PID %d)", self._proc.pid)
        except FileNotFoundError:
            logger.error(
                "ffmpeg not found. Install it or set FFMPEG_PATH env var. "
                "RTSP push is disabled."
            )
            self._proc = None
        except Exception as exc:
            logger.error("Failed to start ffmpeg: %s", exc)
            self._proc = None

    # ── Internal ─────────────────────────────────────────────────────────

    def _drain_stderr(self):
        """Read and log ffmpeg stderr so the pipe never blocks."""
        if not self._proc:
            return
        for line in iter(self._proc.stderr.readline, b""):
            line = line.decode("utf-8", errors="replace").strip()
            if line:
                logger.debug("[ffmpeg] %s", line)

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def is_alive(self) -> bool:
        """Returns True if the ffmpeg process is running."""
        return self._proc is not None and self._proc.poll() is None

    def write(self, frame: np.ndarray) -> bool:
        """
        Write one BGR frame to ffmpeg stdin.

        Returns True on success, False if the frame was dropped.
        Thread-safe (multiple callers are safe but single-writer is faster).
        """
        if not self.is_alive:
            self._dropped += 1
            return False

        # Resize if necessary (frame dimensions must match what ffmpeg expects)
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            import cv2
            frame = cv2.resize(frame, (self.width, self.height))

        try:
            with self._lock:
                self._proc.stdin.write(frame.tobytes())
                self._proc.stdin.flush()
            self._written += 1
            return True
        except BrokenPipeError:
            logger.warning("ffmpeg stdin pipe broken — RTSP writer is dead.")
            self._proc = None
            self._dropped += 1
            return False
        except Exception as exc:
            logger.warning("RTSP write error: %s", exc)
            self._dropped += 1
            return False

    def stats(self) -> dict:
        return {
            "rtsp_url":   self.rtsp_url,
            "alive":      self.is_alive,
            "written":    self._written,
            "dropped":    self._dropped,
        }

    def close(self):
        """Gracefully stop the ffmpeg process."""
        if self._proc and self._proc.poll() is None:
            logger.info("Closing ffmpeg RTSP writer …")
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            logger.info(
                "ffmpeg stopped. Frames written: %d  dropped: %d",
                self._written, self._dropped,
            )
        self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
