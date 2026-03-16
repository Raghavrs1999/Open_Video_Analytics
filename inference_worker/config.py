"""
Inference Worker Configuration

All settings are sourced from environment variables or a .env file.
Copy .env.example to .env and adjust values for your setup.
"""

import json
from typing import Any, List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Video Source ---
    video_source: str = "0"
    """
    RTSP URL, video file path, or webcam index.
    Examples:
      rtsp://user:pass@192.168.1.64/stream
      D:/videos/sample.mp4
      0  (default webcam)
    """

    # --- Model ---
    model_name: str = "yolov8n"
    """
    YOLO model name (without .pt) or full path to a .pt file.
    Must be present in the models/ directory or be a known ultralytics model
    that will be auto-downloaded on first run.
    """

    model_dir: str = "../models"
    """Directory to search for .pt model files."""

    confidence: float = 0.25
    """Minimum detection confidence threshold (0.0–1.0)."""

    iou: float = 0.45
    """IoU threshold for NMS."""

    imgsz: int = 640
    """Inference image size (square)."""

    device: str = ""
    """
    Inference device. Leave empty for automatic selection (CUDA if available, else CPU).
    Examples: 'cuda:0', 'cpu', '0'
    """

    # --- Redis ---
    redis_url: str = "redis://localhost:6379"
    """Redis connection URL."""

    # --- Session ---
    session_id: str = "default"
    """
    Unique identifier for this worker session.
    The gateway subscribes to channels: detections:{session_id}, frames:{session_id},
    and alerts:{session_id}.
    """

    # --- Output ---
    publish_frames: bool = True
    """
    If True, compress each frame to JPEG and publish to Redis frames channel.
    Disable if bandwidth between worker and Redis is limited.
    """

    jpeg_quality: int = 70
    """JPEG compression quality for frames published to Redis (1–95)."""

    target_fps: int = 30
    """Maximum frames per second to process. Worker sleeps to honour this."""

    # --- Logging ---
    log_level: str = "INFO"

    # ── Spatial Risk Engine ───────────────────────────────────────────────
    zones_config: List[Any] = []
    """
    List of restricted zone definitions (JSON).
    Set via ZONES_CONFIG env var as a JSON string, e.g.:
    [
      {
        "name": "Restricted Zone A",
        "vertices": [[100,200],[400,200],[400,500],[100,500]],
        "allowed_classes": ["person"],
        "warning_secs": 5,
        "critical_secs": 15
      }
    ]
    Leave empty [] to disable zone detection.
    """

    tripwires_config: List[Any] = []
    """
    List of tripwire definitions (JSON).
    Set via TRIPWIRES_CONFIG env var as a JSON string, e.g.:
    [
      {
        "name": "Entry Gate",
        "point1": [320, 0],
        "point2": [320, 720],
        "in_direction": "left",
        "allowed_classes": ["person", "car"],
        "cooldown_secs": 2
      }
    ]
    Leave empty [] to disable tripwire detection.
    """

    @field_validator("zones_config", "tripwires_config", mode="before")
    @classmethod
    def parse_json_list(cls, v: Any) -> List[Any]:
        """Allow the field to be set as a JSON string (from env var)."""
        if isinstance(v, str):
            return json.loads(v)
        return v


settings = WorkerConfig()
