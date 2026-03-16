"""
Inference Worker Configuration

All settings are sourced from environment variables or a .env file.
Copy .env.example to .env and adjust values for your setup.
"""

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
    The gateway subscribes to channels named: detections:{session_id} and frames:{session_id}.
    """

    # --- Output ---
    publish_frames: bool = True
    """
    If True, compress each frame to JPEG and publish to Redis frames channel.
    Disable if bandwidth between worker and Redis is limited; the gateway will
    then relay inference results only (no MJPEG stream).
    """

    jpeg_quality: int = 70
    """JPEG compression quality for frames published to Redis (1–95)."""

    target_fps: int = 30
    """Maximum frames per second to process. Worker sleeps to honour this."""

    # --- Logging ---
    log_level: str = "INFO"


settings = WorkerConfig()
