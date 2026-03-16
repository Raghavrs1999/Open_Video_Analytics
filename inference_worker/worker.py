"""
Open Video Analytics — Standalone Inference Worker

Continuously reads frames from a video source (RTSP / file / webcam),
runs YOLO tracking, and publishes results to Redis:

  Channel  detections:{session_id}  →  JSON detection payload per frame
  Channel  frames:{session_id}      →  JPEG-encoded frame bytes (for MJPEG relay)

Usage:
    python worker.py

Environment Variables (see config.py / .env):
    VIDEO_SOURCE, MODEL_NAME, MODEL_DIR, REDIS_URL, SESSION_ID, ...

Signals:
    SIGINT / SIGTERM gracefully stop the loop.
"""

__author__ = "Raghav Sharma"
__version__ = "2.0.0"

import os
import sys
import cv2
import json
import time
import signal
import logging
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import torchvision.io as tv_io
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from ultralytics import YOLO
import redis

from config import settings
from spatial_engine import SpatialEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference_worker")


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_running = True


def _handle_signal(sig, frame):  # noqa: ANN001
    global _running
    logger.info("Received signal %s — shutting down…", sig)
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_name: str, model_dir: str) -> YOLO:
    """
    Load YOLO model from the models directory.
    Falls back to Ultralytics auto-download if the file is not found locally.
    """
    model_path = Path(model_dir) / f"{model_name}.pt"
    if model_path.exists():
        logger.info("Loading model from: %s", model_path)
        return YOLO(str(model_path))

    # Try bare name → Ultralytics will auto-download standard models
    logger.info("Model file not found locally; attempting auto-download: %s", model_name)
    return YOLO(model_name)


# ---------------------------------------------------------------------------
# GPU-accelerated letterbox (mirrors original worker/server.py logic)
# ---------------------------------------------------------------------------

def gpu_letterbox(frame_bytes: bytes, target: int = 640, device: str = "cuda"):
    """
    Decode JPEG bytes → GPU tensor, letterbox-resize to target×target.
    Returns (tensor, scale, pad_left, pad_top, orig_w, orig_h) or None on failure.
    """
    try:
        encoded = torch.frombuffer(frame_bytes, dtype=torch.uint8)
        frame_t = tv_io.decode_jpeg(encoded, device=device)   # C×H×W, uint8
        _, orig_h, orig_w = frame_t.shape

        inp = frame_t.unsqueeze(0).float() / 255.0
        r = min(target / orig_h, target / orig_w)

        if r != 1.0:
            new_h, new_w = int(round(orig_h * r)), int(round(orig_w * r))
            inp = F.interpolate(inp, size=(new_h, new_w), mode="bilinear", align_corners=False)

        dw = (target - int(round(orig_w * r))) / 2
        dh = (target - int(round(orig_h * r))) / 2
        pad_l, pad_r = int(round(dw - 0.1)), int(round(dw + 0.1))
        pad_t, pad_b = int(round(dh - 0.1)), int(round(dh + 0.1))
        inp = F.pad(inp, (pad_l, pad_r, pad_t, pad_b), value=0.447)

        return inp, r, pad_l, pad_t, orig_w, orig_h
    except Exception as exc:
        logger.debug("GPU letterbox failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Detection → dict
# ---------------------------------------------------------------------------

def _scale_bbox(x1, y1, x2, y2, offset_x, offset_y, scale, orig_w, orig_h):
    x1 = max(0.0, (x1 - offset_x) / scale)
    y1 = max(0.0, (y1 - offset_y) / scale)
    x2 = min(float(orig_w), (x2 - offset_x) / scale)
    y2 = min(float(orig_h), (y2 - offset_y) / scale)
    return x1, y1, x2, y2


def extract_detections(results, model_names, offset_x=0.0, offset_y=0.0,
                        scale=1.0, orig_w=0, orig_h=0):
    """Parse YOLO Results objects into a list of plain dicts."""
    detections = []
    for r in results:
        # --- Standard bounding boxes ---
        if r.boxes and len(r.boxes):
            for box in r.boxes:
                track_id = int(box.id.item()) if box.id is not None else -1
                if track_id == -1:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = _scale_bbox(x1, y1, x2, y2, offset_x, offset_y, scale, orig_w, orig_h)
                conf = float(box.conf[0].item())
                cls  = int(box.cls[0].item())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append({
                    "track_id":   track_id,
                    "class_id":   cls,
                    "class_name": model_names.get(cls, str(cls)),
                    "confidence": round(conf, 4),
                    "bbox":       [round(v, 2) for v in [x1, y1, x2, y2]],
                    "centroid":   [round(cx, 2), round(cy, 2)],
                })

        # --- OBB (Oriented Bounding Box) ---
        elif getattr(r, "obb", None) is not None and len(r.obb):
            for obb in r.obb:
                track_id = int(obb.id.item()) if obb.id is not None else -1
                if track_id == -1:
                    continue
                pts = obb.xyxyxyxy[0].cpu().numpy()
                x1, x2 = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
                y1, y2 = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
                x1, y1, x2, y2 = _scale_bbox(x1, y1, x2, y2, offset_x, offset_y, scale, orig_w, orig_h)
                conf = float(obb.conf[0].item())
                cls  = int(obb.cls[0].item())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append({
                    "track_id":   track_id,
                    "class_id":   cls,
                    "class_name": model_names.get(cls, str(cls)),
                    "confidence": round(conf, 4),
                    "bbox":       [round(v, 2) for v in [x1, y1, x2, y2]],
                    "centroid":   [round(cx, 2), round(cy, 2)],
                })
    return detections


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    global _running

    # --- Connect to Redis ---
    logger.info("Connecting to Redis: %s", settings.redis_url)
    r = redis.from_url(settings.redis_url, decode_responses=False)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as exc:
        logger.error("Cannot connect to Redis: %s", exc)
        sys.exit(1)

    det_channel   = f"detections:{settings.session_id}"
    frame_channel = f"frames:{settings.session_id}"
    alert_channel = f"alerts:{settings.session_id}"
    meta_key      = f"meta:{settings.session_id}"

    # --- Load Model ---
    model = load_model(settings.model_name, settings.model_dir)

    # Resolve device
    device = settings.device
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, verbose=False)
    logger.info("Model '%s' warmed up on %s", settings.model_name, device)

    # --- Spatial Risk Engine ---
    spatial_engine = SpatialEngine(
        zones_cfg=settings.zones_config,
        tripwires_cfg=settings.tripwires_config,
    )
    has_spatial = bool(settings.zones_config or settings.tripwires_config)
    if has_spatial:
        logger.info(
            "Spatial Risk Engine active: %d zone(s), %d tripwire(s)",
            len(settings.zones_config), len(settings.tripwires_config),
        )
    else:
        logger.info("Spatial Risk Engine: no zones/tripwires configured (pass ZONES_CONFIG / TRIPWIRES_CONFIG env vars to enable).")

    # Publish session metadata to Redis
    meta = {
        "session_id":  settings.session_id,
        "model_name":  settings.model_name,
        "model_names": model.names,  # {0: 'person', ...}
        "device":      device,
        "started_at":  datetime.datetime.now().isoformat(),
        "status":      "running",
    }
    r.set(meta_key, json.dumps(meta))

    # --- Open Video Source ---
    source = settings.video_source
    # Treat pure digit strings as integer camera index
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("Failed to open video source: %s", settings.video_source)
        r.set(meta_key, json.dumps({**meta, "status": "error", "error": "failed to open source"}))
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / min(settings.target_fps, src_fps if src_fps > 0 else settings.target_fps)

    logger.info(
        "Source opened: %s | FPS: %.1f | Target: %d fps",
        settings.video_source, src_fps, settings.target_fps
    )

    use_gpu_decode = (
        HAS_TORCHVISION
        and torch.cuda.is_available()
        and "cuda" in device
    )
    logger.info("GPU frame decode: %s", "enabled" if use_gpu_decode else "disabled (CPU)")

    frame_id = 0
    loop_stats = {"frames": 0, "errors": 0, "publish_errors": 0}
    last_stat_log = time.time()

    # -----------------------------------------------------------------------
    # Main capture loop
    # -----------------------------------------------------------------------
    while _running:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str) and not source.startswith("rtsp"):
                # End of file — loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                logger.info("End of video file, looping.")
                continue
            else:
                logger.warning("Failed to read frame from source — retrying in 1 s …")
                time.sleep(1.0)
                continue

        frame_id += 1
        orig_h, orig_w = frame.shape[:2]

        # --- Inference ---
        try:
            offset_x, offset_y, scale = 0.0, 0.0, 1.0
            inp = frame  # default: pass BGR ndarray

            if use_gpu_decode:
                # Encode frame to JPEG → GPU decode → letterbox on GPU
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    result = gpu_letterbox(buf.tobytes(), settings.imgsz, device)
                    if result is not None:
                        inp, scale, offset_x, offset_y, orig_w, orig_h = result

            results = model.track(
                inp,
                persist=True,
                verbose=False,
                imgsz=settings.imgsz,
                conf=settings.confidence,
                iou=settings.iou,
            )

            detections = extract_detections(
                results, model.names,
                offset_x=offset_x,
                offset_y=offset_y,
                scale=scale,
                orig_w=orig_w,
                orig_h=orig_h,
            )

            # --- Spatial Risk Engine ---
            alerts = []
            if has_spatial and detections:
                raw_alerts = spatial_engine.evaluate(frame_id, detections)
                alerts = [a.to_dict() for a in raw_alerts]
                # Publish each alert individually for low-latency UI updates
                for alert_dict in alerts:
                    try:
                        r.publish(alert_channel, json.dumps(alert_dict))
                    except Exception as exc:
                        logger.warning("Failed to publish alert: %s", exc)

        except Exception as exc:
            logger.error("Inference error on frame %d: %s", frame_id, exc)
            loop_stats["errors"] += 1
            time.sleep(0.01)
            continue

        # --- Publish detection payload ---
        payload = {
            "session_id":   settings.session_id,
            "frame_id":     frame_id,
            "timestamp":    time.time(),
            "frame_width":  orig_w,
            "frame_height": orig_h,
            "detections":   detections,
            "alerts":       alerts,   # empty list when no spatial rules fired
        }
        try:
            r.publish(det_channel, json.dumps(payload))
        except Exception as exc:
            logger.warning("Failed to publish detections: %s", exc)
            loop_stats["publish_errors"] += 1

        # --- Publish raw frame (MJPEG relay) ---
        if settings.publish_frames:
            try:
                ok, buf = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
                )
                if ok:
                    r.publish(frame_channel, buf.tobytes())
            except Exception as exc:
                logger.warning("Failed to publish frame: %s", exc)

        loop_stats["frames"] += 1

        # --- Periodic stats log ---
        if time.time() - last_stat_log > 10:
            logger.info(
                "Stats [session=%s]: frames=%d  errors=%d  pub_errors=%d",
                settings.session_id,
                loop_stats["frames"],
                loop_stats["errors"],
                loop_stats["publish_errors"],
            )
            last_stat_log = time.time()

        # --- Frame-rate throttle ---
        elapsed = time.time() - loop_start
        sleep_for = max(0.0, frame_delay - elapsed)
        if sleep_for:
            time.sleep(sleep_for)

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    logger.info("Stopping — releasing resources …")
    cap.release()
    r.set(meta_key, json.dumps({**meta, "status": "stopped"}))
    r.close()
    logger.info("Worker stopped cleanly. Total frames: %d", loop_stats["frames"])


if __name__ == "__main__":
    run()
