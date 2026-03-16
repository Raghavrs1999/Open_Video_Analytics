"""
Backend Worker — FastAPI GPU Inference Server

Standalone server that loads YOLO models and runs inference with
object tracking. Deployed on GPU machines and accessed via HTTP
by the frontend application.

Usage:
    python worker/server.py

Environment Variables:
    VIDEO_INFERENCE_MODEL_DIR: Path to directory containing .pt model files
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import os
import cv2
import uuid
import time
import torch
import logging
import uvicorn
import asyncio
import datetime
import contextlib
import numpy as np
import torchvision.io
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, List, Optional
import torch.nn.functional as F

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# --- Data Models ---

class SessionInitRequest(BaseModel):
    session_id: str
    model_name: str

class Detection(BaseModel):
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]
    centroid: List[float]

class TrackResponse(BaseModel):
    session_id: str
    frame_id: int
    detections: List[Detection]
    processing_time_ms: float
    timestamp: float


# --- Session Management ---

class SessionData:
    def __init__(self, session_id: str, model_name: str, model: YOLO):
        self.session_id = session_id
        self.model_name = model_name
        self.model = model
        self.created_at = datetime.datetime.now()
        self.last_activity = datetime.datetime.now()
        self.frame_count = 0

        try:
            self.imgsz = model.overrides.get('imgsz', 640)
            logger.info("Model '%s' training imgsz: %s", model_name, self.imgsz)
        except:
            self.imgsz = 640


class SessionManager:
    def __init__(self, max_sessions: int = 5, inactivity_timeout_mins: int = 30):
        self.sessions: Dict[str, SessionData] = {}
        self.max_sessions = max_sessions
        self.inactivity_timeout = datetime.timedelta(minutes=inactivity_timeout_mins)

    def create_session(self, session_id: str, model_name: str, model_path: str) -> bool:
        if session_id in self.sessions:
            return False

        if len(self.sessions) >= self.max_sessions:
            raise HTTPException(status_code=503, detail="Max concurrent sessions reached")

        try:
            model = YOLO(model_path)
            self.sessions[session_id] = SessionData(session_id, model_name, model)
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    def get_session(self, session_id: str) -> Optional[SessionData]:
        session = self.sessions.get(session_id)
        if session:
            session.last_activity = datetime.datetime.now()
        return session

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            del session.model
            del session
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False

    def cleanup_inactive_sessions(self):
        now = datetime.datetime.now()
        expired_ids = [
            sid for sid, session in self.sessions.items()
            if now - session.last_activity > self.inactivity_timeout
        ]
        for sid in expired_ids:
            logger.info("Cleaning up inactive session: %s", sid)
            self.delete_session(sid)


# --- Global State ---

session_manager = SessionManager()
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_DIR = os.environ.get("VIDEO_INFERENCE_MODEL_DIR", DEFAULT_MODEL_DIR)
os.makedirs(MODEL_DIR, exist_ok=True)


async def scheduled_cleanup():
    while True:
        await asyncio.sleep(300)
        session_manager.cleanup_inactive_sessions()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(scheduled_cleanup())
    yield
    logger.info("Shutting down... cleaning up sessions")
    for sid in list(session_manager.sessions.keys()):
        session_manager.delete_session(sid)


app = FastAPI(lifespan=lifespan)


# --- Endpoints ---

@app.post("/session/init")
async def init_session(request: SessionInitRequest):
    model_filename = f"{request.model_name}.pt"
    if request.model_name == "yolov8n":
        model_filename = "yolov8n.pt"
    elif request.model_name == "yolov8m":
        model_filename = "yolov8m.pt"

    model_path = f"{MODEL_DIR}/{model_filename}"

    if not os.path.isfile(model_path):
        logger.info(f"Model file {model_path} not found locally. Ultralytics will attempt to auto-download it.")
        # We don't raise a 404 here anymore. 
        # By passing just the model name without the '.pt' extension or directory to the YOLO constructor later,
        # Ultralytics will automatically download recognized models (like yolov8n.pt or yolo26n.pt) to the local directory.
        model_path = request.model_name # Fallback to standard name to trigger auto-download

    if session_manager.get_session(request.session_id):
        raise HTTPException(status_code=400, detail="Session already exists")

    session_manager.create_session(request.session_id, request.model_name, model_path)

    # Model warmup
    try:
        session = session_manager.get_session(request.session_id)
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        session.model(dummy_frame, verbose=False)
        logger.info("Model '%s' warmed up successfully.", request.model_name)
    except Exception as e:
        logger.warning("Warmup failed for '%s': %s", request.model_name, e)

    return {
        "status": "ready",
        "session_id": request.session_id,
        "model_info": {"name": request.model_name, "path": model_path},
        "model_classes": session_manager.sessions[request.session_id].model.names
    }


@app.post("/session/{session_id}/track")
async def track_frame(
    session_id: str,
    file: UploadFile = File(...),
    frame_id: int = Form(...)
):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    start_time = time.time()

    try:
        file_bytes = await file.read()

        frame_input = None
        offset_x, offset_y = 0.0, 0.0
        scale_factor = 1.0
        orig_h, orig_w = 0, 0

        # GPU-accelerated decoding
        if torch.cuda.is_available():
            try:
                encoded_tensor = torch.frombuffer(file_bytes, dtype=torch.uint8)
                frame_tensor = torchvision.io.decode_jpeg(encoded_tensor, device='cuda')
                _, orig_h, orig_w = frame_tensor.shape

                frame_input = frame_tensor.unsqueeze(0).float() / 255.0

                # Letterbox Resize
                target_size = (640, 640)
                r = min(target_size[0] / orig_h, target_size[1] / orig_w)

                if r != 1:
                    new_unpad = (int(round(orig_w * r)), int(round(orig_h * r)))
                    frame_input = F.interpolate(frame_input, size=(new_unpad[1], new_unpad[0]), mode='bilinear', align_corners=False)

                dw, dh = target_size[1] - int(round(orig_w * r)), target_size[0] - int(round(orig_h * r))
                dw /= 2
                dh /= 2

                top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

                frame_input = F.pad(frame_input, (left, right, top, bottom), value=0.447)

                offset_x = left
                offset_y = top
                scale_factor = r

            except Exception as e:
                logger.debug("GPU decode failed (likely non-JPEG), falling back to CPU: %s", e)

        # CPU fallback
        if frame_input is None:
            np_arr = np.frombuffer(file_bytes, np.uint8)
            frame_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame_input is not None:
                orig_h, orig_w = frame_input.shape[:2]

        if frame_input is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Run inference
        results = session.model.track(frame_input, persist=True, verbose=False, imgsz=session.imgsz)

        detections = []
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    track_id = int(box.id.item()) if box.id is not None else -1
                    if track_id == -1:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    x1 = max(0.0, (x1 - offset_x) / scale_factor)
                    y1 = max(0.0, (y1 - offset_y) / scale_factor)
                    x2 = min(float(orig_w), (x2 - offset_x) / scale_factor)
                    y2 = min(float(orig_h), (y2 - offset_y) / scale_factor)

                    conf = float(box.conf[0].item())
                    cls = int(box.cls[0].item())
                    class_name = session.model.names[cls]

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    detections.append(Detection(
                        track_id=track_id, class_id=cls, class_name=class_name,
                        confidence=conf, bbox=[x1, y1, x2, y2], centroid=[cx, cy]
                    ))
            elif getattr(r, 'obb', None) is not None:
                # OBB (Oriented Bounding Box) support
                for obb in r.obb:
                    track_id = int(obb.id.item()) if obb.id is not None else -1
                    if track_id == -1:
                        continue

                    pts = obb.xyxyxyxy[0].cpu().numpy()
                    x_coords = pts[:, 0]
                    y_coords = pts[:, 1]

                    x1, x2 = float(np.min(x_coords)), float(np.max(x_coords))
                    y1, y2 = float(np.min(y_coords)), float(np.max(y_coords))

                    x1 = max(0.0, (x1 - offset_x) / scale_factor)
                    y1 = max(0.0, (y1 - offset_y) / scale_factor)
                    x2 = min(float(orig_w), (x2 - offset_x) / scale_factor)
                    y2 = min(float(orig_h), (y2 - offset_y) / scale_factor)

                    conf = float(obb.conf[0].item())
                    cls = int(obb.cls[0].item())
                    class_name = session.model.names[cls]

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    detections.append(Detection(
                        track_id=track_id, class_id=cls, class_name=class_name,
                        confidence=conf, bbox=[x1, y1, x2, y2], centroid=[cx, cy]
                    ))

        session.frame_count += 1
        processing_time = (time.time() - start_time) * 1000

        return TrackResponse(
            session_id=session_id, frame_id=frame_id,
            detections=detections, processing_time_ms=processing_time,
            timestamp=datetime.datetime.now().timestamp()
        )

    except Exception as e:
        logger.error("Tracking error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    frames_proc = session.frame_count
    session_manager.delete_session(session_id)

    return {"status": "deleted", "session_id": session_id, "frames_processed": frames_proc}


@app.get("/health")
async def health_check():
    gpu_stats = {}
    if torch.cuda.is_available():
        gpu_stats = {
            "gpu_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_reserved": torch.cuda.memory_reserved(0)
        }

    return {
        "status": "healthy",
        "active_sessions": len(session_manager.sessions),
        "gpu_stats": gpu_stats,
        "max_sessions": session_manager.max_sessions
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
