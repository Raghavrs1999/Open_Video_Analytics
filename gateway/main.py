"""
Open Video Analytics — FastAPI WebSocket Gateway

Acts as the single connection point for the React frontend:
  - Manages inference worker subprocesses (start / stop)
  - Subscribes to Redis Pub/Sub and fans detections out via WebSocket
  - Relays MJPEG video stream from Redis to the browser (multipart)
  - Exposes REST endpoints for session management and health

Endpoints:
    GET  /health                     → server + Redis health
    GET  /sessions                   → list active sessions
    GET  /models                     → list available model files
    POST /session/start              → start inference worker process
    POST /session/stop/{session_id}  → stop inference worker process
    WS   /ws/{session_id}            → WebSocket detection feed
    GET  /video/{session_id}/stream  → MJPEG frame stream

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

__author__ = "Raghav Sharma"
__version__ = "2.0.0"

import os
import sys
import json
import time
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import contextlib


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class GatewayConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    redis_url: str = "redis://localhost:6379"
    worker_script: str = "../inference_worker/worker.py"
    model_dir: str = "../models"
    log_level: str = "INFO"
    cors_origins: str = "*"


cfg = GatewayConfig()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gateway")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

# session_id → set of WebSocket connections
ws_clients: Dict[str, Set[WebSocket]] = {}

# session_id → asyncio.Task (Redis subscriber: detections + alerts)
subscriber_tasks: Dict[str, asyncio.Task] = {}

# session_id → subprocess.Popen (worker process)
worker_processes: Dict[str, subprocess.Popen] = {}

redis_client: Optional[aioredis.Redis] = None

# Background telemetry task
_telemetry_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, _telemetry_task
    logger.info("Connecting to Redis: %s", cfg.redis_url)
    redis_client = aioredis.from_url(cfg.redis_url, decode_responses=False)
    try:
        await redis_client.ping()
        logger.info("Redis connection OK")
    except Exception as exc:
        logger.error("Redis connection failed: %s  — continuing without Redis.", exc)

    # Start system telemetry broadcast
    _telemetry_task = asyncio.create_task(_telemetry_loop())

    yield

    # Shutdown
    logger.info("Shutting down gateway …")
    if _telemetry_task:
        _telemetry_task.cancel()
    for task in subscriber_tasks.values():
        task.cancel()
    for proc in worker_processes.values():
        try:
            proc.terminate()
        except Exception:
            pass
    if redis_client:
        await redis_client.aclose()
    logger.info("Gateway stopped.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Open Video Analytics Gateway",
    version="2.0.0",
    description="WebSocket gateway that bridges inference workers and the React frontend.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    session_id: str = "default"
    video_source: str
    model_name: str = "yolov8n"
    confidence: float = 0.25
    target_fps: int = 30
    publish_frames: bool = True
    zones_config: List[Any] = []
    tripwires_config: List[Any] = []


# ---------------------------------------------------------------------------
# Helper: Redis subscriber loop
# ---------------------------------------------------------------------------

async def _redis_subscriber(session_id: str):
    """
    Subscribe to Redis detections:{session_id} AND alerts:{session_id}.
    Pushes detection payloads and structured alert events to WebSocket clients.
    """
    det_channel   = f"detections:{session_id}"
    alert_channel = f"alerts:{session_id}"
    logger.info("Starting Redis subscriber for: %s, %s", det_channel, alert_channel)

    async def _broadcast(data: bytes):
        clients = ws_clients.get(session_id, set()).copy()
        dead: Set[WebSocket] = set()
        for ws in clients:
            try:
                await ws.send_text(data.decode("utf-8") if isinstance(data, bytes) else data)
            except Exception:
                dead.add(ws)
        ws_clients[session_id] -= dead

    try:
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(det_channel, alert_channel)

        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            raw  = message["data"]
            chan = message["channel"]
            if isinstance(chan, bytes):
                chan = chan.decode()

            if chan == alert_channel:
                # Wrap alert with envelope so frontend can distinguish
                try:
                    alert_dict = json.loads(raw)
                    envelope   = json.dumps({"type": "alert", "data": alert_dict})
                    await _broadcast(envelope.encode())
                except Exception:
                    pass
            else:
                await _broadcast(raw)

    except asyncio.CancelledError:
        logger.info("Subscriber task cancelled for session: %s", session_id)
    except Exception as exc:
        logger.error("Subscriber error for session %s: %s", session_id, exc)
    finally:
        try:
            await pubsub.unsubscribe(det_channel, alert_channel)
            await pubsub.aclose()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# System Telemetry
# ---------------------------------------------------------------------------

def _get_system_stats() -> dict:
    stats: dict = {"timestamp": time.time()}
    if HAS_PSUTIL:
        stats["cpu_percent"]  = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        stats["ram_used_gb"]  = round(mem.used  / 1024**3, 2)
        stats["ram_total_gb"] = round(mem.total / 1024**3, 2)
        stats["ram_percent"]  = mem.percent
    if HAS_PYNVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_i  = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats["gpu_name"]       = pynvml.nvmlDeviceGetName(handle)
            stats["gpu_percent"]    = util.gpu
            stats["gpu_mem_used_gb"]  = round(mem_i.used  / 1024**3, 2)
            stats["gpu_mem_total_gb"] = round(mem_i.total / 1024**3, 2)
        except Exception:
            pass
    return stats


async def _telemetry_loop():
    """Broadcast system telemetry to Redis telemetry:system every second."""
    channel = "telemetry:system"
    while True:
        try:
            await asyncio.sleep(1)
            if not redis_client:
                continue
            stats = _get_system_stats()
            await redis_client.publish(channel, json.dumps({"type": "telemetry", "data": stats}))
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.debug("Telemetry publish error: %s", exc)


@app.get("/telemetry/system", tags=["Meta"])
async def get_telemetry():
    """One-shot system stats snapshot (CPU, RAM, GPU)."""
    return _get_system_stats()


# ---------------------------------------------------------------------------
# WebSocket — Telemetry (optional dedicated channel)
# ---------------------------------------------------------------------------

@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    """Dedicated WebSocket for system telemetry — subscribes to Redis telemetry:system."""
    await websocket.accept()
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("telemetry:system")
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = message["data"]
            await websocket.send_text(data.decode() if isinstance(data, bytes) else data)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        await pubsub.unsubscribe("telemetry:system")
        await pubsub.aclose()



@app.get("/health", tags=["Meta"])
async def health():
    redis_ok = False
    try:
        if redis_client:
            await redis_client.ping()
            redis_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis": "ok" if redis_ok else "unreachable",
        "active_sessions": list(worker_processes.keys()),
        "ws_connections": {sid: len(conns) for sid, conns in ws_clients.items()},
    }


@app.get("/sessions", tags=["Sessions"])
async def list_sessions():
    sessions = []
    for session_id, proc in worker_processes.items():
        retcode = proc.poll()
        status = "running" if retcode is None else f"stopped ({retcode})"

        # Try to fetch metadata from Redis
        meta = {}
        try:
            raw = await redis_client.get(f"meta:{session_id}")
            if raw:
                meta = json.loads(raw)
        except Exception:
            pass

        sessions.append({"session_id": session_id, "process_status": status, "meta": meta})

    return {"sessions": sessions}


@app.get("/models", tags=["Meta"])
async def list_models():
    model_path = Path(cfg.model_dir)
    if not model_path.exists():
        return {"models": []}
    models = [p.stem for p in model_path.glob("*.pt")]
    return {"models": sorted(models)}


# ---------------------------------------------------------------------------
# Endpoints — Session Management
# ---------------------------------------------------------------------------

@app.post("/session/start", tags=["Sessions"])
async def start_session(req: StartSessionRequest, background_tasks: BackgroundTasks):
    if req.session_id in worker_processes:
        proc = worker_processes[req.session_id]
        if proc.poll() is None:
            raise HTTPException(status_code=409, detail=f"Session '{req.session_id}' is already running")
        else:
            del worker_processes[req.session_id]

    worker_script = Path(cfg.worker_script).resolve()
    if not worker_script.exists():
        raise HTTPException(status_code=500, detail=f"Worker script not found: {worker_script}")

    env = {
        **os.environ,
        "SESSION_ID":        req.session_id,
        "VIDEO_SOURCE":      req.video_source,
        "MODEL_NAME":        req.model_name,
        "CONFIDENCE":        str(req.confidence),
        "TARGET_FPS":        str(req.target_fps),
        "PUBLISH_FRAMES":    str(req.publish_frames).lower(),
        "REDIS_URL":         cfg.redis_url,
        "MODEL_DIR":         str(Path(cfg.model_dir).resolve()),
        "ZONES_CONFIG":      json.dumps(req.zones_config),
        "TRIPWIRES_CONFIG":  json.dumps(req.tripwires_config),
    }

    proc = subprocess.Popen(
        [sys.executable, str(worker_script)],
        env=env,
        cwd=str(worker_script.parent),
    )
    worker_processes[req.session_id] = proc
    logger.info("Started worker process PID %d for session '%s'", proc.pid, req.session_id)

    # Start Redis subscriber if not already running
    if req.session_id not in subscriber_tasks or subscriber_tasks[req.session_id].done():
        ws_clients.setdefault(req.session_id, set())
        task = asyncio.create_task(_redis_subscriber(req.session_id))
        subscriber_tasks[req.session_id] = task

    return {"status": "started", "session_id": req.session_id, "pid": proc.pid}


@app.post("/session/stop/{session_id}", tags=["Sessions"])
async def stop_session(session_id: str):
    if session_id not in worker_processes:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    proc = worker_processes.pop(session_id)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    # Cancel subscriber
    task = subscriber_tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()

    logger.info("Stopped session '%s'", session_id)
    return {"status": "stopped", "session_id": session_id}


# ---------------------------------------------------------------------------
# Endpoint — WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info("WebSocket connected: session=%s  client=%s", session_id, websocket.client)

    ws_clients.setdefault(session_id, set()).add(websocket)

    # If no subscriber is running for this session yet, start one
    if session_id not in subscriber_tasks or subscriber_tasks[session_id].done():
        task = asyncio.create_task(_redis_subscriber(session_id))
        subscriber_tasks[session_id] = task

    # Send initial handshake
    await websocket.send_json({
        "type":       "connected",
        "session_id": session_id,
        "timestamp":  time.time(),
    })

    # Try to send session metadata immediately if available
    try:
        raw = await redis_client.get(f"meta:{session_id}")
        if raw:
            meta = json.loads(raw)
            await websocket.send_json({"type": "meta", "data": meta})
    except Exception:
        pass

    try:
        # Keep connection alive; handle client-sent messages (e.g. ping)
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if msg == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat", "timestamp": time.time()})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WebSocket error session=%s: %s", session_id, exc)
    finally:
        ws_clients.get(session_id, set()).discard(websocket)
        logger.info("WebSocket disconnected: session=%s", session_id)


# ---------------------------------------------------------------------------
# Endpoint — MJPEG frame stream
# ---------------------------------------------------------------------------

@app.get("/video/{session_id}/stream", tags=["Video"])
async def mjpeg_stream(session_id: str):
    """
    Server-Sent MJPEG stream.
    Subscribes to Redis frames:{session_id} and streams JPEG frames
    as multipart/x-mixed-replace to the browser.
    """
    frame_channel = f"frames:{session_id}"

    async def generate():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(frame_channel)
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                frame_bytes = message["data"]
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(frame_channel)
            await pubsub.aclose()

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
