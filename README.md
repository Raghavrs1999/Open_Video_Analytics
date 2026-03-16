<div align="center">

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=13&duration=2500&pause=800&color=00D4FF&center=true&vCenter=true&multiline=true&width=700&height=80&lines=Real-time+AI+video+analytics+pipeline;YOLO+%E2%80%A2+Redis+%E2%80%A2+WebSocket+%E2%80%A2+React;Spatial+risk+intelligence+%E2%80%94+frame+zero+latency" alt="Typing SVG" />

<br/><br/>

# Open Video Analytics

**Enterprise-grade, decoupled video inference pipeline for real-time AI surveillance.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket_Gateway-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Redis](https://img.shields.io/badge/Redis-Pub%2FSub_Broker-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![React](https://img.shields.io/badge/React-Vite_%2B_Tailwind-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://vitejs.dev)
[![YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11%2Fv12-purple?style=for-the-badge&logo=ultralytics&logoColor=white)](https://docs.ultralytics.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com/compose)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-39_passed-22c55e?style=for-the-badge&logo=pytest&logoColor=white)](#testing)

<br/>

> **Purpose-built for production scenarios:** RTSP cameras, file-based analysis, edge deployments, and multi-stream command centers — all through a single, event-driven pipeline.

<br/>

[**Live Demo**](#quickstart) · [**Architecture**](#architecture) · [**Features**](#features) · [**Configuration**](#configuration) · [**Contributing**](#contributing)

</div>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Open Video Analytics v2                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   📷  Camera · RTSP · File · Webcam                                  │
│          │                                                            │
│          ▼                                                            │
│   ┌──────────────────┐                                               │
│   │  Inference Worker │  Python · YOLO · OpenCV · CUDA               │
│   │  (GPU process)    │  • model.track() on every frame              │
│   │                   │  • Spatial Risk Engine (zones + tripwires)   │
│   └────────┬──────────┘  • Publishes detections, frames, alerts      │
│            │                                                          │
│            │  Redis Pub/Sub                                           │
│            │  ├── detections:{session_id}  ← JSON bboxes             │
│            │  ├── frames:{session_id}      ← JPEG bytes              │
│            │  ├── alerts:{session_id}      ← structured risk events  │
│            │  └── telemetry:system         ← CPU / RAM / GPU stats   │
│            ▼                                                          │
│   ┌──────────────────┐                                               │
│   │  FastAPI Gateway  │  Python · asyncio · uvicorn                  │
│   │  (always-on)      │  • Fans detections + alerts to WebSocket     │
│   │                   │  • MJPEG frame relay                         │
│   │                   │  • Telemetry broadcast every 1s              │
│   │                   │  • Manages worker subprocesses               │
│   └────────┬──────────┘                                              │
│            │                                                          │
│            │  WebSocket  ws://gateway:8000/ws/{session_id}           │
│            │  MJPEG      http://gateway:8000/video/{id}/stream       │
│            ▼                                                          │
│   ┌──────────────────┐                                               │
│   │  React UI         │  Vite · Tailwind · Zustand                   │
│   │  Command Center   │  • <canvas> bbox overlay                     │
│   │                   │  • Live alert feed (CRITICAL / WARNING)      │
│   │                   │  • Multi-session CameraGrid                  │
│   │                   │  • System telemetry strip (CPU/GPU/RAM)      │
│   └──────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features

### 🔬 AI Inference
| Feature | Details |
|---|---|
| **YOLO Tracking** | Persistent track IDs across frames via `model.track()` |
| **Multi-Model** | YOLOv8 / v9 / v10 / v11 / v12, ONNX, OBB (oriented bounding boxes) |
| **GPU Decoding** | torchvision JPEG decode directly on CUDA, CPU fallback |
| **Auto Download** | Ultralytics model auto-download on first run |

### 🧠 Spatial Risk Engine
| Feature | Details |
|---|---|
| **Restricted Zones** | Polygon-based intrusion detection via `cv2.pointPolygonTest` |
| **Loitering Detection** | Per-track dwell timer → `INFO` → `WARNING` → `CRITICAL` escalation |
| **Directional Tripwires** | Line-crossing counter with IN / OUT direction, configurable cooldown |
| **Structured Alerts** | `{"type": "alert", "severity": "CRITICAL", "message": "..."}` over WebSocket |

### ⚡ Real-Time Pipeline
| Feature | Details |
|---|---|
| **Redis Pub/Sub** | Sub-millisecond message delivery between worker and gateway |
| **WebSocket Fan-out** | Gateway multicasts frames to all browser clients simultaneously |
| **MJPEG Relay** | Low-friction video stream without a separate media server |
| **Telemetry** | Live CPU / RAM / GPU utilisation broadcast every 1 second |

### 🖥️ Command-Center UI
| Feature | Details |
|---|---|
| **Dark Cyber Theme** | `#090d17` background, cyan accents, JetBrains Mono typography |
| **Canvas BBox Overlay** | Sub-pixel accurate bounding boxes with corner accents + centroid dots |
| **Alert Feed** | CRITICAL alerts pulse red, WARNING flash yellow, zone name labels |
| **CameraGrid** | Adaptive N-column layout for monitoring multiple sessions |
| **System Telemetry Strip** | Animated CPU / RAM / GPU / VRAM bars, turns red above 80% |

### 🏗️ Infrastructure
| Feature | Details |
|---|---|
| **Docker Compose** | One command spins up Redis + Gateway + Worker |
| **GPU Support** | Uncomment 4 lines in `docker-compose.yml` to enable NVIDIA runtime |
| **Env-Driven Config** | All tunable parameters via `.env` or environment variables |
| **Legacy Mode** | Original Streamlit + HTTP inference untouched — both modes coexist |

---

## Quickstart

### Prerequisites
- **Docker + Docker Compose** (for Redis and gateway)  
- **Python 3.10+** with CUDA (for the inference worker)  
- **Node.js 16+** (for the React frontend)

### 1 · Clone

```bash
git clone https://github.com/Raghavrs1999/Open_Video_Analytics.git
cd Open_Video_Analytics
```

### 2 · Configure

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # Linux / macOS
```

Edit `.env` — at minimum set your video source:

```dotenv
VIDEO_SOURCE=rtsp://user:pass@192.168.1.64/stream
# or a file: VIDEO_SOURCE=D:/videos/sample.mp4
# or webcam: VIDEO_SOURCE=0
MODEL_NAME=yolov8n
```

### 3 · Start Redis + Gateway

```bash
docker compose up redis gateway
```

### 4 · Start Inference Worker *(GPU machine)*

```bash
cd inference_worker
pip install -r requirements.txt
python worker.py
```

### 5 · Start the UI

```bash
cd frontend
npm install
npm run dev
# → Open http://localhost:5173
```

In the dashboard: set your Session ID, click **▶ Start Session**, and the live feed appears.

> **No Docker?** `docker run -d -p 6379:6379 redis:7-alpine`, then run gateway with `uvicorn main:app --port 8000` from the `gateway/` directory.

---

## Spatial Risk Engine

Define zones and tripwires in your `.env` file as JSON strings:

```dotenv
# Restricted Zone — triggers WARNING after 5s, CRITICAL after 15s
ZONES_CONFIG=[
  {
    "name": "Restricted Zone A",
    "vertices": [[100,200],[500,200],[500,480],[100,480]],
    "allowed_classes": ["person"],
    "warning_secs": 5,
    "critical_secs": 15
  }
]

# Directional Tripwire — counts people entering vs. exiting
TRIPWIRES_CONFIG=[
  {
    "name": "Entry Gate",
    "point1": [640, 0],
    "point2": [640, 720],
    "in_direction": "left",
    "allowed_classes": ["person"],
    "cooldown_secs": 2
  }
]
```

Alerts are published instantly over the WebSocket with this payload:

```json
{
  "type": "alert",
  "data": {
    "severity": "WARNING",
    "message": "Track #7 (person) loitering in Restricted Zone A for 6s",
    "zone_name": "Restricted Zone A",
    "track_id": 7,
    "class_name": "person",
    "frame_id": 432,
    "timestamp": 1710619532.8
  }
}
```

---

## Configuration

All settings are controlled via environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `VIDEO_SOURCE` | `0` | RTSP URL, file path, or webcam index |
| `MODEL_NAME` | `yolov8n` | YOLO model filename (without `.pt`) |
| `SESSION_ID` | `default` | Unique session identifier |
| `CONFIDENCE` | `0.25` | Detection confidence threshold |
| `TARGET_FPS` | `30` | Max inference FPS |
| `PUBLISH_FRAMES` | `true` | Enable MJPEG relay |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `ZONES_CONFIG` | `[]` | JSON array of zone definitions |
| `TRIPWIRES_CONFIG` | `[]` | JSON array of tripwire definitions |
| `CORS_ORIGINS` | `*` | Gateway CORS allowed origins |
| `VITE_GATEWAY_URL` | `http://localhost:8000` | Frontend gateway URL |

---

## Project Structure

```
Open_Video_Analytics/
│
├── inference_worker/          # Standalone GPU inference process
│   ├── worker.py              # Main loop: capture → YOLO → Redis publish
│   ├── spatial_engine.py      # RestrictedZone + Tripwire rules engine
│   ├── config.py              # Pydantic Settings (all env vars)
│   ├── Dockerfile
│   └── requirements.txt
│
├── gateway/                   # FastAPI WebSocket gateway
│   ├── main.py                # Endpoints: /ws, /video, /session, /telemetry
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                  # React command-center UI
│   └── src/
│       ├── store.js           # Zustand state (WS, detections, alerts, telemetry)
│       ├── App.jsx            # 3-column layout + telemetry strip
│       └── components/
│           ├── VideoPlayer.jsx          # MJPEG stream + stats bar
│           ├── DetectionCanvas.jsx      # Canvas bbox overlay (letterbox-aware)
│           ├── ControlPanel.jsx         # Session config + start/stop
│           ├── AlertFeed.jsx            # Severity-coded live event feed
│           ├── CameraGrid.jsx           # Multi-session adaptive grid
│           └── SystemTelemetryStrip.jsx # CPU / RAM / GPU bars
│
├── docker-compose.yml         # Redis + Gateway + Worker services
├── .env.example               # Full environment variable reference
│
└── ── Legacy Streamlit (v1 — preserved) ──────────────────────────
    ├── app.py                 # streamlit run app.py
    ├── worker/server.py       # FastAPI HTTP GPU inference server
    ├── src/                   # Analytics engine (line/ROI/dwell/heatmap)
    └── config.yaml            # Backend registry
```

---

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Redis + session health check |
| `GET` | `/sessions` | List active worker sessions |
| `GET` | `/models` | List available `.pt` model files |
| `POST` | `/session/start` | Start inference worker subprocess |
| `POST` | `/session/stop/{id}` | Terminate inference worker |
| `WS` | `/ws/{session_id}` | Live detection + alert WebSocket |
| `WS` | `/ws/telemetry` | System telemetry WebSocket |
| `GET` | `/telemetry/system` | One-shot CPU/RAM/GPU snapshot |
| `GET` | `/video/{id}/stream` | MJPEG video relay |

---

## Testing

```bash
# Run the full test suite (39 tests)
pip install pytest
pytest tests/ -v
```

| Test File | Coverage |
|---|---|
| `test_line_counter.py` | 17 tests — crossing, cooldown, filtering |
| `test_sync_buffer.py` | 17 tests — ordering, namespacing, playback |
| `test_roi_congestion.py` | 5 tests — congestion levels, dwell, ID recovery |

---

## GPU Setup (Docker)

Uncomment the `deploy` block in `docker-compose.yml`:

```yaml
# worker service
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## Contributing

Contributions are welcome and appreciated.

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes with a conventional commit message
4. Open a Pull Request

Please open an issue first for large changes so we can align on direction.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with precision by [Raghav Sharma](https://github.com/Raghavrs1999)**

<br/>

*If this project is useful to you, please consider giving it a ⭐ — it helps others discover it.*

<br/>

[![Star History](https://img.shields.io/github/stars/Raghavrs1999/Open_Video_Analytics?style=social)](https://github.com/Raghavrs1999/Open_Video_Analytics/stargazers)

</div>
