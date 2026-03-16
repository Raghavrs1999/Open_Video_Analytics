"""
Backend Client Module

Manages connections to distributed YOLO inference backend servers.
Handles session initialization, health monitoring, frame tracking,
and graceful error recovery.
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import uuid
import asyncio
import cv2
import httpx
import logging
import numpy as np
import streamlit as st
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class BackendClientManager:
    def __init__(self, backend_configs: List[dict]):
        self.backends = backend_configs
        self.base_session_id = str(uuid.uuid4())
        self.active_backends = set(range(len(backend_configs)))
        self.degraded_backends = {}
        self.client = httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_connections=20))
        self.health_status = {}
        self.backend_classes = {}

    async def initialize_sessions(self, quiet=False):
        tasks = []
        for idx, backend in enumerate(self.backends):
            tasks.append(self._init_backend(idx, backend))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Backend %d (%s) init failed: %s", idx, self.backends[idx]['url'], result)
                st.warning(f"Backend {idx} ({self.backends[idx]['url']}) init failed: {result}. Removing from pool.")
                self.active_backends.discard(idx)
                self.health_status[idx] = {"status": "offline", "error": str(result)}
            else:
                if not quiet:
                    st.sidebar.success(f"Backend connected: {self.backends[idx]['name']}")
                self.health_status[idx] = {"status": "online", "sessions": 1}
                self.backend_classes[idx] = result.get('model_classes', {})

        if not self.active_backends:
            logger.error("All selected backends failed to initialize.")
            st.error("CRITICAL: All selected backends failed to initialize.")

    async def _init_backend(self, idx: int, backend: dict):
        url = f"{backend['url']}/session/init"
        payload = {"session_id": f"{self.base_session_id}_{idx}", "model_name": backend.get('model', 'yolov8n')}
        resp = await self.client.post(url, json=payload, timeout=180.0)
        resp.raise_for_status()
        return resp.json()

    def get_all_classes(self):
        """Aggregate unique classes from all active backends."""
        options = set()
        for idx in self.active_backends:
            classes = self.backend_classes.get(idx, {})
            for k, v in classes.items():
                try:
                    namespaced_key = int(k) + (idx * 1000)
                    options.add(f"{namespaced_key}: {v} (Backend {idx})")
                except ValueError:
                    options.add(f"{k}: {v}")

        def sort_key(x):
            try:
                return int(x.split(":")[0])
            except:
                return 0
        return sorted(list(options), key=sort_key)

    async def check_all_health(self):
        """Query /health endpoint for all backends."""
        tasks = []
        for idx, backend in enumerate(self.backends):
            url = f"{backend['url']}/health"
            tasks.append(self.client.get(url, timeout=5.0))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                self.health_status[idx] = {"status": "offline", "error": str(result)}
            else:
                try:
                    data = result.json()
                    self.health_status[idx] = {
                        "status": "online",
                        "gpu_mem": data.get('gpu_stats', {}).get('memory_allocated', 0) / 1024**2,
                        "sessions": data.get('active_sessions', 0),
                        "uptime": data.get('uptime_seconds', 0)
                    }
                except:
                    self.health_status[idx] = {"status": "error", "error": "Invalid response"}

    async def track_frame(self, frame: np.ndarray, frame_id: int):
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()

        tasks = []
        active_indices = list(self.active_backends)

        for idx in active_indices:
            tasks.append(self._track_single(idx, frame_bytes, frame_id))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for i, response in enumerate(responses):
            idx = active_indices[i]
            if isinstance(response, Exception):
                self._handle_error(idx, response)
                results[idx] = None
                self.health_status[idx] = {"status": "offline", "error": str(response)}
            else:
                results[idx] = response
                self.degraded_backends.pop(idx, None)

        return results

    async def _track_single(self, idx: int, frame_bytes: bytes, frame_id: int):
        backend = self.backends[idx]
        url = f"{backend['url']}/session/{self.base_session_id}_{idx}/track"

        files = {'file': ('frame.jpg', frame_bytes, 'image/jpeg')}
        data = {'frame_id': str(frame_id)}

        resp = await self.client.post(url, data=data, files=files)
        resp.raise_for_status()
        return resp.json()

    def _handle_error(self, idx: int, error: Exception):
        count = self.degraded_backends.get(idx, 0) + 1
        self.degraded_backends[idx] = count
        if count >= 3:
            logger.warning("Backend %d failed %d times: %s. Removing from pool.", idx, count, error)
            self.active_backends.discard(idx)

    async def cleanup(self):
        tasks = []
        for idx in range(len(self.backends)):
            try:
                backend = self.backends[idx]
                url = f"{backend['url']}/session/{self.base_session_id}_{idx}"
                tasks.append(self.client.delete(url))
            except:
                pass

        await asyncio.gather(*tasks, return_exceptions=True)
        await self.client.aclose()
