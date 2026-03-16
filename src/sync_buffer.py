"""
Sync Buffer Module

Frame synchronization buffer for the distributed video inference pipeline.
Manages frame ordering, result aggregation, and playback buffering to ensure
analytics receive frames in strict sequential order.
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import time
import logging
import collections
import numpy as np

logger = logging.getLogger(__name__)


class SyncBuffer:
    def __init__(self, num_backends: int, buffer_size: int = 300):
        self.num_backends = num_backends
        self.buffer = collections.deque(maxlen=buffer_size)
        self.pending = {}
        self.next_frame_to_play = 0
        self.total_processed = 0
        self.dropped_frame_count = 0

    def add_frame(self, frame_id: int, frame: np.ndarray):
        if frame_id not in self.pending:
            self.pending[frame_id] = {
                'frame_id': frame_id,
                'frame': frame,
                'results': {},
                'timestamp': time.time(),
                'complete': False
            }

    def add_results(self, frame_id: int, backend_idx: int, detections: list):
        if frame_id not in self.pending:
            return

        # ID Namespace
        namespaced_detections = []
        for det in detections:
            d = det.copy()
            d['track_id'] = (backend_idx * 1_000_000) + d['track_id']
            if 'class_id' in d:
                d['class_id'] = d['class_id'] + (backend_idx * 1000)
            d['source_model_idx'] = backend_idx
            namespaced_detections.append(d)

        self.pending[frame_id]['results'][backend_idx] = namespaced_detections

    def mark_complete(self, frame_id: int):
        if frame_id in self.pending:
            self.pending[frame_id]['complete'] = True

    def try_move_to_buffer(self):
        """Move consecutive completed frames from pending to playback buffer."""
        while len(self.buffer) < self.buffer.maxlen and self.next_frame_to_play in self.pending:
            frame_data = self.pending[self.next_frame_to_play]

            if not frame_data['complete']:
                break  # Strict: wait indefinitely for results

            self.buffer.append(frame_data)
            del self.pending[self.next_frame_to_play]
            self.next_frame_to_play += 1
            self.total_processed += 1

    def is_ready_for_playback(self):
        return len(self.buffer) >= self.buffer.maxlen

    def should_rebuffer(self, threshold=30):
        return len(self.buffer) < threshold

    def get_next_frame(self):
        if self.buffer:
            return self.buffer.popleft()
        return None

    def get_status(self):
        return {
            "buffer": len(self.buffer),
            "pending": len(self.pending),
            "next": self.next_frame_to_play,
            "processed": self.total_processed
        }
