"""
ROI Congestion Module

Measures congestion within polygonal regions of interest and
optionally tracks per-object dwell time with ID recovery.
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import cv2
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class ROICongestion:
    """
    Measure congestion within a polygonal region of interest.

    Counts objects whose centroids fall within the polygon.
    Levels: Low, Medium, High based on configurable thresholds.
    """

    COLORS = {
        "Low": (0, 255, 0),      # Green
        "Medium": (0, 255, 255), # Yellow
        "High": (0, 0, 255)      # Red
    }

    def __init__(self,
                 roi_id: int,
                 vertices: List[Tuple[int, int]],
                 low_threshold: int = 5,
                 high_threshold: int = 15,
                 name: str = None,
                 dwell_threshold: float = 30.0,
                 enable_dwell: bool = False,
                 allowed_classes: List[int] = None):
        self.roi_id = roi_id
        self.vertices = vertices
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.name = name or f"ROI {roi_id + 1}"
        self.dwell_threshold = dwell_threshold
        self.enable_dwell = enable_dwell
        self.allowed_classes = allowed_classes

        self.polygon = np.array(vertices, dtype=np.int32)

        self.current_count = 0
        self.current_level = "Low"

        # Dwell time tracking
        self.dwell_entry_frames = {}
        self.current_dwell_times = {}
        self._last_positions = {}
        self.track_classes = {}

    def is_point_inside(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside the ROI polygon."""
        result = cv2.pointPolygonTest(self.polygon, point, False)
        return result >= 0

    def count_objects(self, centroids: List[Tuple[float, float]]) -> int:
        """Count how many object centroids are inside the ROI, filtering by class."""
        count = 0
        for data in centroids:
            if isinstance(data, (list, tuple)) and len(data) == 2 and isinstance(data[0], (list, tuple)):
                point, class_id = data
            else:
                point = data
                class_id = None

            if self.is_point_inside(point):
                if self.allowed_classes is None or class_id is None or class_id in self.allowed_classes:
                    count += 1

        self.current_count = count
        self.current_level = self.get_congestion_level(count)
        return count

    def get_congestion_level(self, count: int) -> str:
        """Determine congestion level based on object count."""
        if count < self.low_threshold:
            return "Low"
        elif count > self.high_threshold:
            return "High"
        else:
            return "Medium"

    def get_color(self) -> Tuple[int, int, int]:
        """Get the BGR color for current congestion level."""
        return self.COLORS.get(self.current_level, (128, 128, 128))

    def update_dwell_times(self, track_centroids: Dict[int, Tuple[Tuple[float, float], int]],
                           current_frame: int, fps: float) -> Tuple[Dict[int, float], List[Dict]]:
        """Update dwell times for tracked objects in ROI with ID recovery."""
        if not self.enable_dwell:
            return {}, []

        completed_events = []
        current_in_roi = set()
        ID_RECOVERY_THRESHOLD = 80

        for track_id, data in track_centroids.items():
            if isinstance(data, (list, tuple)) and len(data) == 2 and isinstance(data[0], (list, tuple)):
                centroid, class_id = data
            else:
                centroid = data
                class_id = None

            if self.allowed_classes is not None and class_id is not None and class_id not in self.allowed_classes:
                continue

            if self.is_point_inside(centroid):
                current_in_roi.add(track_id)

                if track_id not in self.dwell_entry_frames:
                    # ID recovery: match new ID to nearby old position
                    min_dist = float('inf')
                    best_match_id = None
                    best_match_entry_frame = None

                    for old_id, old_entry_frame in self.dwell_entry_frames.items():
                        if old_id in self._last_positions:
                            old_pos = self._last_positions[old_id]
                            dist = ((centroid[0] - old_pos[0])**2 + (centroid[1] - old_pos[1])**2) ** 0.5

                            if dist < ID_RECOVERY_THRESHOLD:
                                old_class = self.track_classes.get(old_id)
                                if old_class is not None and class_id is not None and old_class != class_id:
                                    continue
                                if dist < min_dist:
                                    min_dist = dist
                                    best_match_id = old_id
                                    best_match_entry_frame = old_entry_frame

                    if best_match_id is not None:
                        # Transfer dwell time from old ID to new ID
                        self.dwell_entry_frames[track_id] = best_match_entry_frame
                        self.dwell_entry_frames.pop(best_match_id, None)
                        self.current_dwell_times.pop(best_match_id, None)
                        self._last_positions.pop(best_match_id, None)
                        if best_match_id in self.track_classes:
                            self.track_classes[track_id] = self.track_classes.pop(best_match_id)
                        logger.debug("ID recovery: %d -> %d in %s", best_match_id, track_id, self.name)
                    else:
                        self.dwell_entry_frames[track_id] = current_frame
                        if class_id is not None:
                            self.track_classes[track_id] = class_id

                self._last_positions[track_id] = centroid
                if class_id is not None:
                    self.track_classes[track_id] = class_id

                frames_in_roi = current_frame - self.dwell_entry_frames[track_id]
                dwell_seconds = frames_in_roi / fps if fps > 0 else 0
                self.current_dwell_times[track_id] = dwell_seconds

        # Remove objects that left ROI
        left_roi = set(self.dwell_entry_frames.keys()) - current_in_roi
        for track_id in left_roi:
            if track_id not in track_centroids:
                if track_id in self.current_dwell_times:
                    dwell_duration = self.current_dwell_times[track_id]
                    if dwell_duration > 1.0:
                        completed_events.append({
                            "track_id": track_id,
                            "roi_name": self.name,
                            "duration": dwell_duration,
                            "entry_frame": self.dwell_entry_frames[track_id],
                            "exit_frame": current_frame
                        })

                self.dwell_entry_frames.pop(track_id, None)
                self.current_dwell_times.pop(track_id, None)
                self._last_positions.pop(track_id, None)
                self.track_classes.pop(track_id, None)

        return self.current_dwell_times, completed_events

    def get_objects_exceeding_threshold(self) -> List[int]:
        """Get track IDs that exceed the dwell threshold."""
        return [tid for tid, dwell in self.current_dwell_times.items()
                if dwell > self.dwell_threshold]

    def draw_on_frame(self, frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Draw the ROI polygon on a frame with congestion color overlay."""
        color = self.get_color()

        poly_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(poly_mask, [self.polygon], 255)
        color_layer = np.full(frame.shape, color, dtype=np.uint8)
        mask_bool = poly_mask > 0
        frame[mask_bool] = cv2.addWeighted(frame, 1 - alpha, color_layer, alpha, 0)[mask_bool]

        cv2.polylines(frame, [self.polygon], True, color, 2)

        M = cv2.moments(self.polygon)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = self.vertices[0]

        label = f"{self.name}: {self.current_count} ({self.current_level})"
        font_scale = 1.2
        thickness = 3

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = cx - text_w // 2
        text_y = cy + text_h // 2

        padding = 10
        cv2.rectangle(frame,
                      (text_x - padding, text_y - text_h - padding),
                      (text_x + text_w + padding, text_y + baseline + padding),
                      (0, 0, 0), -1)
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        return frame

    def get_stats(self) -> Dict:
        """Get current statistics for this ROI."""
        return {
            "roi_id": self.roi_id,
            "name": self.name,
            "count": self.current_count,
            "level": self.current_level,
            "vertices": self.vertices
        }
