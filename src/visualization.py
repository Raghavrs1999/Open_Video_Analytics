"""
Visualization Module

Modern object visualization styles for video inference:
- Corner brackets (replaces traditional bounding boxes)
- Centroid dots
- Dwell time labels
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import cv2
import logging
import numpy as np
import random
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


def draw_corner_brackets(frame: np.ndarray,
                         bbox: Tuple[int, int, int, int],
                         color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 2,
                         corner_length: int = 20) -> None:
    """Draw corner brackets around an object instead of a full bounding box."""
    x1, y1, x2, y2 = map(int, bbox)

    box_width = x2 - x1
    box_height = y2 - y1
    corner_length = min(corner_length, box_width // 3, box_height // 3)

    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)

    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)

    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)


def draw_centroid_dot(frame: np.ndarray,
                      centroid: Tuple[float, float],
                      color: Tuple[int, int, int] = (0, 255, 255),
                      radius: int = 6,
                      filled: bool = True) -> None:
    """Draw a dot at the object's centroid."""
    cx, cy = int(centroid[0]), int(centroid[1])
    thickness = -1 if filled else 2
    cv2.circle(frame, (cx, cy), radius, color, thickness)


def draw_translucent_box(frame: np.ndarray,
                         bbox: Tuple[int, int, int, int],
                         color: Tuple[int, int, int],
                         alpha: float = 0.2) -> None:
    """Draw a translucent filled box (ROI-only blending)."""
    x1, y1, x2, y2 = map(int, bbox)

    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    colored_rect = np.full(roi.shape, color, dtype=np.uint8)
    cv2.addWeighted(colored_rect, alpha, roi, 1 - alpha, 0, roi)


def draw_label(frame: np.ndarray,
               position: Tuple[int, int],
               text: str,
               color: Tuple[int, int, int] = (0, 255, 0),
               font_scale: float = 0.4,
               thickness: int = 1,
               background: bool = True) -> None:
    """Draw a refined text label with semi-transparent background."""
    x, y = int(position[0]), int(position[1])

    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    if background:
        h, w = frame.shape[:2]
        rx1 = max(0, x - 2)
        ry1 = max(0, y - text_h - 4)
        rx2 = min(w, x + text_w + 2)
        ry2 = min(h, y + baseline)
        if rx2 > rx1 and ry2 > ry1:
            roi = frame[ry1:ry2, rx1:rx2]
            black = np.zeros_like(roi)
            cv2.addWeighted(black, 0.6, roi, 0.4, 0, roi)

    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


def draw_dwell_time_label(frame: np.ndarray,
                          centroid: Tuple[float, float],
                          dwell_seconds: float,
                          threshold_seconds: float = 30.0,
                          font_scale: float = 0.6) -> None:
    """Draw dwell time label near centroid with color indicating threshold breach."""
    cx, cy = int(centroid[0]), int(centroid[1])

    if dwell_seconds > threshold_seconds:
        color = (0, 0, 255)  # Red
        text = f"{dwell_seconds:.1f}s!"
    else:
        color = (0, 255, 0)  # Green
        text = f"{dwell_seconds:.1f}s"

    label_y = cy - 15
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    label_x = cx - text_w // 2

    cv2.rectangle(frame, (label_x - 3, label_y - text_h - 3),
                  (label_x + text_w + 3, label_y + baseline + 3), (0, 0, 0), -1)
    cv2.putText(frame, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)


def draw_detection(frame: np.ndarray,
                   bbox: Tuple[int, int, int, int],
                   label: str = "",
                   color: Tuple[int, int, int] = (255, 0, 0),
                   style: str = "corner_brackets",
                   show_label: bool = False) -> Tuple[float, float]:
    """Draw a detection with HUD style: Translucent Fill + Corner Brackets + Refined Label."""
    x1, y1, x2, y2 = map(int, bbox)
    centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

    draw_translucent_box(frame, (x1, y1, x2, y2), color, alpha=0.15)

    if style == "corner_brackets":
        draw_corner_brackets(frame, (x1, y1, x2, y2), color, thickness=2, corner_length=15)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if show_label and label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text = label.upper()
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        label_y = y1 - 6
        if label_y - text_h - 4 < 0:
            label_y = y1 + text_h + 6

        cv2.rectangle(frame,
                      (x1, label_y - text_h - 4),
                      (x1 + text_w + 8, label_y + baseline + 2),
                      (0, 0, 0), -1)
        cv2.putText(frame, text, (x1 + 4, label_y), font, font_scale, (255, 255, 255), thickness)

    return centroid


# Color palette — Vivid colors for detection overlays
CLASS_COLORS = [
    (0, 255, 0),       # Lime Green
    (255, 255, 0),     # Cyan
    (255, 0, 0),       # Blue
    (0, 165, 255),     # Orange
    (0, 0, 255),       # Red
    (255, 0, 255),     # Magenta
    (0, 215, 255),     # Gold
    (128, 0, 128),     # Purple
    (200, 200, 0),     # Dark Cyan / Teal
    (0, 100, 0),       # Dark Green
    (128, 0, 0),       # Navy Blue
    (0, 69, 255),      # Orange Red
    (75, 0, 130),      # Indigo
    (47, 79, 47),      # Dark Slate Grey
    (0, 128, 128),     # Olive
    (180, 105, 255),   # Hot Pink
]

PASTEL_COLORS = [
    (152, 251, 152), (224, 255, 255), (173, 216, 230), (255, 204, 153),
    (255, 153, 153), (255, 153, 255), (255, 236, 139), (221, 160, 221),
    (175, 238, 238), (144, 238, 144), (176, 196, 222), (255, 160, 122),
    (230, 230, 250), (189, 183, 107), (240, 128, 128), (255, 182, 193),
]


def get_class_color(class_id: int, use_pastel: bool = False) -> Tuple[int, int, int]:
    """Get a consistent color for a class ID."""
    palette = PASTEL_COLORS if use_pastel else CLASS_COLORS
    if class_id < len(palette):
        return palette[class_id]

    rng = random.Random(class_id)
    if use_pastel:
        return (rng.randint(150, 255), rng.randint(150, 255), rng.randint(150, 255))
    else:
        return (rng.randint(40, 255), rng.randint(40, 255), rng.randint(40, 255))
