"""
Line Counter Module for In/Out Counting

Detects objects crossing a defined line and counts them as "In" or "Out"
using cross-product geometry and segment-segment intersection tests.
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import logging
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)


class LineCounter:
    """
    Count objects crossing a line in a video frame.

    The line is defined by two points. Objects are tracked by their
    centroid position, and a crossing is detected when the centroid
    moves from one side of the line to the other.
    """
    def __init__(self,
                 line_id: int,
                 point1: Tuple[int, int],
                 point2: Tuple[int, int],
                 in_direction: str = "left",
                 allowed_classes: List[int] = None,
                 cooldown_frames: int = 30):
        self.line_id = line_id
        self.point1 = point1
        self.point2 = point2
        self.in_direction = in_direction.lower()
        self.allowed_classes = allowed_classes

        # Counts
        self.in_count = 0
        self.out_count = 0

        # Track when IDs last crossed to allow re-entry after cooldown
        self.last_crossing_frames: Dict[int, int] = {}
        self.RE_ENTRY_COOLDOWN_FRAMES = cooldown_frames

        # Determine which side is "IN" using the line's normal vector
        mid_x = (point1[0] + point2[0]) / 2
        mid_y = (point1[1] + point2[1]) / 2

        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]

        # Normal vector (-dy, dx) — perpendicular, points "left" relative to p1->p2
        norm_x = -dy
        norm_y = dx

        length = (norm_x**2 + norm_y**2)**0.5
        if length > 0:
            norm_x /= length
            norm_y /= length
        else:
            norm_x, norm_y = 1, 0

        offset = 10

        # Map user direction to a target vector
        target_dx, target_dy = 0, 0
        if "left" in self.in_direction: target_dx = -1
        elif "right" in self.in_direction: target_dx = 1
        elif "top" in self.in_direction: target_dy = -1
        elif "bottom" in self.in_direction: target_dy = 1

        dot = (norm_x * target_dx) + (norm_y * target_dy)

        if dot > 0:
            test_point = (mid_x + norm_x * offset, mid_y + norm_y * offset)
        else:
            test_point = (mid_x - norm_x * offset, mid_y - norm_y * offset)

        self.in_side_val = self._get_side(test_point)

    def _get_side(self, point: Tuple[float, float]) -> str:
        """Determine which side of the line a point is on using the cross product."""
        x, y = point
        x1, y1 = self.point1
        x2, y2 = self.point2

        ab_x = x2 - x1
        ab_y = y2 - y1
        ap_x = x - x1
        ap_y = y - y1

        cross_product = ab_x * ap_y - ab_y * ap_x

        if cross_product > 0:
            return "positive"
        elif cross_product < 0:
            return "negative"
        else:
            return "on_line"

    def check_crossing(self,
                       track_id: int,
                       prev_centroid: Tuple[float, float],
                       curr_centroid: Tuple[float, float],
                       frame_count: int,
                       class_id: int = None,
                       allowed_classes: List[int] = None) -> Optional[str]:
        """
        Check if an object has crossed the line.

        Returns:
            "in" if crossed to the "in" side,
            "out" if crossed to the "out" side,
            None if no crossing or within cooldown.
        """
        # Class filtering
        target_classes = allowed_classes if allowed_classes is not None else self.allowed_classes

        if target_classes is not None and class_id is not None:
            if class_id not in target_classes:
                return None

        # Re-entry cooldown
        if track_id in self.last_crossing_frames:
            entry = self.last_crossing_frames[track_id]
            last_frame = entry[0] if isinstance(entry, tuple) else entry
            if (frame_count - last_frame) < self.RE_ENTRY_COOLDOWN_FRAMES:
                return None

        # Segment-segment intersection test
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        def intersect(A, B, C, D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

        A = self.point1
        B = self.point2
        C = prev_centroid
        D = curr_centroid

        if not intersect(A, B, C, D):
            return None

        prev_side = self._get_side(prev_centroid)
        curr_side = self._get_side(curr_centroid)

        if prev_side == curr_side or "on_line" in (prev_side, curr_side):
            return None

        # Spatial deduplication (20px threshold)
        SPATIAL_THRESHOLD = 20
        cx, cy = curr_centroid
        this_direction = "in" if curr_side == self.in_side_val else "out"

        for other_tid, entry in self.last_crossing_frames.items():
            if other_tid == track_id:
                continue
            if not isinstance(entry, tuple) or len(entry) < 3:
                continue
            other_frame, other_pos, other_dir = entry
            if (frame_count - other_frame) < self.RE_ENTRY_COOLDOWN_FRAMES:
                ox, oy = other_pos
                dist = ((cx - ox)**2 + (cy - oy)**2)**0.5
                if dist < SPATIAL_THRESHOLD and this_direction == other_dir:
                    self.last_crossing_frames[track_id] = (frame_count, curr_centroid, this_direction)
                    return None

        # Crossing detected
        self.last_crossing_frames[track_id] = (frame_count, curr_centroid, this_direction)

        if curr_side == self.in_side_val:
            self.in_count += 1
            logger.debug("Track %d crossed IN on line %d (total: %d)", track_id, self.line_id, self.in_count)
            return "in"
        else:
            self.out_count += 1
            logger.debug("Track %d crossed OUT on line %d (total: %d)", track_id, self.line_id, self.out_count)
            return "out"

    def get_counts(self) -> Dict[str, int]:
        """Get the current in/out counts."""
        return {"in": self.in_count, "out": self.out_count}

    def reset(self):
        """Reset all counts and tracking state."""
        self.in_count = 0
        self.out_count = 0
        self.last_crossing_frames.clear()

    def get_line_coords(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return the line coordinates for drawing."""
        return (self.point1, self.point2)
