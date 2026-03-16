"""
Spatial Risk Engine

Transforms raw YOLO detections into actionable intelligence by evaluating
every tracked object against a set of geometric rules:

  - RestrictedZone   : polygon — emits WARNING/CRITICAL on dwell threshold
  - Tripwire         : line segment — emits events on directional crossing

Usage (from worker.py):
    from spatial_engine import SpatialEngine
    engine = SpatialEngine(zones_config, tripwires_config)
    alerts = engine.evaluate(frame_id, detections, fps)
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import time
import math
import logging
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("spatial_engine")


# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class Alert:
    type: str          # "zone_entry" | "loitering" | "tripwire_crossing"
    severity: str      # "INFO" | "WARNING" | "CRITICAL"
    message: str
    zone_name: str
    track_id: int
    class_name: str
    frame_id: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type":       self.type,
            "severity":   self.severity,
            "message":    self.message,
            "zone_name":  self.zone_name,
            "track_id":   self.track_id,
            "class_name": self.class_name,
            "frame_id":   self.frame_id,
            "timestamp":  self.timestamp,
        }


# ── Geometry helpers ──────────────────────────────────────────────────────

def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """
    Returns True if `point` is inside `polygon`.
    Polygon is an (N,2) float32 numpy array of (x, y) vertices.
    Uses OpenCV's optimised pointPolygonTest.
    """
    px, py = float(point[0]), float(point[1])
    result = cv2.pointPolygonTest(polygon, (px, py), measureDist=False)
    return result >= 0  # 0 = on boundary, +1 = inside, -1 = outside


def side_of_line(p: Tuple[float, float],
                 a: Tuple[float, float],
                 b: Tuple[float, float]) -> float:
    """
    Returns the signed area / cross product of (b-a, p-a).
    Positive → p is to the left of AB, Negative → right.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_cross(p1, p2, p3, p4) -> bool:
    """Do line segments P1-P2 and P3-P4 intersect?"""
    d1 = side_of_line(p3, p4, p1)
    d2 = side_of_line(p3, p4, p2)
    d3 = side_of_line(p1, p2, p3)
    d4 = side_of_line(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


# ── Zone ─────────────────────────────────────────────────────────────────

class RestrictedZone:
    """
    Polygonal zone that tracks dwell time per object.
    Configuration dict keys:
        name            : str
        vertices        : [[x,y], ...]   (pixel coords, 0.0–1.0 normalized OR absolute)
        allowed_classes : [str, ...]     (if None → all classes)
        warning_secs    : float          (dwell before WARNING,  default 5)
        critical_secs   : float          (dwell before CRITICAL, default 15)
    """

    def __init__(self, cfg: dict):
        self.name            = cfg.get("name", "Zone")
        verts                = cfg["vertices"]
        self.polygon         = np.array(verts, dtype=np.float32)
        self.allowed_classes : Optional[List[str]] = cfg.get("allowed_classes")
        self.warning_secs    = float(cfg.get("warning_secs",  5.0))
        self.critical_secs   = float(cfg.get("critical_secs", 15.0))

        # {track_id: {"entry_ts": float, "last_alert_severity": str}}
        self._state: Dict[int, dict] = {}

    def _allowed(self, class_name: str) -> bool:
        if not self.allowed_classes:
            return True
        return class_name in self.allowed_classes

    def evaluate(self, frame_id: int, detections: List[dict]) -> List[Alert]:
        alerts = []
        now = time.time()

        # Which track_ids are currently inside?
        inside_ids = set()
        for det in detections:
            if not self._allowed(det["class_name"]):
                continue
            cx, cy = det["centroid"]
            if point_in_polygon((cx, cy), self.polygon):
                inside_ids.add(det["track_id"])
                det_class = det["class_name"]

                state = self._state.setdefault(det["track_id"], {
                    "entry_ts":          now,
                    "last_alert_severity": None,
                    "class_name":        det_class,
                })

                dwell = now - state["entry_ts"]

                if dwell >= self.critical_secs and state["last_alert_severity"] != "CRITICAL":
                    alerts.append(Alert(
                        type="loitering",
                        severity="CRITICAL",
                        message=f"CRITICAL: Track #{det['track_id']} ({det_class}) loitering in {self.name} for {dwell:.0f}s",
                        zone_name=self.name,
                        track_id=det["track_id"],
                        class_name=det_class,
                        frame_id=frame_id,
                    ))
                    state["last_alert_severity"] = "CRITICAL"

                elif dwell >= self.warning_secs and state["last_alert_severity"] not in ("WARNING", "CRITICAL"):
                    alerts.append(Alert(
                        type="loitering",
                        severity="WARNING",
                        message=f"WARNING: Track #{det['track_id']} ({det_class}) entered {self.name} — {dwell:.0f}s",
                        zone_name=self.name,
                        track_id=det["track_id"],
                        class_name=det_class,
                        frame_id=frame_id,
                    ))
                    state["last_alert_severity"] = "WARNING"

                elif dwell < self.warning_secs and state["last_alert_severity"] is None:
                    # First-frame entry alert (INFO)
                    alerts.append(Alert(
                        type="zone_entry",
                        severity="INFO",
                        message=f"INFO: Track #{det['track_id']} ({det_class}) entered {self.name}",
                        zone_name=self.name,
                        track_id=det["track_id"],
                        class_name=det_class,
                        frame_id=frame_id,
                    ))
                    state["last_alert_severity"] = "INFO"

        # Clear state for objects that left the zone
        for tid in list(self._state.keys()):
            if tid not in inside_ids:
                self._state.pop(tid, None)

        return alerts


# ── Tripwire ──────────────────────────────────────────────────────────────

class Tripwire:
    """
    Directional line segment that counts crossings.
    Configuration dict keys:
        name            : str
        point1          : [x, y]
        point2          : [x, y]
        in_direction    : "left" | "right"   (relative to P1→P2 vector)
        allowed_classes : [str, ...]
        cooldown_secs   : float  (seconds before the same track can re-trigger, default 2)
    """

    def __init__(self, cfg: dict):
        self.name            = cfg.get("name", "Tripwire")
        self.p1              = tuple(cfg["point1"])
        self.p2              = tuple(cfg["point2"])
        self.in_direction    = cfg.get("in_direction", "left")
        self.allowed_classes : Optional[List[str]] = cfg.get("allowed_classes")
        self.cooldown_secs   = float(cfg.get("cooldown_secs", 2.0))

        self.count_in  = 0
        self.count_out = 0

        # {track_id: last_centroid}  — used to compute the movement vector
        self._prev_centroids: Dict[int, Tuple[float, float]] = {}
        # {track_id: last_cross_ts}
        self._cooldowns: Dict[int, float] = {}

    def _allowed(self, class_name: str) -> bool:
        if not self.allowed_classes:
            return True
        return class_name in self.allowed_classes

    def evaluate(self, frame_id: int, detections: List[dict]) -> List[Alert]:
        alerts = []
        now = time.time()

        current_ids = set()
        for det in detections:
            tid = det["track_id"]
            class_name = det["class_name"]
            current_ids.add(tid)

            if not self._allowed(class_name):
                continue

            cx, cy = det["centroid"]
            curr = (cx, cy)
            prev = self._prev_centroids.get(tid)
            self._prev_centroids[tid] = curr

            if prev is None:
                continue

            # Cooldown check
            last_cross = self._cooldowns.get(tid, 0)
            if now - last_cross < self.cooldown_secs:
                continue

            # Did the movement vector P_prev → P_curr cross the tripwire?
            if not segments_cross(prev, curr, self.p1, self.p2):
                continue

            # Determine direction using the sign of the cross product
            cross = side_of_line(curr, self.p1, self.p2)
            if self.in_direction == "left":
                direction = "IN" if cross > 0 else "OUT"
            else:
                direction = "IN" if cross < 0 else "OUT"

            if direction == "IN":
                self.count_in += 1
            else:
                self.count_out += 1

            self._cooldowns[tid] = now

            alerts.append(Alert(
                type="tripwire_crossing",
                severity="INFO",
                message=(
                    f"INFO: {class_name} #{tid} crossed {self.name} [{direction}]"
                    f" | In:{self.count_in} Out:{self.count_out}"
                ),
                zone_name=self.name,
                track_id=tid,
                class_name=class_name,
                frame_id=frame_id,
            ))

        # Prune stale tracks
        for tid in list(self._prev_centroids.keys()):
            if tid not in current_ids:
                self._prev_centroids.pop(tid, None)

        return alerts


# ── Spatial Engine (orchestrator) ─────────────────────────────────────────

class SpatialEngine:
    """
    Holds all zones and tripwires for one session and evaluates them
    against a detection list on every frame.

    Args:
        zones_cfg     : list of zone config dicts  (see RestrictedZone)
        tripwires_cfg : list of tripwire config dicts (see Tripwire)
    """

    def __init__(self, zones_cfg: List[dict], tripwires_cfg: List[dict]):
        self.zones     = [RestrictedZone(c) for c in zones_cfg]
        self.tripwires = [Tripwire(c)       for c in tripwires_cfg]

        logger.info(
            "SpatialEngine initialised: %d zone(s), %d tripwire(s)",
            len(self.zones), len(self.tripwires)
        )

    def evaluate(self, frame_id: int, detections: List[dict]) -> List[Alert]:
        """
        Evaluate all spatial rules against the current frame's detections.
        Returns a (possibly empty) list of Alert objects.
        """
        if not detections:
            return []

        alerts: List[Alert] = []
        for zone in self.zones:
            alerts.extend(zone.evaluate(frame_id, detections))
        for wire in self.tripwires:
            alerts.extend(wire.evaluate(frame_id, detections))

        if alerts:
            logger.debug("Frame %d: %d spatial alert(s)", frame_id, len(alerts))

        return alerts
