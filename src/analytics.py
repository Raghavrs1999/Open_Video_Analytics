"""
Analytics Module

Applies analytics (line counting, ROI congestion, dwell time, heatmap)
to video frames and renders visualizations in a layered architecture.
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import cv2
import logging
import numpy as np
from datetime import datetime
from .visualization import draw_detection, draw_centroid_dot, get_class_color, draw_dwell_time_label

logger = logging.getLogger(__name__)


def apply_analytics(frame, results_dict, line_counters, roi_regions, dwell_zones,
                    heatmap_config, heatmap_state=None, frame_count=0, fps=30.0,
                    track_history=None, stats=None, global_class_filter=None,
                    show_labels=False, detection_only=False):
    """
    Apply analytics visualization on the frame based on aggregated results.
    Separates Logic from Visualization with proper Layering.
    """
    # --- 1. DATA PREPARATION & LOGIC UPDATE PHASE ---

    # Flatten results from all backends
    all_detections = []
    for backend_idx, detections in results_dict.items():
        if detections:
            all_detections.extend(detections)

    # Extract Centroids
    track_centroids = {}
    frame_centroids_list = []
    for det in all_detections:
        centroid = (det['centroid'][0], det['centroid'][1])
        class_id = det['class_id']
        track_id = det['track_id']
        track_centroids[track_id] = (centroid, class_id)
        frame_centroids_list.append((centroid, class_id))

    # Update Line Counters (Logic)
    if line_counters and track_history is not None:
        for track_id, (centroid, class_id) in track_centroids.items():
            prev_centroid = track_history.get(track_id)
            if prev_centroid:
                for lc in line_counters:
                    direction = lc.check_crossing(
                        track_id,
                        prev_centroid,
                        centroid,
                        frame_count=frame_count,
                        class_id=class_id
                    )

                    if direction and stats is not None:
                        current_time_str = datetime.now().strftime("%H:%M:%S")
                        counts = lc.get_counts()
                        class_name = "Unknown"
                        for d in all_detections:
                            if d['track_id'] == track_id:
                                class_name = d['class_name']
                                break

                        stats["event_log"].append({
                            "Time": current_time_str,
                            "Type": "Line Crossing",
                            "Location": f"Line {lc.line_id}",
                            "Details": f"{direction}",
                            "Class": class_name,
                            "Track ID": track_id,
                            "Value": 1,
                            "Total In": counts["in"],
                            "Total Out": counts["out"]
                        })

    # Update Track History
    if track_history is not None:
        for track_id, (centroid, _) in track_centroids.items():
            track_history[track_id] = centroid

    # Update Heatmap State (Logic)
    if heatmap_state is not None and heatmap_config.get("enabled", False):
        h, w = frame.shape[:2]
        if heatmap_state.shape == (h, w):
            for (centroid, cls) in frame_centroids_list:
                heatmap_classes = heatmap_config.get("classes")
                if heatmap_classes is not None and cls is not None:
                    if int(cls) not in heatmap_classes:
                        continue

                cx, cy = int(centroid[0]), int(centroid[1])
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(heatmap_state, (cx, cy), 30, 15.0, -1)

            decay_rate = heatmap_config.get("decay", 0.95)
            heatmap_state[:] = heatmap_state * decay_rate

    # --- 2. VISUALIZATION PHASE (Layered) ---

    # Layer 1: Heatmap (Background)
    if heatmap_state is not None and heatmap_config.get("enabled", False):
        h, w = frame.shape[:2]
        if heatmap_state.shape == (h, w):
            heatmap_normalized = np.clip(heatmap_state, 0, 30)
            heatmap_normalized = (heatmap_normalized / 30.0 * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

            opacity = heatmap_config.get("opacity", 0.4)
            alpha_mask = heatmap_normalized.astype(float) / 255.0 * opacity
            alpha_mask_3ch = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=-1)

            frame = (frame * (1 - alpha_mask_3ch) + heatmap_colored * alpha_mask_3ch).astype(np.uint8)

    # Layer 2: Environment (Zones & Lines)

    # Dwell Zones
    if dwell_zones:
        for dwell in dwell_zones:
            dwell_times, completed_events = dwell.update_dwell_times(track_centroids, frame_count, fps)

            poly_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(poly_mask, [dwell.polygon], 255)
            color_layer = np.full(frame.shape, (0, 140, 255), dtype=np.uint8)
            mask_bool = poly_mask > 0
            frame[mask_bool] = cv2.addWeighted(frame, 0.75, color_layer, 0.25, 0)[mask_bool]
            cv2.polylines(frame, [dwell.polygon], True, (0, 165, 255), 3)

            label = dwell.name
            cx = int(np.mean(dwell.polygon[:, 0]))
            cy = int(np.mean(dwell.polygon[:, 1]))
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame, (cx - text_w//2 - 5, cy - text_h - 5), (cx + text_w//2 + 5, cy + baseline + 5), (0, 0, 0), -1)
            cv2.putText(frame, label, (cx - text_w//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

            for track_id, dwell_secs in dwell_times.items():
                data = track_centroids.get(track_id)
                if data:
                    centroid, _ = data
                    dot_color = (0, 0, 255) if dwell_secs > dwell.dwell_threshold else (0, 255, 255)
                    draw_centroid_dot(frame, centroid, color=dot_color, radius=8)
                    draw_dwell_time_label(frame, centroid, dwell_secs, dwell.dwell_threshold)

    # Line Counters
    if line_counters:
        for lc in line_counters:
            p1 = lc.point1
            p2 = lc.point2
            cv2.line(frame, p1, p2, (0, 0, 255), 3)

            cx = int((p1[0] + p2[0]) / 2)
            cy = int((p1[1] + p2[1]) / 2)

            counts = lc.get_counts()
            label = f"In: {counts['in']}  Out: {counts['out']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            pad = 8
            bg_x1 = max(0, cx - text_w // 2 - pad)
            bg_y1 = max(0, cy - text_h - 10 - pad)
            bg_x2 = min(frame.shape[1], cx + text_w // 2 + pad)
            bg_y2 = min(frame.shape[0], cy - 10 + baseline + pad)
            if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
                black = np.zeros_like(roi)
                cv2.addWeighted(black, 0.7, roi, 0.3, 0, roi)

            cv2.putText(frame, label, (cx - text_w // 2, cy - 10), font, font_scale, (255, 255, 255), thickness)

    # ROI Congestion
    if roi_regions:
        centroid_list_formatted = [((d['centroid'][0], d['centroid'][1]), d['class_id']) for d in all_detections]
        for roi in roi_regions:
            roi.count_objects(centroid_list_formatted)
            roi.draw_on_frame(frame)

    # Layer 3: Foreground (Detections)
    visualization_whitelist = set()
    has_active_filters = False

    if global_class_filter is not None:
        visualization_whitelist = set(global_class_filter)
        has_active_filters = True

    for det in all_detections:
        cls = det['class_id']
        if has_active_filters and cls not in visualization_whitelist:
            continue

        x1, y1, x2, y2 = det['bbox']
        tid = det['track_id']
        color = get_class_color(cls)
        label_text = det['class_name'] if detection_only else f"{det['class_name']} {tid}"
        draw_detection(frame, (x1, y1, x2, y2), label=label_text, color=color, show_label=show_labels)

    return frame
