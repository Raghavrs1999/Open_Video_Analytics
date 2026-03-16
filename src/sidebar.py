"""
Sidebar Module

Renders the Streamlit sidebar with all configuration options:
source selection, model config, backend setup, analytics toggles,
recording settings, and interactive drawing controls.
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import os
import uuid
import logging
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Dict, Any

from .utils import load_config
from .backend_client import BackendClientManager

logger = logging.getLogger(__name__)

# --- Model Directory ---
MODEL_DIR = Path(os.environ.get(
    "VIDEO_INFERENCE_MODEL_DIR",
    str(Path(__file__).resolve().parent.parent / "models")
))

# --- Source → Model Mapping ---
SOURCE_MODEL_CONFIG = {
    "CCTV": [
        {"Type": "Fast General Object Detection", "Model": str(MODEL_DIR / "yolo26n.pt"), "model_type": "Detection"},
        {"Type": "General Object Detection", "Model": str(MODEL_DIR / "yolo26x.pt"), "model_type": "Detection"},
    ],
    "GoPro": [
        {"Type": "Action Cam Detection", "Model": str(MODEL_DIR / "yolo26m.pt"), "model_type": "Detection"},
    ],
    "Dashcam": [
        {"Type": "Vehicle & Person Detection", "Model": str(MODEL_DIR / "yolo26s.pt"), "model_type": "Detection"},
    ],
}


def render_download_list(downloads_placeholder):
    """Render completed segment download buttons in the given placeholder."""
    unique_suffix = uuid.uuid4().hex[:8]

    if st.session_state.get('completed_segments'):
        with downloads_placeholder.container():
            st.caption("Completed Segments:")
            for i, seg in enumerate(st.session_state.completed_segments):
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    try:
                        if os.path.exists(seg['video']):
                            with open(seg['video'], "rb") as f:
                                st.download_button(
                                    f"⬇️ Video {seg['time']}", f,
                                    file_name=os.path.basename(seg['video']),
                                    key=f"dl_vid_{os.path.basename(seg['video'])}_{unique_suffix}"
                                )
                        else:
                            st.error(f"Video missing: {os.path.basename(seg['video'])}")
                    except Exception as e:
                        st.error(f"Error loading video: {e}")

                with col_d2:
                    if seg.get('report'):
                        try:
                            if os.path.exists(seg['report']):
                                with open(seg['report'], "rb") as f:
                                    st.download_button(
                                        f"⬇️ Report {seg['time']}", f,
                                        file_name=os.path.basename(seg['report']),
                                        key=f"dl_rpt_{os.path.basename(seg['report'])}_{unique_suffix}"
                                    )
                            else:
                                st.warning("Report missing")
                        except Exception as e:
                            st.warning("Error loading report")
    else:
        downloads_placeholder.empty()


async def render_sidebar() -> Dict[str, Any]:
    """Render the full sidebar configuration and return a config dict."""
    with st.sidebar:
        st.header("Configuration")

        # --- Source Selection ---
        source_type = st.selectbox("Select Source Type", ["CCTV", "Dashcam", "GoPro"])

        # --- Operation Selection ---
        source_configs = SOURCE_MODEL_CONFIG.get(source_type, [])
        display_map = {cfg.get("Type", "Unknown"): cfg for cfg in source_configs}
        display_ops = list(display_map.keys())

        if not display_ops:
            display_ops = ["No Operations Available"]
            display_map["No Operations Available"] = {"model_type": "Detection", "Model": "yolov8n.pt", "Type": "None"}

        operation_selection = st.multiselect(
            "Select Operation Type(s)", display_ops,
            default=[display_ops[0]] if display_ops else None
        )

        selected_configs = []
        if operation_selection:
            for op in operation_selection:
                selected_configs.append(display_map.get(op))
        else:
            if display_ops:
                selected_configs.append(display_map.get(display_ops[0]))

        descriptions = [cfg.get('Type', 'Unknown') for cfg in selected_configs]
        st.caption(f"Selected Models: {', '.join(descriptions)}")

        # --- Backend Config ---
        config_path = str(Path(__file__).resolve().parent.parent / "config.yaml")
        yaml_config = load_config(config_path)

        yaml_backends = yaml_config.get('backends', [])
        cfg_buffer_size = yaml_config.get('buffer_size', 300)
        cfg_target_fps = yaml_config.get('target_fps', 30)
        cfg_re_entry_cooldown = yaml_config.get('re_entry_cooldown', 30)

        # --- Confidence ---
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

        # --- Video Input ---
        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
        video_path_input = st.text_input("Enter Video URL", "")

        # --- Preview Buttons ---
        col_preview, col_hide = st.columns(2)
        with col_preview:
            preview_frame_btn = st.button("Preview Frame")
        with col_hide:
            hide_preview_btn = st.button("Hide Preview")

        # --- Fetch Classes from Backends ---
        st.subheader("📊 Analytics")
        st.divider()

        @st.cache_resource
        def load_model_names(model_path):
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                return model.names
            except Exception:
                return {}

        class_options = []
        if selected_configs:
            selected_model_names = set()
            for cfg in selected_configs:
                model_path = Path(cfg['Model'])
                selected_model_names.add(model_path.stem)

            relevant_backends = []
            if yaml_backends:
                base_url = yaml_backends[0]['url']
                for i, m_name in enumerate(selected_model_names):
                    relevant_backends.append({
                        'url': base_url,
                        'model': m_name,
                        'name': f"Local GPU ({m_name})"
                    })
            connected_backends = []

            if relevant_backends:
                backend_hash = str(relevant_backends)
                if 'cached_classes' not in st.session_state:
                    st.session_state.cached_classes = {}
                if 'cached_connected_backends' not in st.session_state:
                    st.session_state.cached_connected_backends = {}

                if backend_hash in st.session_state.cached_classes:
                    class_options = st.session_state.cached_classes[backend_hash]
                    connected_backends = st.session_state.cached_connected_backends.get(backend_hash, [])
                else:
                    async def fetch_classes():
                        mgr = BackendClientManager(relevant_backends)
                        await mgr.initialize_sessions(quiet=True)
                        classes = mgr.get_all_classes()
                        connected = [
                            relevant_backends[idx]['name']
                            for idx, status in mgr.health_status.items()
                            if status['status'] == 'online'
                        ]
                        await mgr.cleanup()
                        return classes, connected

                    try:
                        with st.spinner("Initializing models (Downloading weights if necessary, this might take a few minutes)..."):
                            class_options, connected_backends = await fetch_classes()
                            st.session_state.cached_classes[backend_hash] = class_options
                            st.session_state.cached_connected_backends[backend_hash] = connected_backends
                    except Exception as e:
                        logger.error("Failed to fetch classes from backend: %s", e)
                        try:
                            m_path = selected_configs[0].get("Model")
                            if os.path.exists(m_path):
                                class_names = load_model_names(m_path)
                                class_options = [f"{k}: {v}" for k, v in class_names.items()]
                        except:
                            pass

        # --- Detection Only Mode ---
        detection_only_mode = st.checkbox("Enable Detection Only Mode", value=False,
                                          help="Run object detection without analytics logic.")
        detection_only_indices = []

        if detection_only_mode:
            st.info("Analytics features are disabled in Detection Only Mode.")
            detection_only_classes = st.multiselect("Select Classes to Detect", options=class_options)
            detection_only_indices = [int(c.split(":")[0]) for c in detection_only_classes]

        # --- Analytics Toggles ---
        enable_line_counting = False
        enable_roi = False
        enable_dwell = False
        enable_heatmap = False
        st.session_state.invalid_roi_config = False

        # --- Line Counting ---
        if not detection_only_mode:
            st.divider()
            
            def on_line_toggle():
                if 'preview_mount_id' in st.session_state:
                    st.session_state.preview_mount_id += 1
                    
            enable_line_counting = st.checkbox("Enable Line Counting", key="chk_line_counting", on_change=on_line_toggle)

        if 'annotated_lines' not in st.session_state: st.session_state.annotated_lines = []
        if 'current_line_points' not in st.session_state: st.session_state.current_line_points = []
        if 'line_drawing_mode' not in st.session_state: st.session_state.line_drawing_mode = False
        if 'show_preview' not in st.session_state: st.session_state.show_preview = False

        if preview_frame_btn: st.session_state.show_preview = True
        if hide_preview_btn: st.session_state.show_preview = False

        line_configs = []
        if enable_line_counting:
            if st.session_state.line_drawing_mode:
                if st.button("Finish Line"):
                    if len(st.session_state.current_line_points) >= 2:
                        c = st.session_state.current_line_points
                        st.session_state.annotated_lines.append({
                            'coords': [c[0][0], c[0][1], c[1][0], c[1][1]],
                            'in_side': 'top'
                        })
                        st.session_state.current_line_points = []
                    st.session_state.line_drawing_mode = False
                    st.rerun()
            else:
                if st.button("Start Line Drawing"):
                    st.session_state.line_drawing_mode = True
                    st.session_state.current_line_points = []
                    st.session_state.show_preview = True
                    st.rerun()

            if st.session_state.annotated_lines:
                for idx, line in enumerate(st.session_state.annotated_lines):
                    col_label, col_del = st.columns([3, 1])
                    with col_label:
                        st.text(f"Line {idx+1}")
                    with col_del:
                        if st.button("🗑️", key=f"del_line_{idx}", help=f"Delete Line {idx+1}"):
                            st.session_state.annotated_lines.pop(idx)
                            st.rerun()

                    coords = line['coords']
                    dx = abs(coords[2] - coords[0])
                    dy = abs(coords[3] - coords[1])
                    is_horizontal = dx > dy

                    if is_horizontal:
                        options = ["⬇️ Down=IN", "⬆️ Up=IN"]
                        option_values = {"⬇️ Down=IN": "bottom", "⬆️ Up=IN": "top"}
                    else:
                        options = ["➡️ Right=IN", "⬅️ Left=IN"]
                        option_values = {"➡️ Right=IN": "right", "⬅️ Left=IN": "left"}

                    current_value = line.get('in_side', list(option_values.values())[0])
                    current_index = 0
                    for i, (display, value) in enumerate(option_values.items()):
                        if value == current_value:
                            current_index = i
                            break

                    selected_option = st.selectbox(
                        "IN direction", options,
                        key=f"annotated_in_side_{idx}", index=current_index,
                        label_visibility="collapsed"
                    )
                    in_side = option_values.get(selected_option, list(option_values.values())[0])
                    line['in_side'] = in_side

                    selected_classes = st.multiselect(
                        f"Classes for Line {idx+1}", class_options,
                        default=[], key=f"line_classes_{idx}",
                        help="Leave empty to count ALL classes"
                    )
                    allowed_classes = [int(c.split(":")[0]) for c in selected_classes] if selected_classes else None

                    line_configs.append({
                        "coords": line['coords'],
                        "in_side": in_side,
                        "allowed_classes": allowed_classes
                    })
                    line['allowed_classes'] = allowed_classes

                if st.button("Clear All Lines"):
                    st.session_state.annotated_lines = []
                    st.rerun()
                    
        if not enable_line_counting:
            st.session_state.line_drawing_mode = False
            st.session_state.current_line_points = []

        # --- ROI Congestion ---
        if not detection_only_mode:
            st.divider()
            
            def on_roi_toggle():
                if 'preview_mount_id' in st.session_state:
                    st.session_state.preview_mount_id += 1
                    
            enable_roi = st.checkbox("Enable ROI Congestion", key="chk_roi_congestion", on_change=on_roi_toggle)
        if 'annotated_rois' not in st.session_state: st.session_state.annotated_rois = []
        if 'current_roi_points' not in st.session_state: st.session_state.current_roi_points = []
        if 'roi_drawing_mode' not in st.session_state: st.session_state.roi_drawing_mode = False

        roi_configs = []
        if enable_roi:
            if st.session_state.roi_drawing_mode:
                if st.button("Finish ROI"):
                    if len(st.session_state.current_roi_points) >= 3:
                        st.session_state.annotated_rois.append({'vertices': st.session_state.current_roi_points})
                        st.session_state.current_roi_points = []
                    st.session_state.roi_drawing_mode = False
                    st.rerun()
            else:
                if st.button("Start ROI Drawing"):
                    st.session_state.roi_drawing_mode = True
                    st.session_state.current_roi_points = []
                    st.session_state.show_preview = True
                    st.rerun()

            if st.session_state.annotated_rois:
                st.write(f"**{len(st.session_state.annotated_rois)} ROI(s) defined:**")
                for idx, roi_data in enumerate(st.session_state.annotated_rois):
                    col_info, col_del = st.columns([4, 1])
                    with col_info:
                        st.text(f"ROI {idx+1}: {len(roi_data['vertices'])} vertices")
                    with col_del:
                        if st.button("🗑️", key=f"del_roi_{idx}"):
                            st.session_state.annotated_rois.pop(idx)
                            st.rerun()

                    l_thresh = st.number_input(f"Low Threshold (ROI {idx+1})", 1, 500, 5, key=f"roi_lo_{idx}")
                    h_thresh = st.number_input(f"High Threshold (ROI {idx+1})", 1, 1000, 15, key=f"roi_hi_{idx}")

                    if l_thresh >= h_thresh:
                        st.error(f"⚠️ High Threshold must be strictly greater than Low Threshold for ROI {idx+1}.")
                        st.session_state.invalid_roi_config = True

                    selected_classes = st.multiselect(
                        f"Classes for ROI {idx+1}", class_options,
                        default=[], key=f"roi_classes_{idx}",
                        help="Leave empty to include ALL classes"
                    )
                    allowed_classes = [int(c.split(":")[0]) for c in selected_classes] if selected_classes else None

                    roi_configs.append({
                        "vertices": roi_data['vertices'],
                        "low_threshold": l_thresh,
                        "high_threshold": h_thresh,
                        "allowed_classes": allowed_classes
                    })
                    roi_data['allowed_classes'] = allowed_classes

                if st.button("Clear ROIs"):
                    st.session_state.annotated_rois = []
                    st.rerun()
                    
        if not enable_roi:
            st.session_state.roi_drawing_mode = False
            st.session_state.current_roi_points = []

        # --- Dwell Time ---
        if not detection_only_mode:
            st.divider()
            
            def on_dwell_toggle():
                if 'preview_mount_id' in st.session_state:
                    st.session_state.preview_mount_id += 1
                    
            enable_dwell = st.checkbox("Enable Dwell Time", key="chk_dwell_time", on_change=on_dwell_toggle)
        if 'dwell_rois' not in st.session_state: st.session_state.dwell_rois = []
        if 'current_dwell_points' not in st.session_state: st.session_state.current_dwell_points = []
        if 'dwell_drawing_mode' not in st.session_state: st.session_state.dwell_drawing_mode = False

        dwell_configs = []
        dwell_threshold_global = 30.0
        if enable_dwell:
            if st.session_state.dwell_drawing_mode:
                if st.button("Finish Dwell Zone"):
                    if len(st.session_state.current_dwell_points) >= 3:
                        st.session_state.dwell_rois.append({'vertices': st.session_state.current_dwell_points})
                        st.session_state.current_dwell_points = []
                    st.session_state.dwell_drawing_mode = False
                    st.rerun()
            else:
                if st.button("Start Dwell Drawing"):
                    st.session_state.dwell_drawing_mode = True
                    st.session_state.current_dwell_points = []
                    st.session_state.show_preview = True
                    st.rerun()

            dwell_threshold_global = st.slider("Dwell Alert Threshold (seconds)", 5.0, 120.0, 30.0, 1.0)

            if st.session_state.dwell_rois:
                st.write(f"**{len(st.session_state.dwell_rois)} Dwell Zone(s) defined:**")
                for idx, dwell_data in enumerate(st.session_state.dwell_rois):
                    col_info, col_del = st.columns([4, 1])
                    with col_info:
                        st.text(f"Zone {idx+1}")
                    with col_del:
                        if st.button("🗑️", key=f"del_dwell_{idx}"):
                            st.session_state.dwell_rois.pop(idx)
                            st.rerun()

                    selected_classes = st.multiselect(
                        f"Classes for Dwell {idx+1}", class_options,
                        default=[], key=f"dwell_classes_{idx}"
                    )
                    allowed_classes = [int(c.split(":")[0]) for c in selected_classes] if selected_classes else None

                    dwell_configs.append({
                        "vertices": dwell_data['vertices'],
                        "dwell_threshold": dwell_threshold_global,
                        "allowed_classes": allowed_classes
                    })
                    dwell_data['allowed_classes'] = allowed_classes
                    
        if not enable_dwell:
            st.session_state.dwell_drawing_mode = False
            st.session_state.current_dwell_points = []

        # --- Heatmap ---
        if not detection_only_mode:
            st.divider()
            enable_heatmap = st.checkbox("Enable Heatmap")

        heatmap_config = {"enabled": False}
        if enable_heatmap:
            h_decay = st.slider("Decay Rate", 0.90, 0.99, 0.95, 0.01)
            h_opacity = st.slider("Opacity", 0.2, 0.7, 0.4, 0.05)

            selected_classes = st.multiselect("Classes for Heatmap", class_options,
                                              default=[], key="heatmap_classes")
            heatmap_classes = [int(c.split(":")[0]) for c in selected_classes] if selected_classes else None

            heatmap_config = {"enabled": True, "decay": h_decay, "opacity": h_opacity, "classes": heatmap_classes}

        st.divider()

        # --- Recording ---
        segment_duration_mins = 15
        downloads_placeholder = st.empty()

        if 'completed_segments' not in st.session_state:
            st.session_state.completed_segments = []

        if video_path_input:
            st.subheader("💾 Recording")
            segment_duration_mins = st.slider("Segment Duration (mins)", 1, 60, 15, 1,
                                              help="Split video every X minutes")
            
        render_download_list(downloads_placeholder)

        # --- Backend Status ---
        if 'connected_backends' in dir() and connected_backends:
            for name in connected_backends:
                st.toast(f"Backend connected: {name}", icon="✅")

        # st.divider()

        # --- Visualization Options ---
        if 'show_visualization' not in st.session_state:
            st.session_state.show_visualization = True
        if 'show_labels' not in st.session_state:
            st.session_state.show_labels = False
            
        show_visualization = st.checkbox("Show Real-Time Visualization", key="show_visualization")
        
        preview_resolution = 800 # Default if hidden
        
        if show_visualization:
            show_labels = st.checkbox("Show Labels", key="show_labels",
                                      help="Display class name and track ID on bounding boxes")
            preview_res_mapping = {
                "Fast (800p)": 800,
                "HD (1280p)": 1280, 
                "FHD (1920p)": 1920
            }
            res_choice = st.selectbox(
                "Live Preview Quality", 
                list(preview_res_mapping.keys()), 
                index=0, 
                help="Lower quality improves Streamlit playback smoothness without affecting the final recorded video."
            )
            preview_resolution = preview_res_mapping[res_choice]
        else:
            show_labels = False

        st.divider()

        # --- Start/Stop ---
        if 'is_processing' not in st.session_state:
            st.session_state.is_processing = False

        if not st.session_state.is_processing:
            if st.button("Start Processing", use_container_width=True, type="primary"):
                if getattr(st.session_state, 'invalid_roi_config', False):
                    st.toast("⚠️ Fix the invalid ROI Congestion Thresholds before starting.")
                elif not (detection_only_mode or enable_line_counting or enable_roi or enable_dwell or enable_heatmap):
                    st.toast("⚠️ Please select an operation from the Analytics menu or enable Detection Only Mode.")
                else:
                    # Hide the previous playback window so live preview works again
                    st.session_state.show_video_player = False 
                    st.session_state.is_processing = True
                    st.rerun()
        else:
            if st.button("Stop Processing", use_container_width=True, type="primary"):
                st.session_state.is_processing = False
                if 'preview_mount_id' in st.session_state:
                    st.session_state.preview_mount_id += 1

    # --- Return Config ---
    return {
        'source_type': source_type,
        'selected_configs': selected_configs,
        'confidence': confidence,
        'uploaded_video': uploaded_video,
        'video_path_input': video_path_input,
        'detection_only_mode': detection_only_mode,
        'detection_only_indices': detection_only_indices,
        'enable_line_counting': enable_line_counting,
        'enable_roi': enable_roi,
        'enable_dwell': enable_dwell,
        'enable_heatmap': enable_heatmap,
        'heatmap_config': heatmap_config,
        'segment_duration_mins': segment_duration_mins,
        'show_visualization': show_visualization,
        'show_labels': show_labels,
        'preview_resolution': preview_resolution,
        'class_options': class_options,
        'yaml_backends': yaml_backends,
        'cfg_buffer_size': cfg_buffer_size,
        'cfg_target_fps': cfg_target_fps,
        'cfg_re_entry_cooldown': cfg_re_entry_cooldown,
        'downloads_placeholder': downloads_placeholder,
        'line_configs': line_configs,
        'roi_configs': roi_configs,
        'dwell_configs': dwell_configs
    }
