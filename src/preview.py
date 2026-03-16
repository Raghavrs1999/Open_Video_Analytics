"""
Preview Module

Handles the video preview frame display and interactive drawing
for lines, ROI zones, and dwell zones.
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import os
import cv2
import logging
import tempfile
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from .utils import is_url, get_youtube_stream_url

logger = logging.getLogger(__name__)


def render_preview(config: dict, preview_placeholder):
    """
    Render the video preview with interactive annotation drawing.

    Args:
        config: The config dict returned by render_sidebar().
        preview_placeholder: Streamlit placeholder for the preview area.
    """
    uploaded_video = config['uploaded_video']
    video_path_input = config['video_path_input']
    enable_line_counting = config['enable_line_counting']

    preview_source = None
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_video.read())
            preview_source = tfile.name
            uploaded_video.seek(0)
    elif video_path_input:
        if os.path.exists(video_path_input):
            preview_source = video_path_input
        elif is_url(video_path_input):
            if "youtube.com" in video_path_input or "youtu.be" in video_path_input:
                with st.spinner("Resolving YouTube URL..."):
                    preview_source = get_youtube_stream_url(video_path_input)
            else:
                preview_source = video_path_input

    if not preview_source:
        return

    cap_prev = cv2.VideoCapture(preview_source)
    ret, frame = cap_prev.read()
    cap_prev.release()

    if not ret:
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw existing annotations
    if enable_line_counting:
        for line in st.session_state.annotated_lines:
            c = line['coords']
            cv2.line(frame, (c[0], c[1]), (c[2], c[3]), (255, 0, 0), 3)
            
    if config.get('enable_roi', False):
        for roi in st.session_state.annotated_rois:
            pts = np.array(roi['vertices'], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 3)
            
    if config.get('enable_dwell', False):
        for dw in st.session_state.dwell_rois:
            pts = np.array(dw['vertices'], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 165, 255), 3)

    # Draw current in-progress points
    current_col = (0, 255, 0)
    current_pts = st.session_state.current_line_points
    if st.session_state.roi_drawing_mode:
        current_pts = st.session_state.current_roi_points
        current_col = (255, 0, 255)
    if st.session_state.dwell_drawing_mode:
        current_pts = st.session_state.current_dwell_points
        current_col = (0, 255, 255)

    for pt in current_pts:
        cv2.circle(frame, pt, 5, current_col, -1)

    # Calculate scale for click mapping
    h, w = frame.shape[:2]
    display_width = 800
    scale = w / display_width

    if 'preview_mount_id' not in st.session_state:
        st.session_state.preview_mount_id = 0

    with preview_placeholder.container():
        val = streamlit_image_coordinates(frame, width=display_width, key=f"preview_coords_{st.session_state.preview_mount_id}")

        if val:
            x = int(val['x'] * scale)
            y = int(val['y'] * scale)
            pt = (x, y)

            # Line drawing
            if st.session_state.line_drawing_mode:
                if 'last_click' not in st.session_state:
                    st.session_state.last_click = None
                if st.session_state.last_click != pt:
                    st.session_state.last_click = pt
                    st.session_state.current_line_points.append(pt)
                    if len(st.session_state.current_line_points) == 2:
                        c = st.session_state.current_line_points
                        st.session_state.annotated_lines.append({
                            'coords': [c[0][0], c[0][1], c[1][0], c[1][1]],
                            'in_side': 'top'
                        })
                        st.session_state.current_line_points = []
                        st.session_state.line_drawing_mode = False
                    st.rerun()

            # ROI drawing
            elif st.session_state.roi_drawing_mode:
                if 'last_roi_click' not in st.session_state:
                    st.session_state.last_roi_click = None
                if st.session_state.last_roi_click != pt:
                    st.session_state.last_roi_click = pt
                    if len(st.session_state.current_roi_points) >= 3:
                        fpt = st.session_state.current_roi_points[0]
                        if ((pt[0]-fpt[0])**2 + (pt[1]-fpt[1])**2)**0.5 < 30:
                            st.session_state.annotated_rois.append({'vertices': st.session_state.current_roi_points})
                            st.session_state.current_roi_points = []
                            st.session_state.roi_drawing_mode = False
                            st.rerun()
                    st.session_state.current_roi_points.append(pt)
                    st.rerun()

            # Dwell zone drawing
            elif st.session_state.dwell_drawing_mode:
                if 'last_dwell_click' not in st.session_state:
                    st.session_state.last_dwell_click = None
                if st.session_state.last_dwell_click != pt:
                    st.session_state.last_dwell_click = pt
                    if len(st.session_state.current_dwell_points) >= 3:
                        fpt = st.session_state.current_dwell_points[0]
                        if ((pt[0]-fpt[0])**2 + (pt[1]-fpt[1])**2)**0.5 < 30:
                            st.session_state.dwell_rois.append({'vertices': st.session_state.current_dwell_points})
                            st.session_state.current_dwell_points = []
                            st.session_state.dwell_drawing_mode = False
                            st.rerun()
                    st.session_state.current_dwell_points.append(pt)
                    st.rerun()
                    
    # Forcefully wipe states if tools are off
    if not st.session_state.line_drawing_mode:
        st.session_state.current_line_points = []
        st.session_state.last_click = None
        
    if not st.session_state.roi_drawing_mode:
        st.session_state.current_roi_points = []
        st.session_state.last_roi_click = None
        
    if not st.session_state.dwell_drawing_mode:
        st.session_state.current_dwell_points = []
        st.session_state.last_dwell_click = None
