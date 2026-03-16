"""
Open Video Analytics

Entry point for the Streamlit application.
Run with: streamlit run app.py
"""

__author__ = "Raghav Sharma"
__version__ = "0.1.0"

import os
import asyncio
import logging
import streamlit as st
from pathlib import Path

from src.sidebar import render_sidebar, render_download_list
from src.preview import render_preview
from src.processing import run_processing, recover_interrupted_segment

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


async def main():
    st.set_page_config(page_title="Open Video Analytics", page_icon="🎥", layout="wide")

    st.markdown("""
        <style>
        .main .block-container { padding-top: 2rem; }
        .stProgress > div > div > div > div { background-color: #4CAF50; }
        [data-testid="stSidebar"] { min-width: 350px; max-width: 400px; }
        </style>
        """, unsafe_allow_html=True)

    st.title("Open Video Analytics")

    # --- Sidebar ---
    config = await render_sidebar()

    # --- Auto-Preview Logic ---
    current_source_id = None
    if config['uploaded_video']:
        current_source_id = f"upload_{config['uploaded_video'].name}_{config['uploaded_video'].size}"
    elif config['video_path_input']:
        current_source_id = f"path_{config['video_path_input']}"

    if 'last_source_id' not in st.session_state:
        st.session_state.last_source_id = None

    if current_source_id and current_source_id != st.session_state.last_source_id:
        st.session_state.last_source_id = current_source_id
        st.session_state.show_preview = True

    # --- Main Display Area ---
    video_placeholder = st.empty()
    preview_placeholder = st.empty()

    # --- Video Player for Completed Segments ---
    show_video_player = False
    if not st.session_state.is_processing and st.session_state.get('completed_segments') and st.session_state.get('show_video_player', True):
        latest_seg = st.session_state.completed_segments[-1]
        if os.path.exists(latest_seg['video']):
            if latest_seg['video'].endswith('.mp4'):
                st.video(latest_seg['video'])
            else:
                st.info(f"📹 **Video saved as {Path(latest_seg['video']).suffix}**\n\nYour system does not have FFmpeg installed, so the browser cannot play this video format natively. Please download the video from the sidebar to view it in your local media player.", icon="ℹ️")
            show_video_player = True

    # --- Preview ---
    if not show_video_player and st.session_state.show_preview and not st.session_state.is_processing:
        render_preview(config, preview_placeholder)

    # --- Recovery ---
    recover_interrupted_segment(config['downloads_placeholder'])

    # --- Processing ---
    if st.session_state.is_processing:
        await run_processing(config, video_placeholder)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        if type(e).__name__ == "StopException":
            pass
        else:
            raise e
