"""
Processing Module

Contains the producer-consumer video processing pipeline:
- Frame capture and backend inference (producer)
- Playback, analytics, and recording (consumer)
- Segment finalization and crash recovery
"""

__author__ = "Raghav Sharma"
__version__ = "1.0.0"

import os
import cv2
import uuid
import time
import shutil
import asyncio
import logging
import tempfile
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import datetime
from streamlit.runtime.scriptrunner_utils.exceptions import StopException

from .utils import is_url, get_youtube_stream_url
from .video_capture import VideoCapture
from .backend_client import BackendClientManager
from .sync_buffer import SyncBuffer
from .analytics import apply_analytics
from .line_counter import LineCounter
from .roi_congestion import ROICongestion
from .visualization import get_class_color
from .sidebar import render_download_list

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def finalize_segment(temp_path, event_log):
    """Move temp video to final path, convert AVI→MP4, and save report."""
    if not temp_path or not os.path.exists(temp_path):
        return
    try:
        seg_id = uuid.uuid4()
        seg_avi_path = str(OUTPUT_DIR / f"segment_final_{seg_id}.avi")
        seg_mp4_path = str(OUTPUT_DIR / f"segment_final_{seg_id}.mp4")
        shutil.move(temp_path, seg_avi_path)

        # Convert AVI (MJPG) to MP4 (H.264)
        import subprocess
        
        if shutil.which('ffmpeg'):
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-i', seg_avi_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', seg_mp4_path],
                    capture_output=True, timeout=120
                )
                if os.path.exists(seg_mp4_path):
                    os.remove(seg_avi_path)
                    seg_video_path = seg_mp4_path
                else:
                    seg_video_path = seg_avi_path
            except Exception as e:
                logger.warning("FFmpeg conversion failed: %s", e)
                seg_video_path = seg_avi_path
        else:
            # Silently fallback to AVI if ffmpeg is not installed
            seg_video_path = seg_avi_path

        report_path = None
        if event_log:
            import pandas as pd
            df = pd.DataFrame(event_log)
            report_path = str(OUTPUT_DIR / f"report_final_{seg_id}.csv")
            df.to_csv(report_path, index=False)

        if 'completed_segments' not in st.session_state:
            st.session_state.completed_segments = []

        st.session_state.completed_segments.append({
            'video': seg_video_path,
            'report': report_path,
            'time': datetime.now().strftime("%H:%M:%S")
        })

        st.success("Video processing completed — download from the sidebar.", icon="✅")
        st.toast("Video Processing Completed", icon="✅")

    except Exception as e:
        logger.error("Error finalizing segment: %s", e)


def recover_interrupted_segment(downloads_placeholder):
    """Check for and recover dangling temp files from a previous interrupted session."""
    if not st.session_state.is_processing and st.session_state.get('current_temp_video'):
        temp = st.session_state.current_temp_video
        log = st.session_state.get('current_stats_log', [])

        if os.path.exists(temp):
            finalize_segment(temp, log)

        st.session_state.current_temp_video = None
        st.session_state.current_stats_log = []

        render_download_list(downloads_placeholder)


async def run_processing(config: dict, video_placeholder):
    """Run the full producer-consumer processing pipeline."""
    # --- Resolve Video Source ---
    uploaded_video = config['uploaded_video']
    video_path_input = config['video_path_input']
    source = None
    temp_uploaded_source_path = None

    if uploaded_video:
        uploaded_video.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_video.read())
            source = tfile.name
            temp_uploaded_source_path = source
    elif video_path_input:
        source = video_path_input

    if not source:
        st.error("No source provided")
        return

    # --- Setup Backends ---
    yaml_backends = config['yaml_backends']
    selected_configs = config['selected_configs']
    cfg_buffer_size = config['cfg_buffer_size']
    cfg_target_fps = config['cfg_target_fps']
    cfg_re_entry_cooldown = config['cfg_re_entry_cooldown']

    selected_model_names = set()
    for cfg in selected_configs:
        model_path = Path(cfg['Model'])
        selected_model_names.add(model_path.stem)

    backend_configs = []
    if yaml_backends:
        base_url = yaml_backends[0]['url']
        for i, m_name in enumerate(selected_model_names):
            backend_configs.append({
                'url': base_url,
                'model': m_name,
                'name': f"Local GPU ({m_name})"
            })

    if not backend_configs:
        st.error(f"No backends configured for selected models: {selected_model_names}")
        return

    backend_manager = BackendClientManager(backend_configs)
    await backend_manager.initialize_sessions(quiet=True)

    if not backend_manager.active_backends:
        st.error("No active backends")
        return

    # --- Initialize Analytics Objects ---
    enable_line_counting = config['enable_line_counting']
    enable_roi = config['enable_roi']
    enable_dwell = config['enable_dwell']
    detection_only_mode = config['detection_only_mode']
    detection_only_indices = config['detection_only_indices']
    heatmap_config = config['heatmap_config']
    segment_duration_mins = config['segment_duration_mins']
    show_visualization = config['show_visualization']
    show_labels = config['show_labels']
    class_options = config['class_options']
    downloads_placeholder = config['downloads_placeholder']

    line_counters = []
    roi_managers = []
    dwell_managers = []

    saved_line_configs = config.get('line_configs', [])
    roi_configs = config.get('roi_configs', [])
    dwell_configs = config.get('dwell_configs', [])

    if enable_line_counting and saved_line_configs:
        for i, conf in enumerate(saved_line_configs):
            c = conf['coords']
            line_counters.append(LineCounter(
                line_id=i,
                point1=(int(c[0]), int(c[1])),
                point2=(int(c[2]), int(c[3])),
                in_direction=conf['in_side'],
                allowed_classes=conf.get('allowed_classes'),
                cooldown_frames=cfg_re_entry_cooldown
            ))

    if enable_roi and roi_configs:
        for i, conf in enumerate(roi_configs):
            roi_managers.append(ROICongestion(
                roi_id=i, vertices=conf['vertices'],
                low_threshold=conf.get('low_threshold', 5),
                high_threshold=conf.get('high_threshold', 15),
                enable_dwell=False, allowed_classes=conf.get('allowed_classes')
            ))

    if enable_dwell and dwell_configs:
        for i, conf in enumerate(dwell_configs):
            dwell_managers.append(ROICongestion(
                roi_id=100+i, vertices=conf['vertices'],
                name=f"Dwell {i+1}",
                dwell_threshold=conf.get('dwell_threshold', 30.0),
                enable_dwell=True, allowed_classes=conf.get('allowed_classes')
            ))

    heatmap_state = None

    # --- Start Capture ---
    cap = VideoCapture(source, target_fps=cfg_target_fps)
    cap.start()
    sync_buffer = SyncBuffer(len(backend_configs), buffer_size=cfg_buffer_size)

    playback_started = False
    frame_count = 0

    track_history = {}
    stats = {"unique_objects": set(), "event_log": []}

    video_writer = None
    temp_video_path = None

    last_health_check = time.time()
    last_autosave = time.time()
    autosave_interval = 300
    health_check_interval = 2

    health_placeholder = st.sidebar.empty()

    stop_producing_event = asyncio.Event()

    # --- Playback Loop (Consumer) ---
    async def playback_loop():
        try:
            nonlocal video_writer, temp_video_path, playback_started, heatmap_state

            playback_video_placeholder = st.empty()
            legend_placeholder = st.empty()

            # --- Smart Class Filter for Legend ---
            effective_filter = set()
            has_active_analytics = False
            allows_all_classes = False

            if detection_only_mode:
                has_active_analytics = True
                effective_filter.update(detection_only_indices)
            else:
                for items, label in [(line_counters, 'line'), (roi_managers, 'roi'), (dwell_managers, 'dwell')]:
                    if items:
                        has_active_analytics = True
                        for item in items:
                            if not item.allowed_classes:
                                allows_all_classes = True
                            else:
                                effective_filter.update(item.allowed_classes)

                if heatmap_config.get("enabled", False):
                    has_active_analytics = True
                    h_classes = heatmap_config.get("classes")
                    if not h_classes:
                        allows_all_classes = True
                    else:
                        effective_filter.update(h_classes)

            final_visual_classes = None
            if has_active_analytics and not allows_all_classes:
                final_visual_classes = list(effective_filter) if effective_filter else None

            # --- Legend HTML ---
            if class_options:
                legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; margin-bottom: 10px; align-items: center;'>"
                for cls_str in class_options:
                    try:
                        parts = cls_str.split(":")
                        cls_id = int(parts[0])
                        if final_visual_classes is not None and cls_id not in final_visual_classes:
                            continue
                        name = parts[1].strip()
                        color_bgr = get_class_color(cls_id)
                        r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
                        legend_html += f"""<div style='
                            display: inline-flex; align-items: center; gap: 6px;
                            padding: 4px 12px; border-radius: 20px;
                            background: rgba({r},{g},{b}, 0.25);
                            border: 1px solid rgba({r},{g},{b}, 0.5);
                            backdrop-filter: blur(4px);'><div style='
                            width: 8px; height: 8px; border-radius: 50%;
                            background-color: rgb({r},{g},{b});
                            box-shadow: 0 0 6px rgba({r},{g},{b}, 0.6);'></div>
                            <span style='font-size: 12px; font-weight: 500; color: #eee;
                            text-transform: capitalize;'>{name}</span></div>"""
                    except:
                        pass
                legend_html += "</div>"
                legend_placeholder.markdown(legend_html, unsafe_allow_html=True)

            FRAME_DURATION = 1.0 / (cap.fps if cap.fps > 0 else 30.0)
            segment_start_time = time.time()

            while not stop_producing_event.is_set() or len(sync_buffer.buffer) > 0:
                loop_start = time.time()

                if not playback_started:
                    if sync_buffer.is_ready_for_playback() or stop_producing_event.is_set():
                        playback_started = True
                        if stop_producing_event.is_set():
                            st.toast("Draining remaining buffer...", icon="⏳")
                        else:
                            segment_start_time = time.time()
                    else:
                        s = sync_buffer.get_status()
                        buf_fraction = min(s['buffer'] / cfg_buffer_size, 1.0)
                        playback_video_placeholder.progress(buf_fraction, text=f"Buffering video... {int(buf_fraction * 100)}%")
                        await asyncio.sleep(0.1)
                        continue

                if playback_started:
                    if not stop_producing_event.is_set() and sync_buffer.should_rebuffer():
                        st.toast("Re-buffering...", icon="⏳")
                        while not sync_buffer.is_ready_for_playback() and not stop_producing_event.is_set():
                            sync_buffer.try_move_to_buffer()
                            await asyncio.sleep(0.05)
                        continue

                data = sync_buffer.get_next_frame()
                if data:
                    if heatmap_state is None and heatmap_config.get("enabled", False):
                        h_map, w_map = data['frame'].shape[:2]
                        heatmap_state = np.zeros((h_map, w_map), dtype=np.float32)

                    annotated = apply_analytics(
                        data['frame'].copy(), data['results'],
                        line_counters, roi_managers, dwell_managers,
                        heatmap_config, heatmap_state,
                        data['frame_id'], cap.fps,
                        track_history=track_history, stats=stats,
                        global_class_filter=final_visual_classes,
                        show_labels=show_labels, detection_only=detection_only_mode
                    )

                    if show_visualization:
                        ui_h, ui_w = annotated.shape[:2]
                        target_w = config.get('preview_resolution', 800)  # Dynamically set from Sidebar UI
                        if ui_w > target_w:
                            scale = target_w / ui_w
                            ui_frame = cv2.resize(annotated, (int(ui_w * scale), int(ui_h * scale)))
                        else:
                            ui_frame = annotated
                        playback_video_placeholder.image(ui_frame, channels="BGR")

                    if video_writer is None:
                        h, w = data['frame'].shape[:2]
                        temp_video_path = str(OUTPUT_DIR / f"temp_{uuid.uuid4()}.avi")
                        video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'MJPG'), cap.fps, (w, h))
                        st.session_state.current_temp_video = temp_video_path
                        st.session_state.current_stats_log = stats.get("event_log", [])

                    if video_writer:
                        video_writer.write(annotated)

                    # Segment rotation
                    if playback_started and (time.time() - segment_start_time > (segment_duration_mins * 60)):
                        video_writer.release()
                        video_writer = None
                        finalize_segment(temp_video_path, stats["event_log"])
                        st.session_state.current_temp_video = None
                        st.session_state.current_stats_log = []
                        temp_video_path = None
                        render_download_list(downloads_placeholder)
                        segment_start_time = time.time()

                    elapsed = time.time() - loop_start
                    delay = max(0, FRAME_DURATION - elapsed)
                    await asyncio.sleep(delay)
                else:
                    if stop_producing_event.is_set():
                        st.toast("Buffer drained.", icon="✅")
                        break
                    else:
                        await asyncio.sleep(0.01)
        except StopException:
            pass
        except Exception as e:
            logger.error("Playback Loop Error: %s", e)

    # --- Start Consumer ---
    playback_task = asyncio.create_task(playback_loop())

    # --- Producer Loop ---
    background_tasks = set()

    async def process_frame_task(frame_arr, f_id):
        try:
            results = await backend_manager.track_frame(frame_arr, f_id)
            for idx, res in results.items():
                if res:
                    sync_buffer.add_results(f_id, idx, res['detections'])
        except asyncio.CancelledError:
            pass
        except StopException:
            pass
        except Exception as e:
            logger.error("Analysis failed for frame %d: %s", f_id, e)
        finally:
            sync_buffer.mark_complete(f_id)

    try:
        while True:
            now = time.time()

            # Periodic health check
            if now - last_health_check > health_check_interval:
                await backend_manager.check_all_health()
                last_health_check = now

                with health_placeholder.container():
                    st.write("---")
                    st.subheader("System Health")

                    for idx, status in backend_manager.health_status.items():
                        s = status.get("status", "unknown")
                        icon = "🟢" if s == "online" else "🔴"
                        st.markdown(f"**Backend {idx}** {icon}")
                        if s == "online":
                            mem = status.get("gpu_mem", 0)
                            sess = status.get("sessions", 0)
                            st.caption(f"GPU Mem: {mem:.1f} MB | Sessions: {sess}")
                        else:
                            st.caption(f"Error: {status.get('error', 'Unknown')}")

            # Periodic auto-save
            if (now - last_autosave > autosave_interval) or (len(stats["event_log"]) > 5000):
                if stats["event_log"]:
                    try:
                        import pandas as pd
                        df = pd.DataFrame(stats["event_log"])
                        autosave_path = f"outputs/reports/autosave_{backend_manager.base_session_id}.csv"
                        os.makedirs("outputs/reports", exist_ok=True)
                        header = not os.path.exists(autosave_path)
                        df.to_csv(autosave_path, mode='a', header=header, index=False)
                        stats["event_log"] = []
                        last_autosave = now
                    except Exception as e:
                        logger.error("Auto-save failed: %s", e)

            sync_buffer.try_move_to_buffer()

            # Back-pressure throttle
            if len(sync_buffer.pending) > 200:
                await asyncio.sleep(0.01)
                continue

            if playback_task.done():
                st.session_state.is_processing = False
                st.rerun()

            # Capture
            frame_data = cap.get_frame()
            if frame_data:
                fid = frame_data['frame_id']
                frm = frame_data['frame']
                frame_count += 1

                sync_buffer.add_frame(fid, frm)

                task = asyncio.create_task(process_frame_task(frm, fid))
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)

                if len(background_tasks) % 10 == 0:
                    await asyncio.sleep(0)

            elif not cap.running and cap.frame_queue.empty():
                break

            sync_buffer.try_move_to_buffer()
            await asyncio.sleep(0.001)

        # Wait for remaining tasks
        if background_tasks:
            st.toast(f"Finishing analysis for {len(background_tasks)} frames...", icon="⏳")
            await asyncio.gather(*background_tasks)

        sync_buffer.try_move_to_buffer()

        st.toast("Stopping... Draining buffer.", icon="⏳")
        stop_producing_event.set()
        await playback_task

        # Finalize last segment
        if video_writer:
            video_writer.release()
            video_writer = None
            finalize_segment(temp_video_path, stats["event_log"])
            temp_video_path = None

        st.session_state.is_processing = False
        st.session_state.processing_completed_naturally = True
        st.rerun()

    finally:
        st.session_state.is_processing = False
        if 'preview_mount_id' in st.session_state:
            st.session_state.preview_mount_id += 1
            
        if background_tasks:
            for task in background_tasks:
                if not task.done():
                    task.cancel()
        if cap:
            cap.stop()
        await backend_manager.cleanup()
        if video_writer:
            video_writer.release()

        if temp_uploaded_source_path and os.path.exists(temp_uploaded_source_path):
            try:
                os.unlink(temp_uploaded_source_path)
            except Exception as e:
                logger.warning("Failed to cleanup temp upload: %s", e)

        if st.session_state.pop('processing_completed_naturally', False):
            st.success("✅ Video processing completed. Download from the sidebar.")
        elif not st.session_state.is_processing:
            st.warning("Processing Stopped by User")
