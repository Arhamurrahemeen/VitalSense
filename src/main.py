"""VitalSense real-time health monitor orchestrator.

This module owns webcam capture, MediaPipe face tracking, signal extraction,
DSP/HRV/stress analysis, Groq feedback scheduling, and shared-state updates
for the Matplotlib dashboard.

The dashboard itself is kept in `dashboard.py` so visualization stays separate
from camera and AI control flow.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from collections import deque
from typing import Any

import cv2
import mediapipe as mp
import mediapipe.tasks as tasks
import numpy as np

from ai_feedback import FeedbackManager
from dashboard import DashboardApp, start_dashboard
from dsp_pipeline import FS, HIGH, LOW, apply_filter, compute_bpm_and_confidence, design_bandpass
from hrv import classify_stress, compute_hrv, get_stress_color
from lighting import classify_lighting, get_lighting_color

logger = logging.getLogger(__name__)

FS = 30
WINDOW_SECONDS = 15.0
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
FEEDBACK_INTERVAL = 30

MIN_FOREHEAD_HEIGHT = 20
MAX_FOREHEAD_HEIGHT = 180
START_DASHBOARD = True

# Forehead landmarks used for a stable ROI over the upper face.
FOREHEAD_IDS = [10, 109, 67, 103, 338, 297, 332, 284]

# Shared real-time state.
state_lock = threading.Lock()
green_buffer = deque(maxlen=WINDOW_SAMPLES)
timestamps = deque(maxlen=WINDOW_SAMPLES)
lighting_means: list[float] = []
measurement_state = "collecting"
window_frame_count = 0
session_start_time: float | None = None
last_feedback_time = 0.0
latest_feedback = "Initializing..."
latest_bpm = 0
latest_confidence = 0
latest_stress = "uncertain"
latest_rmssd: float | None = None
latest_sdnn: float | None = None
session_stats = {
    "max_bpm": 0,
    "min_bpm": 999,
    "sum_bpm": 0,
    "readings_count": 0,
}
window_index = 1
latest_distance = "DISTANCE OK"
latest_lighting = "good"
latest_luminance = 0.0
latest_fps = 0.0
latest_filtered_signal = np.asarray([], dtype=float)
shutdown_event = threading.Event()
face_landmarks_result = None

shared_state: dict[str, Any] = {
    "lock": state_lock,
    "green_buffer": green_buffer,
    "timestamps": timestamps,
    "raw_buffer": [],
    "filtered_buffer": [],
    "fft_freqs": [],
    "fft_mags": [],
    "bpm": 0,
    "confidence": 0,
    "stress": "uncertain",
    "timestamp": 0.0,
    "latest_bpm": latest_bpm,
    "latest_confidence": latest_confidence,
    "latest_stress": latest_stress,
    "latest_rmssd": latest_rmssd,
    "latest_sdnn": latest_sdnn,
    "latest_feedback": latest_feedback,
    "latest_distance": latest_distance,
    "latest_lighting": latest_lighting,
    "latest_luminance": latest_luminance,
    "latest_fps": latest_fps,
    "latest_filtered_signal": latest_filtered_signal,
    "latest_session_stats": session_stats,
    "measurement_state": measurement_state,
    "session_start_time": session_start_time,
    "last_feedback_time": last_feedback_time,
    "shutdown_event": shutdown_event,
}

bandpass_sos = design_bandpass(LOW, HIGH, FS, order=2)
feedback_manager = FeedbackManager()
dashboard_app: DashboardApp | None = None
dashboard_thread: threading.Thread | None = None
face_landmarker = None
cap = None
last_timestamp_ms = 0


def _log_main_error(operation: str, error: Exception, fallback: str) -> None:
    """Log orchestrator errors in the VitalSense three-part format.

    Args:
        operation: The operation that failed.
        error: The exception that caused the failure.
        fallback: The fallback strategy applied after the failure.

    Returns:
        None.
    """
    logger.error(
        "Main error | what failed=%s | what caused it=%s | fallback=%s",
        operation,
        error,
        fallback,
        exc_info=False,
    )
    print(
        f"[ERROR] {operation}: {type(error).__name__}: {error} | "
        f"fallback={fallback}"
    )


def _update_shared_state(**updates: Any) -> None:
    """Update the shared state dictionary under lock.

    Args:
        **updates: Key-value pairs to merge into the shared state.

    Returns:
        None.
    """
    fallback = "skip the shared-state update because the lock could not be acquired"

    try:
        with state_lock:
            for key, value in updates.items():
                shared_state[key] = value

    except Exception as exc:
        _log_main_error("_update_shared_state", exc, fallback)


def _next_timestamp_ms() -> int:
    """Return a strictly increasing millisecond timestamp for MediaPipe input.

    Args:
        None.

    Returns:
        A monotonically increasing integer timestamp in milliseconds.
    """
    global last_timestamp_ms

    fallback = "reuse the previous timestamp plus one millisecond"

    try:
        current_timestamp_ms = time.monotonic_ns() // 1_000_000
        if current_timestamp_ms <= last_timestamp_ms:
            current_timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = current_timestamp_ms
        return current_timestamp_ms

    except Exception as exc:
        _log_main_error("_next_timestamp_ms", exc, fallback)
        last_timestamp_ms += 1
        return last_timestamp_ms


def _shutdown_resources() -> None:
    """Release webcam, MediaPipe, and OpenCV resources exactly once.

    Args:
        None.

    Returns:
        None.
    """
    global face_landmarker, cap

    fallback = "continue shutdown even if one resource fails to close"

    try:
        shutdown_event.set()

        if face_landmarker is not None:
            try:
                face_landmarker.close()
            except Exception as exc:
                _log_main_error("_shutdown_resources face_landmarker.close", exc, fallback)

        if cap is not None:
            try:
                cap.release()
            except Exception as exc:
                _log_main_error("_shutdown_resources cap.release", exc, fallback)

        try:
            cv2.destroyAllWindows()
        except Exception as exc:
            _log_main_error("_shutdown_resources cv2.destroyAllWindows", exc, fallback)

        if dashboard_app is not None:
            try:
                dashboard_app.close()
            except Exception as exc:
                _log_main_error("_shutdown_resources dashboard_app.close", exc, fallback)

    finally:
        _update_shared_state(
            measurement_state=measurement_state,
            session_start_time=session_start_time,
            last_feedback_time=last_feedback_time,
            latest_feedback=latest_feedback,
            latest_bpm=latest_bpm,
            latest_confidence=latest_confidence,
            latest_stress=latest_stress,
            latest_rmssd=latest_rmssd,
            latest_sdnn=latest_sdnn,
            latest_distance=latest_distance,
            latest_lighting=latest_lighting,
            latest_luminance=latest_luminance,
            latest_fps=latest_fps,
            latest_filtered_signal=latest_filtered_signal,
            latest_session_stats=dict(session_stats),
        )


def _draw_right_aligned_text(frame: np.ndarray, text: str, y: int, color, scale: float = 0.5, thickness: int = 1) -> None:
    """Draw a text label aligned to the right edge of the frame.

    Args:
        frame: OpenCV BGR frame to annotate.
        text: Text to draw.
        y: Baseline y-coordinate.
        color: OpenCV BGR color tuple.
        scale: Font scale.
        thickness: Font thickness.

    Returns:
        None.
    """
    fallback = "skip the text overlay because annotation failed"

    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, _text_height), _baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(10, frame.shape[1] - text_width - 10)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness)

    except Exception as exc:
        _log_main_error("_draw_right_aligned_text", exc, fallback)


def draw_text_with_background(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.8, thickness: int = 2) -> None:
    """Draw text with a solid background for improved readability."""
    fallback = "skip the annotated text because drawing failed"

    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x0 = max(0, x - 5)
        y0 = max(0, y - text_height - 5)
        x1 = min(frame.shape[1], x + text_width + 5)
        y1 = min(frame.shape[0], y + baseline + 5)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)
    except Exception as exc:
        _log_main_error("draw_text_with_background", exc, fallback)


def draw_top_banner(frame: np.ndarray, text: str) -> None:
    """Draw a top banner with semi-transparent red overlay and white text."""
    try:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        draw_text_with_background(frame, text, 10, 28, (255, 255, 255), scale=0.8, thickness=2)
    except Exception as exc:
        _log_main_error("draw_top_banner", exc, "skip banner drawing")


def _draw_feedback_panel(frame: np.ndarray, feedback_text: str) -> None:
    """Draw the cached AI feedback as a bottom overlay panel.

    Args:
        frame: OpenCV BGR frame to annotate.
        feedback_text: Current feedback text from the Groq manager.

    Returns:
        None.
    """
    fallback = "skip the feedback overlay because annotation failed"

    try:
        panel_text = str(feedback_text).strip() or "AI feedback pending..."
        wrapped_lines = []
        words = panel_text.split()
        current_line = []

        for word in words:
            current_line.append(word)
            if len(" ".join(current_line)) >= 48:
                wrapped_lines.append(" ".join(current_line))
                current_line = []

        if current_line:
            wrapped_lines.append(" ".join(current_line))

        wrapped_lines = wrapped_lines[:3] or ["AI feedback pending..."]

        x0 = 10
        y0 = max(180, frame.shape[0] - (30 + 22 * len(wrapped_lines) + 20))
        x1 = min(frame.shape[1] - 10, x0 + 640)
        y1 = min(frame.shape[0] - 10, y0 + 30 + 22 * len(wrapped_lines) + 10)

        panel_fill = (20, 20, 20)
        panel_border = (0, 180, 255)
        title_text = "Feedback"

        if panel_text == "Groq Quota Exceeded. Retrying in next window...":
            panel_border = (0, 165, 255)
            title_text = "Feedback | API Cooldown"

        cv2.rectangle(frame, (x0, y0), (x1, y1), panel_fill, -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), panel_border, 1)
        cv2.putText(frame, title_text, (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, panel_border, 2)

        text_y = y0 + 44
        for line in wrapped_lines:
            cv2.putText(frame, line, (x0 + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text_y += 22

    except Exception as exc:
        _log_main_error("_draw_feedback_panel", exc, fallback)


def _extract_forehead_roi(frame: np.ndarray, landmarks) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Extract a forehead ROI from the current face landmarks.

    Args:
        frame: Current BGR webcam frame.
        landmarks: Face landmark list returned by MediaPipe.

    Returns:
        A tuple of (roi, bounding_box). Returns (None, None) if the ROI cannot
        be computed.
    """
    fallback = "skip ROI extraction and wait for the next valid face frame"

    try:
        if frame is None or frame.size == 0:
            raise ValueError("frame is empty")
        if landmarks is None:
            raise ValueError("landmarks are missing")

        height, width, _ = frame.shape
        points = np.asarray(
            [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in FOREHEAD_IDS],
            dtype=int,
        )

        x, y, roi_width, roi_height = cv2.boundingRect(points)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(width, x + roi_width)
        y1 = min(height, y + roi_height)

        if x1 <= x0 or y1 <= y0:
            return None, None

        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None, None

        return roi, (x0, y0, x1 - x0, y1 - y0)

    except Exception as exc:
        _log_main_error(
            "_extract_forehead_roi",
            exc,
            fallback,
        )
        return None, None


def _get_green_signal(roi: np.ndarray) -> float:
    """Compute the mean green-channel signal from a forehead ROI.

    Args:
        roi: Cropped forehead region in BGR format.

    Returns:
        Mean green-channel intensity or 0.0 on failure.
    """
    fallback = "return 0.0 because the ROI is invalid"

    try:
        if roi is None or roi.size == 0:
            raise ValueError("ROI is empty")

        return float(np.mean(roi[:, :, 1]))

    except Exception as exc:
        _log_main_error("_get_green_signal", exc, fallback)
        return 0.0


def _process_completed_window() -> None:
    """Process a full window of buffered rPPG samples.

    Args:
        None.

    Returns:
        None.
    """
    global measurement_state, window_frame_count, latest_bpm, latest_confidence
    global latest_stress, latest_rmssd, latest_sdnn, latest_feedback
    global latest_filtered_signal, session_start_time, last_feedback_time

    fallback = "reset the window and continue capturing with safe defaults"

    try:
        with state_lock:
            raw_signal = np.asarray(green_buffer, dtype=float)

        filtered_signal = apply_filter(raw_signal, bandpass_sos) if raw_signal.size > 0 else np.asarray([], dtype=float)
        latest_filtered_signal = filtered_signal

        bpm = 0
        confidence = 0
        rmssd_value: float | None = None
        sdnn_value: float | None = None
        stress_label = "uncertain"

        if filtered_signal.size > 0:
            bpm, confidence = compute_bpm_and_confidence(filtered_signal, fs=FS, low=LOW, high=HIGH)

            if confidence >= 40:
                rmssd_value, sdnn_value = compute_hrv(filtered_signal, fs=FS)
                stress_label = classify_stress(bpm, rmssd_value, confidence)
            else:
                stress_label = "uncertain"

        latest_bpm = bpm
        latest_confidence = confidence
        latest_rmssd = rmssd_value
        latest_sdnn = sdnn_value
        latest_stress = stress_label

        if bpm > 0 and confidence >= 40:
            session_stats["readings_count"] += 1
            session_stats["max_bpm"] = max(session_stats["max_bpm"], bpm)
            session_stats["min_bpm"] = min(session_stats["min_bpm"], bpm)

        rmssd_display = "--" if rmssd_value is None else f"{rmssd_value:.1f}"
        sdnn_display = "--" if sdnn_value is None else f"{sdnn_value:.1f}"
        min_display = "--" if session_stats["min_bpm"] == 999 else str(session_stats["min_bpm"])

        print(
            f"Window complete: BPM={bpm}, Confidence={confidence}%, "
            f"RMSSD={rmssd_display} ms, SDNN={sdnn_display} ms, Stress={stress_label}"
        )
        print("--- Session Range ---")
        print(f"Max BPM: {session_stats['max_bpm']} | Min BPM: {min_display}")

        duration_min = 0.0
        if session_start_time is not None:
            duration_min = (time.time() - session_start_time) / 60.0

        if duration_min >= 1.0 and confidence >= 40:
            if (time.time() - last_feedback_time) >= FEEDBACK_INTERVAL:
                scheduled = feedback_manager.request_feedback(
                    bpm=bpm,
                    rmssd=rmssd_value,
                    stress_label=stress_label,
                    confidence=confidence,
                    duration_min=duration_min,
                )
                if scheduled:
                    last_feedback_time = time.time()

        latest_feedback = feedback_manager.get_latest()
        print(f"Feedback cache: {latest_feedback or 'pending'}")

        with state_lock:
            green_buffer.clear()
            timestamps.clear()

        window_frame_count = 0
        measurement_state = "collecting"

        _update_shared_state(
            latest_bpm=latest_bpm,
            latest_confidence=latest_confidence,
            latest_stress=latest_stress,
            latest_rmssd=latest_rmssd,
            latest_sdnn=latest_sdnn,
            latest_feedback=latest_feedback,
            latest_filtered_signal=latest_filtered_signal,
            latest_session_stats=dict(session_stats),
            measurement_state=measurement_state,
            session_start_time=session_start_time,
            last_feedback_time=last_feedback_time,
        )

    except Exception as exc:
        _log_main_error("_process_completed_window", exc, fallback)
        measurement_state = "collecting"
        window_frame_count = 0
        with state_lock:
            green_buffer.clear()
            timestamps.clear()


def _build_face_landmarker():
    """Create the MediaPipe Face Landmarker with the Tasks API.

    Args:
        None.

    Returns:
        A configured FaceLandmarker instance.
    """
    fallback = "return None because MediaPipe initialization failed"

    try:
        BaseOptions = tasks.BaseOptions
        FaceLandmarker = tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        return FaceLandmarker.create_from_options(options)

    except Exception as exc:
        _log_main_error("_build_face_landmarker", exc, fallback)
        return None


def _open_camera() -> cv2.VideoCapture | None:
    """Open the default webcam.

    Args:
        None.

    Returns:
        An initialized OpenCV VideoCapture object, or None on failure.
    """
    fallback = "return None so the caller can exit cleanly"

    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("camera could not be opened")
            
        # Request standard resolution and FPS to help stabilize the capture pipeline.
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, FS)
        
        return camera

    except Exception as exc:
        _log_main_error("_open_camera", exc, fallback)
        return None


def run() -> None:
    """Run the VitalSense monitoring loop.

    Args:
        None.

    Returns:
        None.
    """
    global face_landmarker, cap, measurement_state, window_frame_count
    global session_start_time, latest_feedback, latest_bpm, latest_confidence
    global latest_stress, latest_rmssd, latest_sdnn, latest_luminance, latest_fps
    global latest_distance, latest_lighting, face_landmarks_result

    fallback = "keep the webcam loop alive and continue on recoverable failures"

    try:
        face_landmarker = _build_face_landmarker()
        if face_landmarker is None:
            return

        cap = _open_camera()
        if cap is None:
            return

        cv2.namedWindow("VitalSense Main Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VitalSense Main Monitor", 640, 480)
        cv2.moveWindow("VitalSense Main Monitor", 50, 50)

        if START_DASHBOARD:
            try:
                global dashboard_app, dashboard_thread
                dashboard_app, dashboard_thread = start_dashboard(shared_state)
            except Exception as exc:
                _log_main_error("run dashboard startup", exc, "continue without the Matplotlib dashboard")

        prev_time = time.time()
        fps_ema = 0.0
        alpha = 0.1  # Smoothing factor for FPS display
        print("VitalSense Stage 1: Press 'q' to stop.")

        while cap.isOpened() and not shutdown_event.is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    _log_main_error("camera read", RuntimeError("frame capture failed"), "skip this frame and continue")
                    continue

                current_time = time.time()
                elapsed = current_time - prev_time
                prev_time = current_time
                
                # Calculate FPS with EMA smoothing to avoid jittery display
                fps = 1.0 / elapsed if elapsed > 1e-6 else 0.0
                if fps_ema == 0.0:
                    fps_ema = fps
                else:
                    fps_ema = alpha * fps + (1.0 - alpha) * fps_ema
                latest_fps = fps_ema

                frame = cv2.flip(frame, 1)

                lighting_status, mean_luminance = classify_lighting(frame, lighting_means, window=30)
                latest_lighting = lighting_status
                latest_luminance = mean_luminance
                lighting_color = get_lighting_color(lighting_status)
                ok_color = get_lighting_color("good")
                alert_color = get_lighting_color("flicker")
                stress_color = get_stress_color(latest_stress)

                if measurement_state == "collecting":
                    window_frame_count += 1

                if lighting_status != "good":
                    latest_distance = "N/A"
                    cv2.putText(
                        frame,
                        "Lighting poor - signal extraction paused",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        alert_color,
                        2,
                    )
                else:
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        face_landmarks_result = face_landmarker.detect_for_video(mp_image, _next_timestamp_ms())
                    except Exception as exc:
                        _log_main_error(
                            "MediaPipe frame processing",
                            exc,
                            "skip ROI extraction for this frame and continue capturing",
                        )
                        face_landmarks_result = None

                    if face_landmarks_result and face_landmarks_result.face_landmarks:
                        landmarks = face_landmarks_result.face_landmarks[0]
                        roi, bbox = _extract_forehead_roi(frame, landmarks)

                        if bbox is not None:
                            x, y, bw, bh = bbox
                            if bh < MIN_FOREHEAD_HEIGHT:
                                latest_distance = "MOVE CLOSER"
                                distance_color = alert_color
                            elif bh > MAX_FOREHEAD_HEIGHT:
                                latest_distance = "TOO CLOSE"
                                distance_color = alert_color
                            else:
                                latest_distance = "DISTANCE OK"
                                distance_color = ok_color

                            cv2.putText(
                                frame,
                                latest_distance,
                                (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                distance_color,
                                2,
                            )

                            if roi is not None and roi.size > 0:
                                g_mean = _get_green_signal(roi)

                                if lighting_status == "good" and latest_distance == "DISTANCE OK":
                                    with state_lock:
                                        green_buffer.append(g_mean)
                                        timestamps.append(time.time())

                                    if session_start_time is None:
                                        session_start_time = time.time()

                                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                                cv2.putText(
                                    frame,
                                    f"G-Mean: {g_mean:.2f}",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                )

                recording_seconds = max(len(green_buffer), window_frame_count) / FS
                recording_seconds = min(recording_seconds, WINDOW_SECONDS)

                cv2.putText(
                    frame,
                    f"FPS: {latest_fps:.1f} | Lighting: {latest_lighting} ({latest_luminance:.1f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    lighting_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"Recording: {recording_seconds:.1f}s / {WINDOW_SECONDS:.1f}s",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    lighting_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"BPM: {latest_bpm if latest_bpm > 0 else '--'} | Conf: {latest_confidence if latest_confidence > 0 else '--'}% | Stress: {latest_stress}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    stress_color,
                    2,
                )

                _draw_right_aligned_text(frame, f"Max BPM: {session_stats['max_bpm']}", 30, ok_color)
                min_display = "--" if session_stats["min_bpm"] == 999 else str(session_stats["min_bpm"])
                _draw_right_aligned_text(frame, f"Min BPM: {min_display}", 50, ok_color)
                _draw_right_aligned_text(frame, f"Distance: {latest_distance}", 70, ok_color)

                _draw_feedback_panel(frame, feedback_manager.get_latest())

                cv2.imshow("VitalSense Main Monitor", frame)

                if measurement_state == "collecting" and len(green_buffer) >= WINDOW_SAMPLES:
                    measurement_state = "processing"
                    _process_completed_window()

                _update_shared_state(
                    latest_bpm=latest_bpm,
                    latest_confidence=latest_confidence,
                    latest_stress=latest_stress,
                    latest_rmssd=latest_rmssd,
                    latest_sdnn=latest_sdnn,
                    latest_feedback=latest_feedback,
                    latest_distance=latest_distance,
                    latest_lighting=latest_lighting,
                    latest_luminance=latest_luminance,
                    latest_fps=latest_fps,
                    latest_filtered_signal=latest_filtered_signal,
                    latest_session_stats=dict(session_stats),
                    measurement_state=measurement_state,
                    session_start_time=session_start_time,
                    last_feedback_time=last_feedback_time,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception as exc:
                _log_main_error(
                    "run loop iteration",
                    exc,
                    "continue the webcam loop so the monitor does not crash",
                )
                time.sleep(0.01)
                continue

    except Exception as exc:
        _log_main_error("run", exc, fallback)

    finally:
        _shutdown_resources()
        print("Signal capture stopped.")
        print(f"Remaining buffered samples: {len(green_buffer)}")
        print(
            "Session summary: "
            f"Max BPM={session_stats['max_bpm']} | "
            f"Min BPM={'--' if session_stats['min_bpm'] == 999 else session_stats['min_bpm']} | "
            f"Readings={session_stats['readings_count']}"
        )


def main() -> None:
    """Program entry point for the VitalSense monitor.

    Args:
        None.

    Returns:
        None.
    """
    run()


atexit.register(_shutdown_resources)

if __name__ == "__main__":
    main()
