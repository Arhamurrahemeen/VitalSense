"""Signal extraction helpers for VitalSense.

This module is import-safe and contains only reusable signal-extraction
utilities: MediaPipe face landmarker creation, forehead ROI extraction, green
channel sampling, and a small frame-analysis helper.

It does not open the webcam, start a dashboard, or run a main loop at import
time. That behavior belongs in `main.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import mediapipe.tasks as tasks
import numpy as np

from dsp_pipeline import FS

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 15.0
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
MIN_FOREHEAD_HEIGHT = 20
MAX_FOREHEAD_HEIGHT = 180
DEFAULT_MODEL_PATH = "face_landmarker.task"

# Stable upper-face landmark set for the forehead ROI.
FOREHEAD_IDS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379]

__all__ = [
    "WINDOW_SECONDS",
    "WINDOW_SAMPLES",
    "MIN_FOREHEAD_HEIGHT",
    "MAX_FOREHEAD_HEIGHT",
    "FOREHEAD_IDS",
    "SignalExtractionResult",
    "create_face_landmarker",
    "build_mp_image",
    "extract_forehead_roi",
    "get_green_signal",
    "classify_distance_from_bbox",
    "process_frame",
]


@dataclass(slots=True)
class SignalExtractionResult:
    """Container for one frame of signal-extraction output.

    Args:
        has_face: True when a face was detected.
        roi: Cropped forehead ROI in BGR format, or None.
        bounding_box: ROI bounding box as (x, y, width, height), or None.
        landmark_points: Forehead landmark points in pixel coordinates.
        green_mean: Mean green-channel intensity from the ROI.
        distance_status: Distance classification label.
        error: Optional error text when extraction fails.

    Returns:
        None.

    Raises:
        None.
    """

    has_face: bool
    roi: np.ndarray | None
    bounding_box: tuple[int, int, int, int] | None
    landmark_points: list[tuple[int, int]]
    green_mean: float
    distance_status: str
    error: str | None = None


def _log_signal_error(operation: str, error: Exception, fallback: str) -> None:
    """Log signal-extraction errors in the VitalSense three-part format.

    Args:
        operation: The operation that failed.
        error: The exception that caused the failure.
        fallback: The fallback strategy applied after the failure.

    Returns:
        None.

    Raises:
        None.
    """
    logger.error(
        "Signal extraction error | what failed=%s | what caused it=%s | fallback=%s",
        operation,
        error,
        fallback,
        exc_info=False,
    )
    print(
        f"[ERROR] {operation}: {type(error).__name__}: {error} | "
        f"fallback={fallback}"
    )


def _is_valid_frame(frame: Any) -> bool:
    """Check whether a frame looks like a valid OpenCV image.

    Args:
        frame: Candidate frame object.

    Returns:
        True when the frame is a non-empty NumPy array; otherwise False.

    Raises:
        None.
    """
    try:
        return isinstance(frame, np.ndarray) and frame.size > 0 and frame.ndim >= 2
    except Exception:
        return False


def _collect_forehead_points(
    frame: np.ndarray,
    landmarks: Any,
    forehead_ids: list[int] | tuple[int, ...] = FOREHEAD_IDS,
) -> list[tuple[int, int]]:
    """Convert normalized MediaPipe landmarks into pixel coordinates.

    Args:
        frame: Current BGR frame.
        landmarks: MediaPipe landmark list.
        forehead_ids: Landmark indices used to define the forehead ROI.

    Returns:
        A list of (x, y) pixel coordinates. Returns an empty list when
        conversion fails.

    Raises:
        None.
    """
    fallback = "return an empty landmark list because coordinate conversion failed"

    try:
        if not _is_valid_frame(frame):
            raise ValueError("frame is empty or invalid")
        if landmarks is None:
            raise ValueError("landmarks are missing")

        height, width = frame.shape[:2]
        points: list[tuple[int, int]] = []

        for index in forehead_ids:
            landmark = landmarks[index]
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if not np.isfinite(x) or not np.isfinite(y):
                continue

            points.append((x, y))

        return points

    except Exception as exc:
        _log_signal_error("_collect_forehead_points", exc, fallback)
        return []


def create_face_landmarker(
    model_asset_path: str = DEFAULT_MODEL_PATH,
    num_faces: int = 1,
    min_face_detection_confidence: float = 0.5,
    min_face_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Any | None:
    """Create a MediaPipe Face Landmarker using the Tasks API.

    Args:
        model_asset_path: Path to the `.task` face-landmarker model.
        num_faces: Maximum number of faces to detect.
        min_face_detection_confidence: Detection confidence threshold.
        min_face_presence_confidence: Face presence confidence threshold.
        min_tracking_confidence: Tracking confidence threshold.

    Returns:
        A configured FaceLandmarker instance, or None if initialization fails.

    Raises:
        None.
    """
    fallback = "return None because MediaPipe initialization failed"

    try:
        model_path = Path(model_asset_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model asset not found: {model_asset_path}")

        BaseOptions = tasks.BaseOptions
        FaceLandmarker = tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        return FaceLandmarker.create_from_options(options)

    except Exception as exc:
        _log_signal_error("create_face_landmarker", exc, fallback)
        return None


def build_mp_image(frame: np.ndarray) -> mp.Image | None:
    """Convert a BGR OpenCV frame into a MediaPipe SRGB image.

    Args:
        frame: Current BGR webcam frame.

    Returns:
        A MediaPipe Image in SRGB format, or None if conversion fails.

    Raises:
        None.
    """
    fallback = "return None because the frame could not be converted to MediaPipe"

    try:
        if not _is_valid_frame(frame):
            raise ValueError("frame is empty or invalid")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    except Exception as exc:
        _log_signal_error("build_mp_image", exc, fallback)
        return None


def extract_forehead_roi(
    frame: np.ndarray,
    landmarks: Any,
    forehead_ids: list[int] | tuple[int, ...] = FOREHEAD_IDS,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Extract a forehead ROI from MediaPipe face landmarks.

    Args:
        frame: Current BGR webcam frame.
        landmarks: MediaPipe landmark list for one detected face.
        forehead_ids: Landmark indices used to build the forehead region.

    Returns:
        A tuple of (roi, bounding_box). Returns (None, None) when extraction
        fails or the ROI is degenerate.

    Raises:
        None.
    """
    fallback = "return (None, None) because the ROI could not be computed"

    try:
        points = _collect_forehead_points(frame, landmarks, forehead_ids)
        if len(points) < 3:
            raise ValueError(f"insufficient forehead landmarks: {len(points)}")

        points_array = np.asarray(points, dtype=np.int32)
        x, y, width, height = cv2.boundingRect(points_array)

        frame_height, frame_width = frame.shape[:2]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(frame_width, x + width)
        y1 = min(frame_height, y + height)

        if x1 <= x0 or y1 <= y0:
            return None, None

        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None, None

        return roi, (x0, y0, x1 - x0, y1 - y0)

    except Exception as exc:
        _log_signal_error("extract_forehead_roi", exc, fallback)
        return None, None


def get_green_signal(roi: np.ndarray | None) -> float:
    """Compute the mean green-channel intensity for a forehead ROI.

    Args:
        roi: Cropped forehead region in BGR format.

    Returns:
        Mean green-channel intensity, or 0.0 when the ROI is invalid.

    Raises:
        None.
    """
    fallback = "return 0.0 because the ROI is invalid"

    try:
        if roi is None or not isinstance(roi, np.ndarray) or roi.size == 0:
            raise ValueError("ROI is empty or invalid")

        if roi.ndim != 3 or roi.shape[2] < 2:
            raise ValueError(f"ROI shape {roi.shape} does not contain a green channel")

        return float(np.mean(roi[:, :, 1]))

    except Exception as exc:
        _log_signal_error("get_green_signal", exc, fallback)
        return 0.0


def classify_distance_from_bbox(
    bounding_box: tuple[int, int, int, int] | None,
) -> str:
    """Classify face distance from the forehead ROI height.

    Args:
        bounding_box: ROI bounding box as (x, y, width, height), or None.

    Returns:
        One of: MOVE CLOSER, TOO CLOSE, DISTANCE OK, or N/A.

    Raises:
        None.
    """
    fallback = "N/A"

    try:
        if bounding_box is None:
            return fallback

        _, _, _, height = bounding_box

        if height < MIN_FOREHEAD_HEIGHT:
            return "MOVE CLOSER"
        if height > MAX_FOREHEAD_HEIGHT:
            return "TOO CLOSE"
        return "DISTANCE OK"

    except Exception as exc:
        _log_signal_error(
            "classify_distance_from_bbox",
            exc,
            "return 'N/A' because the bounding-box state is invalid",
        )
        return fallback


def process_frame(
    frame: np.ndarray,
    face_landmarker: Any,
    timestamp_ms: int,
) -> SignalExtractionResult:
    """Process one webcam frame into ROI, signal, and distance metadata.

    Args:
        frame: Current BGR webcam frame.
        face_landmarker: Initialized MediaPipe Face Landmarker instance.
        timestamp_ms: Monotonic timestamp in milliseconds.

    Returns:
        A SignalExtractionResult with safe defaults on failure.

    Raises:
        None.
    """
    fallback = "return a safe empty result because frame processing failed"

    try:
        if not _is_valid_frame(frame):
            raise ValueError("frame is empty or invalid")
        if face_landmarker is None:
            raise ValueError("face_landmarker is not initialized")

        mp_image = build_mp_image(frame)
        if mp_image is None:
            raise ValueError("MediaPipe image conversion failed")

        result = face_landmarker.detect_for_video(mp_image, int(timestamp_ms))
        if not result or not getattr(result, "face_landmarks", None):
            return SignalExtractionResult(
                has_face=False,
                roi=None,
                bounding_box=None,
                landmark_points=[],
                green_mean=0.0,
                distance_status="N/A",
                error=None,
            )

        landmarks = result.face_landmarks[0]
        roi, bounding_box = extract_forehead_roi(frame, landmarks)
        landmark_points = _collect_forehead_points(frame, landmarks)
        green_mean = get_green_signal(roi)
        distance_status = classify_distance_from_bbox(bounding_box)

        return SignalExtractionResult(
            has_face=roi is not None,
            roi=roi,
            bounding_box=bounding_box,
            landmark_points=landmark_points,
            green_mean=green_mean,
            distance_status=distance_status,
            error=None,
        )

    except Exception as exc:
        _log_signal_error("process_frame", exc, fallback)
        return SignalExtractionResult(
            has_face=False,
            roi=None,
            bounding_box=None,
            landmark_points=[],
            green_mean=0.0,
            distance_status="N/A",
            error=str(exc),
        )
