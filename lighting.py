"""Lighting quality classification utilities for VitalSense.

This module performs pre-extraction lighting assessment using YCrCb luminance
so the rPPG pipeline can reject poor frames before signal extraction.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_LIGHTING_COLORS = {
    "good": (0, 200, 100),
    "low_light": (0, 100, 255),
    "backlit": (0, 165, 255),
    "flicker": (0, 0, 255),
}

# Poor illumination, glare, and unstable exposure degrade rPPG signal quality
# by weakening the facial blood-volume pulse component and increasing noise,
# which is why VitalSense checks lighting before signal extraction.
# Reference: npj Digital Medicine (2025).


def _log_lighting_error(operation: str, error: Exception, fallback: str) -> None:
    """Log lighting-module errors in the VitalSense three-part format.

    Args:
        operation: The operation that failed.
        error: The exception that caused the failure.
        fallback: The fallback strategy applied after the failure.

    Returns:
        None.
    """
    logger.exception(
        "Lighting error | what failed=%s | what caused it=%s | fallback=%s",
        operation,
        error,
        fallback,
    )


def get_lighting_color(status: str) -> Tuple[int, int, int]:
    """Return the OpenCV BGR color for a lighting status.

    Args:
        status: Lighting status name. Expected values are good, low_light,
            backlit, and flicker.

    Returns:
        A BGR tuple suitable for OpenCV drawing operations.
    """
    return _LIGHTING_COLORS.get(status, _LIGHTING_COLORS["good"])


def classify_lighting(
    frame: np.ndarray,
    prev_means: List[float],
    window: int = 30,
) -> Tuple[str, float]:
    """Classify frame lighting quality before signal extraction.

    The function uses YCrCb luminance, updates the caller-managed history list
    in place, and returns a single lighting label plus the current mean
    luminance.

    Args:
        frame: A BGR image frame from OpenCV.
        prev_means: Caller-managed list of prior mean luminance values. This
            function mutates the list in place to maintain the temporal window.
        window: Number of recent frames to retain for flicker detection.

    Returns:
        A tuple of (status, mean_luminance), where status is one of:
        good, low_light, backlit, or flicker.
    """
    fallback_status = "good"
    fallback_mean = 0.0

    try:
        if frame is None or frame.size == 0:
            raise ValueError("frame is empty or None")

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]

        mean_luminance = float(np.mean(y_channel))
        std_luminance = float(np.std(y_channel))

        prev_means.append(mean_luminance)
        if window > 0 and len(prev_means) > window:
            del prev_means[:-window]

        temporal_window = prev_means[-window:] if window > 0 else prev_means
        temporal_var = float(np.var(temporal_window)) if len(temporal_window) >= 2 else 0.0

        if mean_luminance < 60:
            return "low_light", mean_luminance
        if std_luminance > 80:
            return "backlit", mean_luminance
        if temporal_var > 25:
            return "flicker", mean_luminance

        return "good", mean_luminance

    except Exception as exc:
        _log_lighting_error(
            "classify_lighting",
            exc,
            f"return ('{fallback_status}', {fallback_mean}) and preserve downstream processing",
        )
        return fallback_status, fallback_mean
