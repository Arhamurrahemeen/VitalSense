"""HRV and stress-classification utilities for VitalSense.

This module is a pure signal-analysis layer. It consumes filtered NumPy arrays
from the rPPG pipeline, extracts HRV metrics, and maps them to a simple
rule-based stress label for downstream overlays or reports.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

_STRESS_COLORS = {
    "calm": (0, 255, 0),
    "moderate_stress": (0, 255, 255),
    "high_stress": (0, 0, 255),
    "uncertain": (255, 255, 255),
}

__all__ = [
    "compute_hrv",
    "compute_lf_hf",
    "classify_stress",
    "get_stress_color",
]


def _log_hrv_error(operation: str, error: Exception, fallback: str) -> None:
    """Log HRV-module errors in the VitalSense three-part format.

    Args:
        operation: The operation that failed.
        error: The exception that caused the failure.
        fallback: The fallback strategy applied after the failure.

    Returns:
        None.
    """
    logger.exception(
        "HRV error | what failed=%s | what caused it=%s | fallback=%s",
        operation,
        error,
        fallback,
    )

def compute_lf_hf(filtered_signal: np.ndarray, fs: float = 30) -> float | None:
    """Compute LF/HF ratio from a filtered rPPG signal using Welch PSD.

    LF band: 0.04–0.15 Hz (sympathetic + parasympathetic activity).
    HF band: 0.15–0.40 Hz (parasympathetic / respiratory activity).
    Elevated LF/HF ratio is associated with sympathetic dominance (stress).

    Note: Reliable LF/HF estimation requires ~5 min of signal. At 60s,
    treat this as an approximation and report accordingly in the paper.

    Args:
        filtered_signal: One-dimensional filtered NumPy array from rPPG pipeline.
        fs: Sampling rate in Hz. Defaults to 30.

    Returns:
        LF/HF ratio rounded to 2 decimal places, or None if unreliable.
    """
    fallback = "return None because Welch PSD or band power computation failed"

    try:
        samples = np.asarray(filtered_signal, dtype=float).ravel()
        samples = samples[np.isfinite(samples)]

        if samples.size < int(fs * 30):  # Need at least 30 seconds
            raise ValueError(f"signal too short for LF/HF: {samples.size} samples")

        from scipy.signal import welch

        freqs, psd = welch(samples, fs=fs, nperseg=min(256, samples.size // 2))

        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

        lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask]))
        hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask]))

        if hf_power <= 0:
            raise ValueError("HF power is zero or negative — unreliable signal")

        return round(lf_power / hf_power, 2)

    except Exception as exc:
        _log_hrv_error("compute_lf_hf", exc, fallback)
        return None

def compute_hrv(filtered_signal: np.ndarray, fs: float = 30) -> Tuple[float | None, float | None]:
    """Compute RMSSD and SDNN from a filtered rPPG waveform.

    Zhao et al. (2023), Section 3, discusses HRV feature extraction from rPPG
    signals and supports interval-based features derived from heartbeat peaks.
    Fontes et al. (2024), Table 2, is also relevant when mapping RMSSD ranges
    to stress-related interpretation.

    Args:
        filtered_signal: One-dimensional filtered NumPy array from the rPPG
            pipeline.
        fs: Sampling rate in Hz. Defaults to 30.

    Returns:
        A tuple of (rmssd, sdnn), each rounded to 1 decimal place. Returns
        (None, None) if the signal is too short or if fewer than 3 peaks are
        detected.
    """
    fallback = "return (None, None) because the peak sequence or IBI state is not reliable"

    try:
        samples = np.asarray(filtered_signal, dtype=float).ravel()
        samples = samples[np.isfinite(samples)]

        if samples.size < 3:
            raise ValueError(f"signal length {samples.size} is too short for HRV analysis")

        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"sampling rate {fs} is invalid")

        peaks, _ = find_peaks(samples, distance=int(fs * 0.4), prominence=0.01)

        if peaks.size < 3:
            raise ValueError(f"peak count {peaks.size} is too small for reliable HRV")

        ibi_ms = np.diff(peaks) / fs * 1000.0

        if ibi_ms.size < 2:
            raise ValueError("IBI array state is too short to compute successive differences")

        ibi_diff = np.diff(ibi_ms)

        # RMSSD thresholds here are a practical starting point for stress
        # interpretation and should be calibrated against pulse oximeter data
        # for the CEP report.
        rmssd = float(np.sqrt(np.mean(ibi_diff ** 2)))
        sdnn = float(np.std(ibi_ms))

        return round(rmssd, 1), round(sdnn, 1)

    except Exception as exc:
        _log_hrv_error(
            "compute_hrv",
            exc,
            fallback,
        )
        return None, None


def classify_stress(
    bpm: float | int | None,
    rmssd: float | None,
    confidence: float | int | None,
    lf_hf: float | None = None,
) -> str:
    """Classify stress level from BPM, RMSSD, confidence, and optional LF/HF.

    Fontes et al. (2024), Table 2, provides RMSSD guidance that can inform
    rule-based stress labeling. The current thresholds are intentionally
    conservative and should be tuned against ground-truth measurements.

    Args:
        bpm: Estimated heart rate in beats per minute.
        rmssd: RMSSD value in milliseconds from compute_hrv().
        confidence: Signal confidence score from the BPM pipeline.
        lf_hf: Optional LF/HF ratio from compute_lf_hf().

    Returns:
        One of: uncertain, calm, moderate_stress, or high_stress.
    """
    fallback = "uncertain"

    try:
        if bpm is None or rmssd is None or confidence is None:
            raise ValueError("bpm, rmssd, or confidence is missing")

        bpm_value = float(bpm)
        rmssd_value = float(rmssd)
        confidence_value = float(confidence)

        if not np.isfinite(bpm_value):
            raise ValueError(f"bpm value {bpm_value} is not finite")
        if not np.isfinite(rmssd_value):
            raise ValueError(f"rmssd value {rmssd_value} is not finite")
        if not np.isfinite(confidence_value):
            raise ValueError(f"confidence value {confidence_value} is not finite")

        if confidence_value < 40:
            return "uncertain"

        # LF/HF > 2.0 with low RMSSD is a strong stress indicator.
        if lf_hf is not None and lf_hf > 2.0 and rmssd_value < 25:
            return "high_stress"

        # RMSSD thresholds (20, 30 ms) are approximate starting points from
        # literature. Calibrate against pulse oximeter readings for your CEP
        # report.
        if bpm_value > 90 and rmssd_value < 20:
            return "high_stress"

        if bpm_value > 75 or rmssd_value < 30:
            return "moderate_stress"

        return "calm"

    except Exception as exc:
        _log_hrv_error(
            "classify_stress",
            exc,
            f"return '{fallback}' because bpm, rmssd, or confidence is not reliable",
        )
        return fallback


def get_stress_color(status: str) -> Tuple[int, int, int]:
    """Return the OpenCV BGR color for a stress status.

    Args:
        status: Stress label name. Expected values are calm, moderate_stress,
            high_stress, and uncertain.

    Returns:
        A BGR tuple suitable for OpenCV drawing operations.
    """
    fallback = "uncertain"

    try:
        if not isinstance(status, str):
            raise TypeError(f"status must be a string, got {type(status).__name__}")

        if status not in _STRESS_COLORS:
            raise ValueError(f"unknown stress status '{status}'")

        return _STRESS_COLORS[status]

    except Exception as exc:
        _log_hrv_error(
            "get_stress_color",
            exc,
            "return the gray uncertain color because the requested stress status is invalid",
        )
        return _STRESS_COLORS[fallback]
