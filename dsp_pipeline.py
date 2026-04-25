"""Core DSP utilities for VitalSense.

This module stays strictly focused on NumPy/SciPy signal processing:
Butterworth bandpass filter design, zero-phase filtering, FFT-based BPM
extraction, confidence scoring, and a static frequency-response plot.

The upstream VitalSense signal extractor uses the green channel because the
pulsatile rPPG component is typically strongest there (Verkruysse et al., 2008).
Poh et al. (2010) describe FFT-based heart-rate estimation from the dominant
spectral peak in the cardiac band.
"""

from __future__ import annotations

import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import signal as sp_signal

logger = logging.getLogger(__name__)

FS = 30
LOW = 0.7
HIGH = 4.0

__all__ = [
    "FS",
    "LOW",
    "HIGH",
    "design_bandpass",
    "apply_filter",
    "compute_bpm_and_confidence",
    "plot_frequency_response",
]


def _log_dsp_error(operation: str, error: Exception, fallback: str) -> None:
    """Log VitalSense DSP errors in the required three-part format.

    Args:
        operation: The operation that failed.
        error: The exception that caused the failure.
        fallback: The fallback strategy applied after the failure.

    Returns:
        None.
    """
    logger.exception(
        "DSP error | what failed=%s | what caused it=%s | fallback=%s",
        operation,
        error,
        fallback,
    )


def design_bandpass(low: float, high: float, fs: float, order: int = 2) -> np.ndarray:
    """Design a Butterworth bandpass filter in SOS form.

    SciPy best practice is to use Second-Order Sections for IIR filters because
    SOS reduces numerical instability compared with a single transfer-function
    polynomial representation, especially as filter order increases.

    Args:
        low: Low cutoff frequency in Hz.
        high: High cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Butterworth filter order.

    Returns:
        A NumPy array containing SOS filter coefficients. Returns an empty SOS
        array if filter design fails.
    """
    fallback = "return an empty SOS array so downstream code can skip filtering"

    try:
        sos = sp_signal.butter(
            order,
            [low, high],
            btype="bandpass",
            fs=fs,
            output="sos",
        )
        return np.asarray(sos, dtype=float)

    except Exception as exc:
        _log_dsp_error("design_bandpass", exc, fallback)
        return np.empty((0, 6), dtype=float)


def apply_filter(signal: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Apply zero-phase bandpass filtering to the input signal.

    The signal is detrended first by removing its mean so lighting-induced DC
    offset does not dominate the cardiac component. Zero-phase filtering uses
    sosfiltfilt instead of lfilter so the output is not time-shifted.

    Args:
        signal: One-dimensional NumPy array containing the raw rPPG signal.
        sos: SOS coefficients from design_bandpass().

    Returns:
        A filtered NumPy array. For short or invalid inputs, returns a detrended
        fallback signal rather than raising an exception.
    """
    fallback = "return the detrended signal unchanged because filtering is unsafe"

    try:
        samples = np.asarray(signal, dtype=float).ravel()

        if samples.size == 0:
            raise ValueError("signal is empty")

        detrended = samples - float(np.mean(samples))

        if samples.size < 20:
            # Short sequences do not provide enough padding for sosfiltfilt.
            return detrended

        sos_array = np.asarray(sos, dtype=float)
        if sos_array.size == 0:
            raise ValueError("SOS coefficients are empty")

        return sp_signal.sosfiltfilt(sos_array, detrended)

    except Exception as exc:
        _log_dsp_error("apply_filter", exc, fallback)
        try:
            samples = np.asarray(signal, dtype=float).ravel()
            return samples - float(np.mean(samples)) if samples.size else np.asarray([], dtype=float)
        except Exception as inner_exc:
            _log_dsp_error(
                "apply_filter fallback",
                inner_exc,
                "return an empty array because the input could not be normalized",
            )
            return np.asarray([], dtype=float)


def compute_bpm_and_confidence(
    filtered_signal: np.ndarray,
    fs: float = FS,
    low: float = LOW,
    high: float = HIGH,
) -> Tuple[int, int]:
    """Estimate BPM and confidence from a filtered rPPG signal.

    The algorithm uses np.fft.rfft and np.fft.rfftfreq to analyze the real FFT,
    restricts analysis to the cardiac band, finds the strongest spectral peak,
    and converts that peak frequency to BPM.

    Args:
        filtered_signal: One-dimensional filtered signal from apply_filter().
        fs: Sampling rate in Hz.
        low: Low cutoff frequency in Hz.
        high: High cutoff frequency in Hz.

    Returns:
        A tuple of (bpm, confidence). If no valid peak is found, returns (0, 0).
    """
    fallback = "return (0, 0) because the signal or spectrum is not reliable"

    try:
        samples = np.asarray(filtered_signal, dtype=float).ravel()
        samples = samples[np.isfinite(samples)]

        if samples.size < 4:
            raise ValueError(f"signal length {samples.size} is too short for FFT analysis")

        centered = samples - float(np.mean(samples))

        fft_values = np.fft.rfft(centered)
        fft_freqs = np.fft.rfftfreq(centered.size, d=1.0 / fs)
        magnitudes = np.abs(fft_values)
        band_mask = (fft_freqs >= low) & (fft_freqs <= high)

        if not np.any(band_mask):
            raise ValueError("no FFT bins fall inside the cardiac band")

        band_freqs = fft_freqs[band_mask]
        band_power = magnitudes[band_mask] ** 2

        if band_power.size < 2:
            raise ValueError("insufficient band resolution to determine peak confidence")

        peak_index = int(np.argmax(band_power))
        peak_power = float(band_power[peak_index])
        peak_freq = float(band_freqs[peak_index])

        if not np.isfinite(peak_freq) or peak_freq <= 0.0 or peak_power <= 0.0:
            raise ValueError("peak frequency or peak power is invalid")

        # Poh et al. (2010): estimate heart rate from the dominant FFT peak in
        # the cardiac band after band-limited spectral analysis.
        noise_candidates = np.delete(band_power, peak_index)
        noise_power = float(np.mean(noise_candidates)) if noise_candidates.size else peak_power
        noise_power = max(noise_power, np.finfo(float).eps)

        bpm_value = peak_freq * 60.0
        confidence_value = (peak_power / noise_power / 10.0) * 100.0

        bpm = int(round(bpm_value))
        confidence = int(min(100.0, round(confidence_value)))

        if bpm <= 0:
            return 0, 0

        return bpm, confidence

    except Exception as exc:
        _log_dsp_error("compute_bpm_and_confidence", exc, fallback)
        return 0, 0


def plot_frequency_response(sos: np.ndarray, fs: float) -> Figure:
    """Plot the static frequency response of a bandpass SOS filter.

    Args:
        sos: SOS coefficients from design_bandpass().
        fs: Sampling rate in Hz.

    Returns:
        A Matplotlib Figure containing the filter magnitude response.
    """
    fallback = "return a blank figure because frequency-response computation failed"

    try:
        sos_array = np.asarray(sos, dtype=float)
        fig, ax = plt.subplots(figsize=(8, 4.5))

        if sos_array.size == 0:
            raise ValueError("SOS coefficients are empty")

        # SOS frequency response is the numerically stable way to inspect an
        # IIR filter designed in second-order sections.
        freqs, response = sp_signal.sosfreqz(sos_array, worN=2048, fs=fs)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(float).tiny))

        ax.plot(freqs, magnitude_db, label="Butterworth bandpass response", linewidth=2)
        ax.axvline(LOW, color="orange", linestyle="--", label=f"Low cutoff {LOW:.1f} Hz")
        ax.axvline(HIGH, color="red", linestyle="--", label=f"High cutoff {HIGH:.1f} Hz")
        ax.axvspan(LOW, HIGH, color="green", alpha=0.12, label="Cardiac band")

        ax.set_title("VitalSense DSP Bandpass Frequency Response")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    except Exception as exc:
        _log_dsp_error("plot_frequency_response", exc, fallback)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.set_title("VitalSense DSP Bandpass Frequency Response")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
