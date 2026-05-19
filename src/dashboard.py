"""VitalSense real-time Matplotlib dashboard.

This module is visualization-only. It reads shared state produced by the
VitalSense main orchestrator and renders raw, filtered, and spectral views of
the current green-channel signal.

The dashboard intentionally does not touch OpenCV, MediaPipe, or Groq APIs.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from dsp_pipeline import FS, HIGH, LOW, apply_filter, compute_bpm_and_confidence, design_bandpass

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 15.0
DEFAULT_INTERVAL_MS = 500
CARDIAC_BAND_LOW_HZ = 0.7
CARDIAC_BAND_HIGH_HZ = 4.0

__all__ = [
    "create_shared_state",
    "DashboardApp",
    "start_dashboard",
    "run_dashboard",
]


def _log_dashboard_error(operation: str, error: Exception, fallback: str) -> None:
    """Log dashboard errors in the VitalSense three-part format.

    Args:
        operation: The operation that failed.
        error: The exception that caused the failure.
        fallback: The fallback strategy applied after the failure.

    Returns:
        None.
    """
    logger.error(
        "Dashboard error | what failed=%s | what caused it=%s | fallback=%s",
        operation,
        error,
        fallback,
        exc_info=False,
    )
    print(
        f"[ERROR] {operation}: {type(error).__name__}: {error} | "
        f"fallback={fallback}"
    )


def create_shared_state(window_samples: int) -> dict[str, Any]:
    """Create the shared dictionary used by the main loop and dashboard.

    Args:
        window_samples: Maximum number of samples retained in the rolling buffer.

    Returns:
        A dictionary containing the shared buffer, metrics, and synchronization
        primitives required by the dashboard and main orchestrator.

    Raises:
        ValueError: If window_samples is not positive.
    """
    fallback = "return an empty shared-state dictionary so the caller can recover"

    try:
        samples = int(window_samples)
        if samples <= 0:
            raise ValueError(f"window_samples must be positive, got {window_samples!r}")

        return {
            "lock": threading.Lock(),
            "green_buffer": deque(maxlen=samples),
            "timestamps": deque(maxlen=samples),
            "raw_buffer": [],
            "filtered_buffer": [],
            "fft_freqs": [],
            "fft_mags": [],
            "bpm": 0,
            "confidence": 0,
            "stress": "uncertain",
            "timestamp": 0.0,
            "latest_bpm": 0,
            "latest_confidence": 0,
            "latest_stress": "uncertain",
            "latest_feedback": "Initializing...",
            "latest_distance": "DISTANCE OK",
            "latest_lighting": "good",
            "latest_luminance": 0.0,
            "latest_fps": 0.0,
            "latest_filtered_signal": np.asarray([], dtype=float),
            "latest_session_stats": {
                "max_bpm": 0,
                "min_bpm": 999,
                "readings_count": 0,
            },
            "measurement_state": "collecting",
            "session_start_time": None,
            "last_feedback_time": 0.0,
            "shutdown_event": threading.Event(),
        }

    except Exception as exc:
        _log_dashboard_error("create_shared_state", exc, fallback)
        return {
            "lock": threading.Lock(),
            "green_buffer": deque(maxlen=1),
            "timestamps": deque(maxlen=1),
            "raw_buffer": [],
            "filtered_buffer": [],
            "fft_freqs": [],
            "fft_mags": [],
            "bpm": 0,
            "confidence": 0,
            "stress": "uncertain",
            "timestamp": 0.0,
            "latest_bpm": 0,
            "latest_confidence": 0,
            "latest_stress": "uncertain",
            "latest_feedback": "Initializing...",
            "latest_distance": "DISTANCE OK",
            "latest_lighting": "good",
            "latest_luminance": 0.0,
            "latest_fps": 0.0,
            "latest_filtered_signal": np.asarray([], dtype=float),
            "latest_session_stats": {
                "max_bpm": 0,
                "min_bpm": 999,
                "readings_count": 0,
            },
            "measurement_state": "collecting",
            "session_start_time": None,
            "last_feedback_time": 0.0,
            "shutdown_event": threading.Event(),
        }


class DashboardApp:
    """Render a live three-panel visualization for VitalSense."""

    def __init__(
        self,
        shared_state: dict[str, Any],
        fs: float = FS,
        window_seconds: float = WINDOW_SECONDS,
    ) -> None:
        """Initialize the dashboard controller.

        Args:
            shared_state: Shared dictionary populated by the main orchestrator.
            fs: Sampling rate in Hz.
            window_seconds: Rolling display window in seconds.

        Returns:
            None.

        Raises:
            ValueError: If the shared state is missing required keys.
        """
        fallback = "delay dashboard startup until shared state is valid"

        try:
            required_keys = {"lock", "green_buffer", "shutdown_event"}
            missing_keys = required_keys.difference(shared_state.keys())
            if missing_keys:
                raise ValueError(f"shared_state is missing keys: {sorted(missing_keys)}")

            self.shared_state = shared_state
            self.fs = float(fs)
            self.window_seconds = float(window_seconds)

            green_buffer = shared_state["green_buffer"]
            buffer_window = int(getattr(green_buffer, "maxlen", 0) or 0)
            self.window_samples = buffer_window if buffer_window > 0 else max(1, int(self.fs * self.window_seconds))

            self.bandpass_sos = design_bandpass(LOW, HIGH, self.fs, order=2)

            self.figure: plt.Figure | None = None
            self.axes_raw = None
            self.axes_filtered = None
            self.axes_fft = None
            self.raw_line = None
            self.filtered_line = None
            self.fft_bar_container = None
            self.raw_placeholder = None
            self.filtered_placeholder = None
            self.fft_placeholder = None
            self.footer_text: Any | None = None
            self.animation: animation.FuncAnimation | None = None

        except Exception as exc:
            _log_dashboard_error("DashboardApp.__init__", exc, fallback)
            self.shared_state = shared_state
            self.fs = float(fs)
            self.window_seconds = float(window_seconds)
            self.window_samples = max(1, int(self.fs * self.window_seconds))
            self.bandpass_sos = design_bandpass(LOW, HIGH, self.fs, order=2)
            self.figure = None
            self.axes_raw = None
            self.axes_filtered = None
            self.axes_fft = None
            self.raw_line = None
            self.filtered_line = None
            self.fft_line = None
            self.raw_placeholder = None
            self.filtered_placeholder = None
            self.fft_placeholder = None
            self.footer_text = None
            self.animation = None

    def _snapshot_state(self) -> dict[str, Any]:
        """Copy the current shared state under lock for plotting.

        Args:
            None.

        Returns:
            A dictionary containing immutable snapshots of the current signal and
            metrics.

        Raises:
            RuntimeError: If the shared lock cannot be acquired.
        """
        fallback = "return an empty snapshot so the UI can continue updating"

        try:
            lock = self.shared_state["lock"]
            with lock:
                # Prefer raw_buffer if it contains data, otherwise fall back to green_buffer
                raw_data = self.shared_state.get("raw_buffer", [])
                if not isinstance(raw_data, (np.ndarray, list)) or len(raw_data) == 0:
                    raw_data = self.shared_state.get("green_buffer", [])
                raw_buffer = np.asarray(raw_data, dtype=float)

                filtered_buffer = np.asarray(self.shared_state.get("filtered_buffer", []), dtype=float)
                fft_freqs = np.asarray(self.shared_state.get("fft_freqs", []), dtype=float)
                fft_mags = np.asarray(self.shared_state.get("fft_mags", []), dtype=float)
                latest_bpm = int(self.shared_state.get("latest_bpm", 0) or 0)
                latest_confidence = int(self.shared_state.get("latest_confidence", 0) or 0)
                latest_stress = str(self.shared_state.get("latest_stress", "uncertain"))
                latest_feedback = str(self.shared_state.get("latest_feedback", ""))
                latest_distance = str(self.shared_state.get("latest_distance", "DISTANCE OK"))
                latest_lighting = str(self.shared_state.get("latest_lighting", "good"))
                latest_luminance = float(self.shared_state.get("latest_luminance", 0.0) or 0.0)
                latest_fps = float(self.shared_state.get("latest_fps", 0.0) or 0.0)
                latest_session_stats = dict(
                    self.shared_state.get(
                        "latest_session_stats",
                        {"max_bpm": 0, "min_bpm": 999, "readings_count": 0},
                    )
                )

            return {
                "raw_buffer": raw_buffer,
                "filtered_buffer": filtered_buffer,
                "fft_freqs": fft_freqs,
                "fft_mags": fft_mags,
                "latest_bpm": latest_bpm,
                "latest_confidence": latest_confidence,
                "latest_stress": latest_stress,
                "latest_feedback": latest_feedback,
                "latest_distance": latest_distance,
                "latest_lighting": latest_lighting,
                "latest_luminance": latest_luminance,
                "latest_fps": latest_fps,
                "latest_session_stats": latest_session_stats,
            }

        except Exception as exc:
            _log_dashboard_error("DashboardApp._snapshot_state", exc, fallback)
            return {
                "raw_buffer": np.asarray([], dtype=float),
                "filtered_buffer": np.asarray([], dtype=float),
                "fft_freqs": np.asarray([], dtype=float),
                "fft_mags": np.asarray([], dtype=float),
                "latest_bpm": 0,
                "latest_confidence": 0,
                "latest_stress": "uncertain",
                "latest_feedback": "",
                "latest_distance": "DISTANCE OK",
                "latest_lighting": "good",
                "latest_luminance": 0.0,
                "latest_fps": 0.0,
                "latest_session_stats": {"max_bpm": 0, "min_bpm": 999, "readings_count": 0},
            }

    @staticmethod
    def _format_feedback(text: str) -> str:
        """Trim feedback text for the lower overlay area.

        Args:
            text: Full feedback message from the AI feedback manager.

        Returns:
            A single-line summary suitable for the dashboard footer.
        """
        fallback = "Feedback unavailable"

        try:
            feedback = " ".join(str(text).split())
            if not feedback:
                return fallback

            if len(feedback) <= 110:
                return feedback

            return f"{feedback[:107].rstrip()}..."

        except Exception as exc:
            _log_dashboard_error("DashboardApp._format_feedback", exc, fallback)
            return fallback

    def _compute_filtered_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """Filter the buffered signal for the second panel.

        Args:
            raw_signal: Raw green-channel samples.

        Returns:
            A detrended and bandpass-filtered NumPy array.
        """
        fallback = "return the raw buffer after detrending because filtering failed"

        try:
            if raw_signal.size == 0:
                return np.asarray([], dtype=float)

            return apply_filter(raw_signal, self.bandpass_sos)

        except Exception as exc:
            _log_dashboard_error("DashboardApp._compute_filtered_signal", exc, fallback)
            try:
                samples = np.asarray(raw_signal, dtype=float).ravel()
                return samples - float(np.mean(samples)) if samples.size else np.asarray([], dtype=float)
            except Exception as inner_exc:
                _log_dashboard_error(
                    "DashboardApp._compute_filtered_signal fallback",
                    inner_exc,
                    "return an empty array because the fallback detrend failed",
                )
                return np.asarray([], dtype=float)

    def _compute_fft(self, filtered_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the frequency spectrum for the third panel.

        Args:
            filtered_signal: Filtered rPPG signal.

        Returns:
            A tuple of (frequencies, magnitudes).
        """
        fallback = "return empty arrays because the spectrum could not be computed"

        try:
            samples = np.asarray(filtered_signal, dtype=float).ravel()
            samples = samples[np.isfinite(samples)]

            if samples.size < 4:
                return np.asarray([], dtype=float), np.asarray([], dtype=float)

            centered = samples - float(np.mean(samples))
            frequencies = np.fft.rfftfreq(centered.size, d=1.0 / self.fs)
            magnitudes = np.abs(np.fft.rfft(centered))
            return frequencies, magnitudes

        except Exception as exc:
            _log_dashboard_error("DashboardApp._compute_fft", exc, fallback)
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

    def _setup_figure(self) -> None:
        """Create axes, line artists, and placeholder text once.

        Returns:
            None.
        """
        fallback = "leave the dashboard uninitialized if figure setup fails"

        try:
            if self.figure is None or self.axes_raw is None or self.axes_filtered is None or self.axes_fft is None:
                raise ValueError("figure or axes are not initialized")

            self.figure.suptitle("VitalSense — Real-Time Monitor", fontsize=14, fontweight="bold")

            for axis in (self.axes_raw, self.axes_filtered, self.axes_fft):
                axis.grid(True, alpha=0.25)

            self.axes_raw.set_title("Raw green channel G(t)")
            self.axes_raw.set_ylabel("Amplitude")
            self.axes_raw.set_xlim(0.0, self.window_seconds)

            self.axes_filtered.set_ylabel("Amplitude")
            self.axes_filtered.set_xlim(0.0, self.window_seconds)

            self.axes_fft.set_xlabel("Frequency (Hz)")
            self.axes_fft.set_ylabel("Magnitude")
            self.axes_fft.set_xlim(0.0, 5.0)
            self.axes_fft.axvspan(
                CARDIAC_BAND_LOW_HZ,
                CARDIAC_BAND_HIGH_HZ,
                color="#90EE90",
                alpha=0.18,
                label="Cardiac band",
            )

            (self.raw_line,) = self.axes_raw.plot([], [], color="gray", lw=1.0)
            (self.filtered_line,) = self.axes_filtered.plot([], [], color="#378ADD", lw=1.5)
            (self.fft_line,) = self.axes_fft.plot([], [], color="#1D9E75", lw=1.5)

            placeholder_style = dict(
                ha="center",
                va="center",
                transform=self.axes_raw.transAxes,
                color="#666666",
            )
            self.raw_placeholder = self.axes_raw.text(0.5, 0.5, "Waiting for samples...", **placeholder_style)
            self.filtered_placeholder = self.axes_filtered.text(0.5, 0.5, "Waiting for samples...", **placeholder_style)
            self.fft_placeholder = self.axes_fft.text(0.5, 0.5, "Waiting for samples...", **placeholder_style)

            self.raw_placeholder.set_visible(False)
            self.filtered_placeholder.set_visible(False)
            self.fft_placeholder.set_visible(False)

        except Exception as exc:
            _log_dashboard_error("DashboardApp._setup_figure", exc, fallback)

    def _update_figure(self, _frame_index: int) -> None:
        """Refresh all dashboard axes.

        Args:
            _frame_index: Animation frame index supplied by FuncAnimation.

        Returns:
            None.
        """
        fallback = "keep the previous dashboard frame because the new draw failed"

        try:
            if self.figure is None or self.axes_raw is None or self.axes_filtered is None or self.axes_fft is None:
                return

            snapshot = self._snapshot_state()
            raw_signal = np.asarray(snapshot["raw_buffer"], dtype=float)[-150:]
            filtered_signal = np.asarray(snapshot.get("filtered_buffer", []), dtype=float)
            fft_freqs = np.asarray(snapshot.get("fft_freqs", []), dtype=float)
            fft_mags = np.asarray(snapshot.get("fft_mags", []), dtype=float)

            latest_bpm = snapshot["latest_bpm"]
            latest_confidence = snapshot["latest_confidence"]
            latest_feedback = self._format_feedback(snapshot["latest_feedback"])
            latest_distance = snapshot["latest_distance"]
            latest_lighting = snapshot["latest_lighting"]
            latest_luminance = snapshot["latest_luminance"]
            latest_fps = snapshot["latest_fps"]
            latest_stress = snapshot["latest_stress"]

            display_bpm = latest_bpm
            display_confidence = latest_confidence

            if raw_signal.size >= 4:
                if filtered_signal.size < 4:
                    filtered_signal = self._compute_filtered_signal(raw_signal)
                else:
                    filtered_signal = filtered_signal[-150:]

                if (display_bpm <= 0 or display_confidence <= 0) and filtered_signal.size >= 4:
                    bpm_estimate, confidence_estimate = compute_bpm_and_confidence(
                        filtered_signal,
                        fs=self.fs,
                        low=LOW,
                        high=HIGH,
                    )
                    display_bpm = display_bpm or bpm_estimate
                    display_confidence = display_confidence or confidence_estimate

                raw_time_axis = np.arange(raw_signal.size, dtype=float) / self.fs
                filtered_time_axis = np.arange(filtered_signal.size, dtype=float) / self.fs
                if fft_freqs.size >= 1 and fft_mags.size == fft_freqs.size:
                    frequencies, magnitudes = fft_freqs, fft_mags
                else:
                    frequencies, magnitudes = self._compute_fft(filtered_signal)

                self.raw_line.set_data(raw_time_axis, raw_signal)
                self.filtered_line.set_data(filtered_time_axis, filtered_signal)
                self.fft_line.set_data(frequencies, magnitudes)

                self.raw_placeholder.set_visible(False)
                self.filtered_placeholder.set_visible(False)
                self.fft_placeholder.set_visible(False)

                self.axes_raw.relim()
                self.axes_raw.autoscale_view(scalex=False, scaley=True)

                self.axes_filtered.relim()
                self.axes_filtered.autoscale_view(scalex=False, scaley=True)

                self.axes_fft.relim()
                self.axes_fft.autoscale_view(scalex=False, scaley=True)
                self.axes_fft.set_xlim(0.0, 5.0)
            else:
                empty = np.asarray([], dtype=float)
                self.raw_line.set_data(empty, empty)
                self.filtered_line.set_data(empty, empty)
                self.fft_line.set_data(empty, empty)

                self.raw_placeholder.set_visible(True)
                self.filtered_placeholder.set_visible(True)
                self.fft_placeholder.set_visible(True)

            self.axes_raw.set_title("Raw green channel G(t)")
            self.axes_filtered.set_title(
                f"Filtered signal | BPM: {display_bpm if display_bpm > 0 else '--'} | "
                f"Conf: {display_confidence if display_confidence > 0 else '--'}%"
            )
            self.axes_fft.set_title("FFT spectrum")

            stats = snapshot["latest_session_stats"]
            max_bpm = stats.get("max_bpm", 0)
            min_bpm = stats.get("min_bpm", 999)
            min_text = "--" if min_bpm == 999 else str(min_bpm)

            footer = (
                f"FPS: {latest_fps:.1f} | Lighting: {latest_lighting} ({latest_luminance:.1f}) | "
                f"Distance: {latest_distance} | Stress: {latest_stress} | "
                f"Max BPM: {max_bpm} | Min BPM: {min_text}"
            )
            footer_text = f"{footer}\nFeedback: {latest_feedback}"

            if self.footer_text is None:
                self.footer_text = self.figure.text(
                    0.01,
                    0.01,
                    footer_text,
                    fontsize=8,
                    va="bottom",
                    ha="left",
                )
            else:
                self.footer_text.set_text(footer_text)

            self.figure.canvas.draw_idle()

        except Exception as exc:
            _log_dashboard_error("DashboardApp._update_figure", exc, fallback)

    def run(self) -> None:
        """Start the Matplotlib dashboard loop.

        Args:
            None.

        Returns:
            None.
        """
        fallback = "close the figure if the UI loop cannot start"

        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            self.figure, axes = plt.subplots(3, 1, figsize=(7, 7), constrained_layout=True)
            self.axes_raw, self.axes_filtered, self.axes_fft = axes

            self._setup_figure()

            # Position the dashboard window to avoid overlapping with the camera
            mngr = plt.get_current_fig_manager()
            if hasattr(mngr, "window"):
                try:
                    mngr.window.wm_geometry("+700+50")
                except Exception:
                    try:
                        mngr.window.move(700, 50)
                    except Exception:
                        pass

            self.animation = animation.FuncAnimation(
                self.figure,
                self._update_figure,
                interval=DEFAULT_INTERVAL_MS,
                blit=False,
                cache_frame_data=False,
            )

            plt.show()

        except KeyboardInterrupt:
            self.close()

        except Exception as exc:
            _log_dashboard_error("DashboardApp.run", exc, fallback)
            self.close()

    def close(self) -> None:
        """Close the dashboard window and stop future updates.

        Args:
            None.

        Returns:
            None.
        """
        fallback = "ignore close errors because the UI may already be gone"

        try:
            shutdown_event = self.shared_state.get("shutdown_event")
            if shutdown_event is not None:
                shutdown_event.set()

            if self.figure is not None:
                plt.close(self.figure)

        except Exception as exc:
            _log_dashboard_error("DashboardApp.close", exc, fallback)


def start_dashboard(shared_state: dict[str, Any]) -> tuple[DashboardApp, threading.Thread]:
    """Launch the dashboard in a daemon thread.

    Args:
        shared_state: Shared dictionary populated by the main orchestrator.

    Returns:
        A tuple of (DashboardApp, Thread) so the caller can close the figure
        during shutdown.
    """
    fallback = "run the dashboard in the current thread if thread creation fails"

    try:
        app = DashboardApp(shared_state)
        thread = threading.Thread(target=app.run, daemon=True, name="VitalSenseDashboard")
        thread.start()
        return app, thread

    except Exception as exc:
        _log_dashboard_error("start_dashboard", exc, fallback)
        app = DashboardApp(shared_state)
        thread = threading.Thread(target=app.run, daemon=True, name="VitalSenseDashboardFallback")
        thread.start()
        return app, thread


def run_dashboard(shared_state: dict[str, Any]) -> None:
    """Run the dashboard in the current thread.

    Args:
        shared_state: Shared dictionary populated by the main orchestrator.

    Returns:
        None.
    """
    app = DashboardApp(shared_state)
    app.run()


if __name__ == "__main__":
    demo_state = create_shared_state(int(FS * WINDOW_SECONDS))
    run_dashboard(demo_state)
