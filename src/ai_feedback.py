"""AI-powered health feedback utilities for VitalSense.

This module is intentionally pure integration logic: it loads the Groq API
key from the environment, formats a prompt from real-time readings, and sends
asynchronous feedback requests on a background thread. It does not contain any
OpenCV code, signal processing, or visualization logic.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback for environments without python-dotenv
    load_dotenv = None

from openai import OpenAI
import httpx

logger = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_WORDS = 60
CALL_INTERVAL_SECONDS = 60

SYSTEM_PROMPT = """You are a health monitoring assistant.
You receive physiological data from a webcam-based heart rate monitor.
Your rules:
1. Acknowledge the user's current state in one sentence.
2. If stress is detected, suggest ONE breathing or relaxation technique with clear steps.
3. If readings are elevated for >5 minutes, recommend speaking to a healthcare professional.
4. NEVER diagnose conditions. NEVER fabricate data.
5. Keep response under 60 words.
6. If confidence is below 40%, ask user to improve lighting instead of giving health advice."""

_FEEDBACK_KEY = "GROQ_API_KEY"
_LOW_QUALITY_MESSAGE = "Signal quality too low for health feedback. Improve lighting and stay still."
_API_KEY_MESSAGE = "API key not configured. Set GROQ_API_KEY in .env"
_RATE_LIMIT_MESSAGE = "Groq Quota Exceeded. Retrying in next window..."
_RATE_LIMIT_COOLDOWN_SECONDS = 300.0
_FALLBACK_MESSAGE = "Feedback unavailable — check connection"

__all__ = [
    "MODEL",
    "MAX_WORDS",
    "CALL_INTERVAL_SECONDS",
    "SYSTEM_PROMPT",
    "create_feedback_message",
    "get_llm_feedback",
    "FeedbackManager",
]


def _log_ai_error(operation: str, error: Exception, fallback: str) -> None:
    """Log AI-feedback errors in the VitalSense three-part format.

    Args:
        operation: The operation that failed.
        error: The exception that caused the failure.
        fallback: The fallback strategy applied after the failure.

    Returns:
        None.
    """
    logger.error(
        "AI feedback error | what failed=%s | what caused it=%s | fallback=%s",
        operation,
        error,
        fallback,
        exc_info=False,
    )


def _load_environment() -> None:
    """Load environment variables from `.env` using python-dotenv when available.

    Args:
        None.

    Returns:
        None.
    """
    try:
        if load_dotenv is not None:
            load_dotenv()
            return

        env_path = Path(".env")
        if not env_path.exists():
            return

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    except Exception as exc:
        _log_ai_error(
            "_load_environment",
            exc,
            "continue with the current process environment because .env loading failed",
        )


def _build_client() -> OpenAI | None:
    """Build a Groq-compatible OpenAI client from the current environment.

    Args:
        None.

    Returns:
        A configured `OpenAI` client when `GROQ_API_KEY` is available; otherwise
        `None`.
    """
    try:
        _load_environment()
        api_key = os.getenv(_FEEDBACK_KEY, "").strip()
        if not api_key:
            raise ValueError(f"{_FEEDBACK_KEY} is missing or empty")

        try:
            return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, httpx_client=httpx.Client(proxies=None))
        except TypeError:
            return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    except Exception as exc:
        _log_ai_error(
            "_build_client",
            exc,
            "defer API calls until GROQ_API_KEY is configured",
        )
        return None


def _truncate_words(text: str, max_words: int = MAX_WORDS) -> str:
    """Truncate text to a maximum number of words.

    Args:
        text: Input text to truncate.
        max_words: Maximum number of words to keep.

    Returns:
        The truncated text if necessary; otherwise the original text.
    """
    try:
        words = str(text).split()
        if len(words) <= max_words:
            return " ".join(words).strip()

        truncated = " ".join(words[:max_words]).strip()
        return f"{truncated}..."

    except Exception as exc:
        _log_ai_error(
            "_truncate_words",
            exc,
            "return an empty string because response truncation failed",
        )
        return ""


def _format_duration(duration_min: float | int | None) -> float:
    """Normalize session duration to a bounded floating-point value.

    Args:
        duration_min: Session duration in minutes.

    Returns:
        A float duration clipped to the range 0 to 120 minutes. Invalid values
        return 0.0.
    """
    try:
        if duration_min is None:
            return 0.0

        duration_value = float(duration_min)
        if duration_value != duration_value or duration_value in (float("inf"), float("-inf")):
            raise ValueError("duration is not finite")

        return max(0.0, min(duration_value, 120.0))

    except Exception as exc:
        _log_ai_error(
            "_format_duration",
            exc,
            "return 0.0 because the duration value is invalid",
        )
        return 0.0


def _format_optional_number(value: float | int | None, unit: str) -> str:
    """Format a numeric reading for prompt construction.

    Args:
        value: Numeric value or None.
        unit: Unit suffix to append when the value is present.

    Returns:
        A human-readable string.
    """
    try:
        if value is None:
            return "unknown"

        numeric = float(value)
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            raise ValueError("value is not finite")

        return f"{numeric:.1f} {unit}"

    except Exception as exc:
        _log_ai_error(
            "_format_optional_number",
            exc,
            "return 'unknown' because the reading could not be formatted",
        )
        return "unknown"


def _is_low_confidence(confidence: float | int | None) -> bool:
    """Check whether the signal confidence is below the safe AI-feedback threshold.

    Args:
        confidence: Signal confidence value.

    Returns:
        True when confidence is missing or below 40%; otherwise False.
    """
    try:
        if confidence is None:
            return True

        confidence_value = float(confidence)
        if confidence_value != confidence_value:
            return True

        return confidence_value < 40.0

    except Exception as exc:
        _log_ai_error(
            "_is_low_confidence",
            exc,
            "treat the signal as low confidence because the value is invalid",
        )
        return True


def _exception_text(error: Exception) -> str:
    """Return a lower-case text representation of an exception."""
    try:
        return str(error).lower()
    except Exception:
        return ""


def _is_rate_limit_error(error: Exception) -> bool:
    """Check whether an exception represents a Groq quota or rate-limit failure.

    Args:
        error: Exception raised by the Groq/OpenAI-compatible SDK.

    Returns:
        True when the error looks like a quota/rate-limit failure; otherwise False.
    """
    try:
        status_code = getattr(error, "status_code", None)
        error_text = _exception_text(error)
        class_name = error.__class__.__name__.lower()

        return (
            status_code == 429
            or "ratelimit" in class_name
            or "rate limit" in error_text
            or "429" in error_text
            or "resource_exhausted" in error_text
            or "quota" in error_text
            or "too many requests" in error_text
        )

    except Exception as exc:
        _log_ai_error(
            "_is_rate_limit_error",
            exc,
            "treat the error as non-rate-limited because inspection failed",
        )
        return False


def _is_auth_error(error: Exception) -> bool:
    """Check whether an exception looks like an authentication failure."""
    try:
        status_code = getattr(error, "status_code", None)
        error_text = _exception_text(error)
        class_name = error.__class__.__name__.lower()

        return (
            status_code in (401, 403)
            or "auth" in class_name
            or "authentication" in class_name
            or "unauthor" in error_text
            or "forbidden" in error_text
            or "invalid api key" in error_text
        )

    except Exception as exc:
        _log_ai_error(
            "_is_auth_error",
            exc,
            "treat the error as non-authenticated because inspection failed",
        )
        return False


def _is_network_error(error: Exception) -> bool:
    """Check whether an exception looks like a timeout or network failure."""
    try:
        error_text = _exception_text(error)
        class_name = error.__class__.__name__.lower()

        return (
            "timeout" in class_name
            or "connection" in class_name
            or "timeout" in error_text
            or "timed out" in error_text
            or "deadline" in error_text
            or "network" in error_text
            or "connection" in error_text
        )

    except Exception as exc:
        _log_ai_error(
            "_is_network_error",
            exc,
            "treat the error as non-network-related because inspection failed",
        )
        return False


def _extract_response_text(response: object) -> str:
    """Extract assistant text from a Groq/OpenAI chat completion response."""
    try:
        if isinstance(response, dict):
            choices = response.get("choices")
        else:
            choices = getattr(response, "choices", None)

        if not choices:
            return ""

        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
        else:
            message = getattr(first_choice, "message", None)

        if message is None:
            return ""

        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")

        if content is None:
            return ""

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = None
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                else:
                    text = getattr(item, "text", None)
                    if text is None:
                        text = getattr(item, "content", None)
                if text:
                    parts.append(str(text))
            return "\n".join(parts).strip()

        return str(content).strip()

    except Exception as exc:
        _log_ai_error(
            "_extract_response_text",
            exc,
            "return an empty string because the response body could not be parsed",
        )
        return ""


def _launch_feedback_thread(
    client: OpenAI | None,
    user_message: str,
    callback: Callable[[str], None] | None,
) -> threading.Thread | None:
    """Launch the Groq request on a daemon thread.

    Args:
        client: Configured Groq-compatible OpenAI client.
        user_message: Fully formatted user message.
        callback: Function that receives the generated feedback text.

    Returns:
        The created daemon thread, or `None` if the request could not be started.
    """
    fallback_message = _FALLBACK_MESSAGE

    try:
        if callback is None:
            raise ValueError("callback is missing")

        if client is None:
            raise ValueError("Groq client is not initialized")

        def worker() -> None:
            """Perform the Groq request and deliver the result to the callback.

            Args:
                None.

            Returns:
                None.
            """
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.4,
                    max_tokens=160,
                )

                response_text = _truncate_words(
                    _extract_response_text(response),
                    MAX_WORDS,
                )

                if not response_text:
                    raise ValueError("response text is empty")

                try:
                    callback(response_text)
                except Exception as callback_exc:
                    _log_ai_error(
                        "_launch_feedback_thread worker callback",
                        callback_exc,
                        "drop the generated text because the consumer callback failed",
                    )

            except Exception as exc:
                error_text = _exception_text(exc)

                if _is_rate_limit_error(exc):
                    message = _RATE_LIMIT_MESSAGE
                    fallback = "return the temporary rate-limit message to the callback"
                    _log_ai_error("_launch_feedback_thread worker", exc, fallback)
                elif _is_auth_error(exc):
                    message = fallback_message
                    fallback = "return the connection-check fallback to the callback after an authentication failure"
                    _log_ai_error("_launch_feedback_thread worker", exc, fallback)
                elif _is_network_error(exc):
                    message = fallback_message
                    fallback = "return the connection-check fallback to the callback after a timeout or network failure"
                    _log_ai_error("_launch_feedback_thread worker", exc, fallback)
                else:
                    message = fallback_message
                    fallback = "return the connection-check fallback to the callback after an API error"
                    _log_ai_error("_launch_feedback_thread worker", exc, fallback)

                try:
                    callback(message)
                except Exception as callback_exc:
                    _log_ai_error(
                        "_launch_feedback_thread worker error callback",
                        callback_exc,
                        "drop the fallback text because the consumer callback failed",
                    )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread

    except Exception as exc:
        _log_ai_error(
            "_launch_feedback_thread",
            exc,
            "return None because the feedback thread could not be started",
        )

        if callback is not None:
            try:
                callback(fallback_message)
            except Exception as callback_exc:
                _log_ai_error(
                    "_launch_feedback_thread startup callback",
                    callback_exc,
                    "drop the fallback text because the consumer callback failed",
                )

        return None


def create_feedback_message(
    bpm: float | int | None,
    rmssd: float | None,
    stress_label: str | None,
    confidence: float | int | None,
    duration_min: float | int | None,
) -> str:
    """Format a clear user prompt for the Groq feedback request.

    Args:
        bpm: Current heart-rate estimate.
        rmssd: Current HRV RMSSD value.
        stress_label: Current stress classification label.
        confidence: Signal confidence value.
        duration_min: Session duration in minutes.

    Returns:
        A formatted string ready to be appended to the system prompt.
    """
    try:
        bpm_text = _format_optional_number(bpm, "bpm")
        rmssd_text = _format_optional_number(rmssd, "ms")

        if confidence is None:
            confidence_text = "unknown"
        else:
            confidence_value = float(confidence)
            if confidence_value != confidence_value or confidence_value in (float("inf"), float("-inf")):
                raise ValueError("confidence is not finite")
            confidence_text = f"{confidence_value:.1f}%"

        stress_text = str(stress_label).strip() if stress_label else "unknown"
        duration_text = f"{_format_duration(duration_min):.1f}"

        return (
            "Current readings:\n"
            f"- BPM: {bpm_text}\n"
            f"- RMSSD: {rmssd_text}\n"
            f"- Stress label: {stress_text}\n"
            f"- Confidence: {confidence_text}\n"
            f"Session duration: {duration_text} minutes\n\n"
            "Give concise feedback under 60 words.\n"
            "If stress is detected, suggest one breathing or relaxation technique with clear steps.\n"
            "If readings are elevated for more than 5 minutes, recommend speaking to a healthcare professional.\n"
            "Never diagnose conditions. Never fabricate data."
        )

    except Exception as exc:
        _log_ai_error(
            "create_feedback_message",
            exc,
            "return a minimal prompt because the user message could not be constructed",
        )
        return "Current readings unavailable. Session duration: 0.0 minutes"


def get_llm_feedback(
    bpm: float | int | None,
    rmssd: float | None,
    stress_label: str | None,
    confidence: float | int | None,
    duration_min: float | int | None,
    callback: Callable[[str], None] | None,
) -> threading.Thread | None:
    """Request asynchronous Groq feedback for the current physiological readings.

    Args:
        bpm: Current heart-rate estimate.
        rmssd: Current HRV RMSSD value.
        stress_label: Current stress classification label.
        confidence: Signal confidence value.
        duration_min: Session duration in minutes.
        callback: Function that receives the generated response text.

    Returns:
        The created daemon thread when a request is scheduled; otherwise `None`.
    """
    try:
        _load_environment()

        api_key = os.getenv(_FEEDBACK_KEY, "").strip()
        if not api_key:
            _log_ai_error(
                "get_llm_feedback",
                ValueError(f"{_FEEDBACK_KEY} is missing or empty"),
                "call the callback with the API key configuration warning",
            )
            if callback is not None:
                callback(_API_KEY_MESSAGE)
            return None

        if _is_low_confidence(confidence):
            _log_ai_error(
                "get_llm_feedback",
                ValueError(f"confidence is below 40 or invalid: {confidence!r}"),
                "call the callback with the low-quality signal warning and skip the API call",
            )
            if callback is not None:
                callback(_LOW_QUALITY_MESSAGE)
            return None

        duration_value = _format_duration(duration_min)
        if duration_value < 1.0:
            _log_ai_error(
                "get_llm_feedback",
                ValueError(f"session duration is below 1 minute: {duration_value:.1f}"),
                "wait until at least 1 minute of session time has elapsed before calling the API",
            )
            return None

        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        user_message = create_feedback_message(
            bpm=bpm,
            rmssd=rmssd,
            stress_label=stress_label,
            confidence=confidence,
            duration_min=duration_value,
        )

        return _launch_feedback_thread(client, user_message, callback)

    except Exception as exc:
        _log_ai_error(
            "get_llm_feedback",
            exc,
            "call the callback with the connection-check fallback",
        )
        if callback is not None:
            try:
                callback(_FALLBACK_MESSAGE)
            except Exception as callback_exc:
                _log_ai_error(
                    "get_llm_feedback fallback callback",
                    callback_exc,
                    "drop the fallback text because the consumer callback failed",
                )
        return None


class FeedbackManager:
    """Manage rate-limited Groq feedback requests and cached responses."""

    def __init__(self) -> None:
        """Initialize the Groq client, rate-limit state, and cached feedback.

        Args:
            None.

        Returns:
            None.
        """
        try:
            _load_environment()
            self._lock = threading.Lock()
            self._latest_feedback = ""
            self._last_call_time = 0.0
            self._cooldown_until = 0.0
            self._in_flight = False
            self._client = None

        except Exception as exc:
            _log_ai_error(
                "FeedbackManager.__init__",
                exc,
                "initialize with an empty cache and allow future recovery attempts",
            )
            self._lock = threading.Lock()
            self._latest_feedback = ""
            self._last_call_time = 0.0
            self._cooldown_until = 0.0
            self._in_flight = False
            self._client = None

    def _build_client(self) -> OpenAI | None:
        """Build and cache a Groq-compatible OpenAI client for the manager.

        Args:
            None.

        Returns:
            A configured `OpenAI` client when the API key is available; otherwise
            `None`.
        """
        try:
            api_key = os.getenv(_FEEDBACK_KEY, "").strip()
            if not api_key:
                raise ValueError(f"{_FEEDBACK_KEY} is missing or empty")

            try:
                return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, httpx_client=httpx.Client(proxies=None))
            except TypeError:
                return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

        except Exception as exc:
            _log_ai_error(
                "FeedbackManager._build_client",
                exc,
                "defer managed feedback requests until GROQ_API_KEY is configured",
            )
            return None

    def _store_feedback(self, text: str) -> None:
        """Store the latest feedback text in a thread-safe way.

        Args:
            text: Feedback text to cache.

        Returns:
            None.
        """
        try:
            with self._lock:
                self._latest_feedback = str(text).strip()

        except Exception as exc:
            _log_ai_error(
                "FeedbackManager._store_feedback",
                exc,
                "drop the feedback text because the cache could not be updated",
            )

    def _set_cooldown(self, seconds: float) -> None:
        """Set a temporary cooldown window before the next Groq request.

        Args:
            seconds: Number of seconds to suppress additional API calls.

        Returns:
            None.
        """
        try:
            cooldown_seconds = max(0.0, float(seconds))
            with self._lock:
                self._cooldown_until = max(self._cooldown_until, time.time() + cooldown_seconds)

        except Exception as exc:
            _log_ai_error(
                "FeedbackManager._set_cooldown",
                exc,
                "skip cooldown updates because the timer state could not be stored",
            )

    def _handle_feedback(self, text: str) -> None:
        """Cache the latest feedback and clear the in-flight flag.

        Args:
            text: Feedback text returned by Groq or a fallback message.

        Returns:
            None.
        """
        try:
            self._store_feedback(text)
            if str(text).strip() == _RATE_LIMIT_MESSAGE:
                self._set_cooldown(_RATE_LIMIT_COOLDOWN_SECONDS)

        except Exception as exc:
            _log_ai_error(
                "FeedbackManager._handle_feedback",
                exc,
                "drop the feedback text because cache storage failed",
            )

        finally:
            try:
                with self._lock:
                    self._in_flight = False

            except Exception as exc:
                _log_ai_error(
                    "FeedbackManager._handle_feedback release",
                    exc,
                    "leave the in-flight state unchanged because the lock operation failed",
                )

    def should_call(self) -> bool:
        """Check whether enough time has passed since the last Groq request.

        Args:
            None.

        Returns:
            `True` when a new request is allowed; otherwise `False`.
        """
        try:
            current_time = time.time()
            with self._lock:
                last_call_time = self._last_call_time
                cooldown_until = self._cooldown_until

            if cooldown_until > current_time:
                return False

            if last_call_time <= 0.0:
                return True

            return (current_time - last_call_time) >= CALL_INTERVAL_SECONDS

        except Exception as exc:
            _log_ai_error(
                "FeedbackManager.should_call",
                exc,
                "treat the call as blocked to avoid flooding the API",
            )
            return False

    def request_feedback(
        self,
        bpm: float | int | None,
        rmssd: float | None,
        stress_label: str | None,
        confidence: float | int | None,
        duration_min: float | int | None,
    ) -> bool:
        """Schedule a background Groq feedback request if the guardrails allow it.

        Args:
            bpm: Current heart-rate estimate.
            rmssd: Current HRV RMSSD value.
            stress_label: Current stress classification label.
            confidence: Signal confidence value.
            duration_min: Session duration in minutes.

        Returns:
            `True` when a request is scheduled; otherwise `False`.
        """
        try:
            if _is_low_confidence(confidence):
                self._store_feedback(_LOW_QUALITY_MESSAGE)
                return False

            duration_value = _format_duration(duration_min)
            if duration_value < 1.0:
                return False

            if not self.should_call():
                return False

            with self._lock:
                if self._in_flight:
                    return False

                self._in_flight = True

            if self._client is None:
                self._client = self._build_client()

            if self._client is None:
                self._store_feedback(_API_KEY_MESSAGE)
                with self._lock:
                    self._in_flight = False
                return False

            user_message = create_feedback_message(
                bpm=bpm,
                rmssd=rmssd,
                stress_label=stress_label,
                confidence=confidence,
                duration_min=duration_value,
            )

            thread = _launch_feedback_thread(self._client, user_message, self._handle_feedback)
            if thread is None:
                with self._lock:
                    self._in_flight = False
                self._store_feedback(_FALLBACK_MESSAGE)
                return False

            with self._lock:
                self._last_call_time = time.time()

            return True

        except Exception as exc:
            _log_ai_error(
                "FeedbackManager.request_feedback",
                exc,
                "store the fallback message and stop scheduling new requests",
            )
            self._store_feedback(_FALLBACK_MESSAGE)
            try:
                with self._lock:
                    self._in_flight = False
            except Exception as lock_exc:
                _log_ai_error(
                    "FeedbackManager.request_feedback release",
                    lock_exc,
                    "leave the in-flight state unchanged because the lock operation failed",
                )
            return False

    def get_latest(self) -> str:
        """Return the latest cached feedback text.

        Args:
            None.

        Returns:
            The last feedback string received, or an empty string if nothing has
            been cached yet.
        """
        try:
            with self._lock:
                return self._latest_feedback

        except Exception as exc:
            _log_ai_error(
                "FeedbackManager.get_latest",
                exc,
                "return an empty string because the cache could not be read",
            )
            return ""
