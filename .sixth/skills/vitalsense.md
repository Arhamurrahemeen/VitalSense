# Skill: VitalSense — rPPG Real-Time Health Monitor

## Description
Standards for building a webcam-based remote photoplethysmography (rPPG) system that extracts heart rate, HRV, and stress levels from facial video using OpenCV, MediaPipe, SciPy, and Gemini AI. Covers signal extraction, DSP filtering, FFT analysis, HRV computation, and LLM health feedback.

## Context
- **Files**: `*.py` in `VitalSense/` directory
- **When**: Any rPPG, computer vision, signal processing, or AI integration code
- **Tech Stack**: OpenCV, MediaPipe, NumPy, SciPy, Matplotlib, google-genai

## Rules

### Architecture & Structure
- Follow the modular file structure strictly:
  - `main.py` — entry point, main loop, orchestration only
  - `signal_extractor.py` — MediaPipe face mesh, ROI extraction, green channel signal
  - `lighting.py` — YCrCb lighting quality classifier (good/low_light/backlit/flicker)
  - `dsp_pipeline.py` — Butterworth bandpass filter design, signal filtering, FFT, BPM + confidence
  - `hrv.py` — Peak detection, IBI computation, RMSSD/SDNN, stress classification
  - `ai_feedback.py` — Gemini API async health feedback with safety guardrails
  - `dashboard.py` — Matplotlib real-time 3-panel visualization
- Never mix DSP logic into `main.py` — keep it an orchestrator
- Import between modules explicitly: `from dsp_pipeline import apply_filter, compute_bpm_and_confidence`

### Signal Processing Standards
- **Frame rate**: Target 30 FPS (`FS = 30`). Always declare as global constant.
- **Filter**: Butterworth bandpass, 2nd order, SOS format, zero-phase (`sosfiltfilt`)
  - Low cutoff: 0.7 Hz (42 BPM)
  - High cutoff: 4.0 Hz (240 BPM)
- **Signal buffer**: Maintain rolling window of 300 samples (10 seconds at 30 FPS)
- **Detrend before filtering**: Remove DC offset with `signal - np.mean(signal)`
- **FFT**: Use `np.fft.rfft` and `np.fft.rfftfreq` for real signals
- **BPM confidence**: Compute as `(peak_power / noise_power / 10) * 100`, cap at 100%

### MediaPipe & ROI Extraction
- Use `mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)`
- Forehead landmark IDs (fixed set): `[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379]`
- Extract ROI bounding box from landmark coordinates, then crop frame
- Green channel extraction: `np.mean(roi[:, :, 1])` — index 1 is G in BGR

### Error Handling
- Every function must have robust error handling with **three-part logging**:
  1. **What failed**: Exception type and message
  2. **What caused it**: Input state, frame count, signal length, or parameter values
  3. **How it was solved**: Fallback value, graceful degradation, or retry strategy
- Use `try/except` blocks around OpenCV capture, MediaPipe processing, and API calls
- Log to stdout with `print(f"[ERROR] {func_name}: {e} | signal_len={len(signal)} | fallback=0")`
- Never let the main loop crash — return safe defaults (BPM=0, confidence=0, stress="uncertain")

### Performance & Optimization
- Pre-allocate NumPy arrays where possible
- Use `cv2.cvtColor` in-place color space conversions
- Avoid list appending in hot loops — use `collections.deque` with maxlen for signal buffers
- Run Gemini API calls in daemon threads (`threading.Thread(daemon=True)`) to avoid blocking UI
- Use `FuncAnimation` with `blit=True` for Matplotlib dashboard when possible

### Safety & Ethics (AI Feedback)
- **NEVER diagnose medical conditions** in Gemini prompts or code comments
- System prompt must include: "NEVER diagnose conditions. NEVER fabricate data."
- Always check `confidence &gt;= 40%` before classifying stress or giving health advice
- If confidence &lt; 40%, advise improving lighting, not health recommendations
- All health suggestions must be framed as "consider speaking to a healthcare professional"

### Code Documentation
- Every function gets a docstring with: Args, Returns, and Raises
- Inline comments explain **why**, not **what** (the code shows what)
- Reference research papers in comments for DSP decisions:
  - Verkruysse et al. 2008 — green channel rationale
  - Poh et al. 2010 — FFT-based HR estimation
  - Fontes et al. 2024 — RMSSD stress thresholds
  - Zhao et al. 2023 — HRV from rPPG
- Add `# TODO:` markers for calibration tasks (threshold tuning, threshold experiments)

## Examples

### ✅ Good — Modular signal extractor with error handling
```python
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
FOREHEAD_IDS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379]

# Rolling buffer: 300 samples = 10s at 30 FPS
green_buffer = deque(maxlen=300)

def extract_forehead_roi(frame: np.ndarray) -&gt; tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Extract forehead ROI from frame using MediaPipe face mesh.
    
    Args:
        frame: BGR image from webcam
        
    Returns:
        Tuple of (roi_crop, landmark_points) — roi is None if no face detected
        
    Raises:
        ValueError: If frame is not a valid numpy array
    """
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError(f"Invalid frame: type={type(frame)}, size={getattr(frame, 'size', 'N/A')}")
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if not results.multi_face_landmarks:
        return None, []
    
    lm = results.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in FOREHEAD_IDS]
    
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    
    # Clamp to frame bounds to prevent indexing errors
    y_min, y_max = max(0, min(ys)), min(h, max(ys))
    x_min, x_max = max(0, min(xs)), min(w, max(xs))
    
    if y_max &lt;= y_min or x_max &lt;= x_min:
        return None, pts  # Degenerate ROI
    
    roi = frame[y_min:y_max, x_min:x_max]
    return roi, pts

def get_green_signal(roi: np.ndarray) -&gt; float:
    """
    Compute mean green channel intensity from ROI.
    Verkruysse et al. 2008: green channel carries strongest PPG signal.
    
    Args:
        roi: Cropped forehead region (BGR)
        
    Returns:
        Mean green channel value or 0.0 if ROI invalid
    """
    if roi is None or roi.size == 0:
        return 0.0
    return float(np.mean(roi[:, :, 1]))

def process_frame(frame: np.ndarray) -&gt; dict:
    """
    Full pipeline for one frame: ROI extraction → green signal → buffer update.
    
    Args:
        frame: Current webcam frame
        
    Returns:
        Dict with keys: signal, has_face, landmarks, error
    """
    result = {"signal": 0.0, "has_face": False, "landmarks": [], "error": None}
    
    try:
        roi, pts = extract_forehead_roi(frame)
        if roi is not None:
            result["has_face"] = True
            result["landmarks"] = pts
            signal = get_green_signal(roi)
            green_buffer.append(signal)
            result["signal"] = signal
        else:
            # No face detected — append 0 to maintain buffer continuity
            green_buffer.append(0.0)
            
    except Exception as e:
        # Log what failed, what caused it, and the fallback
        print(f"[ERROR] process_frame: {type(e).__name__}: {e} | "
              f"frame_shape={getattr(frame, 'shape', 'N/A')} | "
              f"buffer_len={len(green_buffer)} | fallback=0.0")
        result["error"] = str(e)
        green_buffer.append(0.0)
    
    return result
### ❌ Bad — Monolithic, no error handling, no buffer management
```python
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
green = []

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp.solutions.face_mesh.FaceMesh().process(rgb)
    lm = results.multi_face_landmarks[0].landmark
    # Crashes if no face, no error info, unbounded list growth
    g = np.mean(frame[int(lm[10].y):int(lm[338].y), int(lm[10].x):int(lm[338].x), 1])
    green.append(g)
```

### ✅ Good — DSP pipeline with research-backed comments
```python
from scipy.signal import butter, sosfiltfilt
import numpy as np

FS = 30  # Webcam frame rate — fixed for consistent filtering
LOW, HIGH = 0.7, 4.0  # Cardiac band: 42–240 BPM (Poh et al. 2010)

def design_bandpass(low: float, high: float, fs: float, order: int = 2):
    """
    Design Butterworth SOS bandpass filter.
    SOS format is numerically stable for low-order filters (SciPy best practice).
    """
    nyq = fs / 2
    sos = butter(order, [low/nyq, high/nyq], btype='bandpass', output='sos')
    return sos

def apply_filter(signal: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """
    Apply zero-phase bandpass filter.
    sosfiltfilt prevents time shift that would distort BPM calculation.
    """
    if len(signal) < 10:
        print(f"[WARN] apply_filter: signal too short ({len(signal)} samples), returning raw")
        return signal
    try:
        detrended = signal - np.mean(signal)  # Remove DC offset from lighting changes
        filtered = sosfiltfilt(sos, detrended)
        return filtered
    except Exception as e:
        print(f"[ERROR] apply_filter: {e} | signal_len={len(signal)} | sos_shape={sos.shape} | returning zeros")
        return np.zeros_like(signal)
```

## References
- Verkruysse et al. 2008 — "Remote plethysmographic imaging using ambient light" (green channel rationale)
- Poh et al. 2010 — "Non-contact, automated cardiac pulse measurements using video imaging and blind source separation" (FFT-based HR)
- Fontes et al. 2024 — RMSSD stress thresholds for rPPG HRV analysis
- Zhao et al. 2023 — HRV feature extraction from camera-based PPG
- SciPy Signal Processing: https://docs.scipy.org/doc/scipy/reference/signal.html
- MediaPipe Face Mesh: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- Gemini API (google-genai): https://github.com/googleapis/python-genai