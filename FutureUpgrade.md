# VitalSense Future Upgrade Plan

## 1. Purpose

This document describes where VitalSense is today, why the current BPM pipeline can fail, and which upgrades would most improve reliability and accuracy.

The goal is to move VitalSense from a **basic webcam-based pulse estimator** to a **quality-aware rPPG system** that can better tolerate lighting changes, motion, and weak facial signal conditions.

---

## 2. Current VitalSense Tech Stack

### 2.1 Runtime and language
- **Python**  
- Core code is organized into:
  - `lighting.py`
  - `signal_extractor.py`
  - `dsp_pipeline.py`

### 2.2 Installed libraries
From `requirements.txt`, the relevant stack includes:

- `opencv-python==4.13.0.92`
- `opencv-contrib-python==4.13.0.92`
- `mediapipe==0.10.32`
- `numpy==2.4.4`
- `scipy==1.17.1`
- `matplotlib==3.10.8`
- `sounddevice==0.5.5`

Supporting packages also present:
- `requests==2.33.1`
- `pydantic==2.13.3`
- `google-genai==1.73.1`
- `websockets==16.0`

For the BPM pipeline itself, the most important dependencies are **OpenCV**, **MediaPipe**, **NumPy**, **SciPy**, and **Matplotlib**.

---

## 3. What VitalSense Currently Does

### 3.1 `lighting.py`
This module acts as a **signal quality gatekeeper** before BPM extraction.

It currently checks:
- **Mean luminance**
- **Brightness standard deviation**
- **Temporal luminance variance**

It classifies frames as:
- `good`
- `low_light`
- `backlit`
- `flicker`

### 3.2 `signal_extractor.py`
This is the live webcam pipeline.

Current flow:
1. Capture webcam frame
2. Flip frame horizontally
3. Classify lighting
4. Run MediaPipe Face Landmarker
5. Use forehead landmarks to build a ROI
6. Compute the **mean green channel** from the ROI
7. Store samples only when lighting and distance are acceptable
8. After a 15-second window:
   - bandpass filter the signal
   - compute BPM using FFT peak selection
   - calculate confidence
   - update session min/max BPM

### 3.3 `dsp_pipeline.py`
This module does the signal processing.

Current steps:
- Butterworth bandpass filtering in SOS form
- Zero-phase filtering with `sosfiltfilt`
- FFT-based BPM estimation
- Confidence based on dominant spectral peak vs average noise
- Static filter response plotting

---

## 4. Weaknesses of the Current Algorithm

The current pipeline is functional, but it is still a **baseline rPPG implementation**. Its main weaknesses are:

### 4.1 Raw green-channel averaging is fragile
VitalSense currently extracts a simple mean green value from the forehead ROI.

#### Weaknesses
- Sensitive to head motion
- Sensitive to ROI drift
- Sensitive to lighting changes
- Sensitive to skin tone variation
- Sensitive to camera auto-exposure and auto-gain

#### Result
The pulse component can be overwhelmed by noise, especially when the subject moves slightly or the room lighting changes.

---

### 4.2 Lighting classification is helpful but still heuristic
`lighting.py` uses fixed thresholds for:
- mean luminance
- luminance standard deviation
- temporal variance

#### Weaknesses
- Thresholds are static
- They may not generalize across cameras, rooms, and skin tones
- They detect bad conditions, but they do not recover signal quality

#### Result
The module can prevent bad frames from entering the pipeline, but it cannot make a weak pulse signal stronger.

---

### 4.3 FFT peak picking can lock onto the wrong frequency
The BPM estimator uses a dominant FFT peak in the cardiac band.

#### Weaknesses
- Motion artifacts can create false peaks
- Harmonics can be mistaken for heart rate
- Short windows have poor frequency resolution
- A single spectral peak can be unstable

#### Result
BPM may jump between windows or lock onto an incorrect frequency.

---

### 4.4 Confidence scoring is too simple
Current confidence is derived from a basic peak-to-noise ratio.

#### Weaknesses
- Does not include motion quality
- Does not include ROI stability
- Does not include lighting quality
- Does not include consistency over time

#### Result
A signal may appear confident even when the estimate is unstable.

---

### 4.5 No temporal BPM tracking
Current output is window-by-window.

#### Weaknesses
- No smoothing across windows
- No history-aware correction
- No suppression of sudden outliers

#### Result
The BPM display can look noisy even when the true pulse is stable.

---

## 5. Better Algorithms to Use

This section lists the most useful upgrades, in order of practical impact.

---

### Stage 1 — Improve signal extraction

#### Current method
- Mean green-channel intensity from the forehead ROI

#### Better methods
1. **POS (Plane-Orthogonal-to-Skin)**
2. **CHROM**
3. **ICA-based rPPG**

#### Why these are better
These methods separate pulse-related color variation from motion and illumination artifacts much better than raw green-channel averaging.

#### Best recommendation
- **POS** for practical webcam-based use
- **CHROM** as a strong alternative
- **ICA** if you want a separation-based method and can handle more complexity

#### Benefit
- Better motion robustness
- Better lighting robustness
- Better BPM stability

---

### Stage 2 — Stabilize the ROI

#### Current method
- Forehead ROI from face landmarks
- Bounding rectangle over landmark points

#### Better methods
- Track a more stable facial region over time
- Use forehead plus cheek ROIs
- Smooth landmark positions across frames
- Reject frames when ROI size or position changes too much
- Add motion quality scoring

#### Why this is better
The rPPG signal is very small. Even tiny ROI shifts can dominate the biological signal.

#### Benefit
- Less noise from face movement
- More stable pulse extraction
- Better window consistency

---

### Stage 3 — Improve BPM estimation

#### Current method
- FFT peak picking in the cardiac band

#### Better methods
1. **Welch Power Spectral Density**
2. **Autocorrelation-based BPM estimation**
3. **Peak tracking across time**
4. **Harmonic suppression logic**
5. **Spectral fusion of FFT + autocorrelation**

#### Why Welch is better
Welch PSD reduces variance compared with a single FFT, especially in noisy signals.

#### Why autocorrelation helps
Autocorrelation can detect periodicity even when the frequency spectrum is messy.

#### Best recommendation
- Use **Welch PSD** as the primary estimator
- Optionally compare it with autocorrelation for consistency

#### Benefit
- More stable BPM
- Better resistance to noisy windows
- Less sensitivity to a single bad spectral bin

---

### Stage 4 — Add temporal smoothing

#### Current method
- No history-aware BPM smoothing

#### Better methods
- Median filter over the last N BPM values
- Exponential moving average
- Kalman filter for state tracking

#### Why this is better
True heart rate does not jump wildly from window to window. Smoothing helps reject isolated outliers.

#### Best recommendation
- Start with a **moving median**
- Upgrade to a **Kalman filter** if you want stronger tracking

#### Benefit
- Cleaner BPM display
- Less jitter
- Better user trust

---

### Stage 5 — Build a stronger quality score

#### Current method
- Lighting status
- Distance check
- FFT peak-to-noise confidence

#### Better quality signals
- ROI stability
- Motion level
- Lighting quality
- Spectral peak dominance
- BPM consistency with previous windows
- Signal-to-noise ratio in the cardiac band

#### Why this is better
A real quality score should reflect both the environment and the signal itself.

#### Benefit
- Better rejection of bad measurements
- More trustworthy BPM values
- Fewer false positives

---

### Stage 6 — Use quality-aware output logic

#### Current method
- Output BPM if confidence exceeds a threshold

#### Better method
- Return BPM only when quality is high enough
- Otherwise:
  - keep the previous trusted BPM
  - or show “unreliable measurement”

#### Why this is better
It is better to show no value than to show a false value.

#### Benefit
- Higher trust
- Better user experience
- Safer interpretation of measurements

---

## 6. Recommended Upgrade Roadmap

### Phase 1 — High impact, low complexity
- Keep `lighting.py` as a gatekeeper
- Replace green-channel-only extraction with **POS**
- Keep the current bandpass filter
- Add a stronger confidence rule

### Phase 2 — Estimation improvement
- Replace raw FFT peak picking with **Welch PSD**
- Add harmonic rejection
- Add BPM smoothing across windows

### Phase 3 — Quality intelligence
- Add ROI stability metrics
- Add motion rejection
- Add a combined signal-quality score

### Phase 4 — Reliability hardening
- Calibrate thresholds using recorded sessions
- Build fallback behavior for poor-quality windows
- Log quality metrics for later tuning

---

## 7. How Each Upgrade Helps

| Upgrade | Problem Solved | Why It Helps |
|---|---|---|
| POS / CHROM | Motion and lighting noise | Extracts pulse more robustly than raw green |
| ROI stabilization | Landmark drift | Keeps the signal source consistent |
| Welch PSD | FFT instability | Reduces variance and noise sensitivity |
| Autocorrelation | Weak spectral peaks | Detects periodicity even when spectrum is messy |
| Temporal smoothing | BPM jitter | Prevents wild frame-to-frame changes |
| Quality scoring | False confidence | Combines multiple reliability signals |
| Quality-aware rejection | Bad readings | Avoids forcing an incorrect BPM |

---

## 8. Research and References

Below are useful papers and references for rPPG and BPM estimation.

### Core rPPG and pulse estimation papers
1. **Verkruysse, Svaasand, Nelson (2008)**  
   *Remote plethysmographic imaging using ambient light*  
   - Early foundational work showing pulse can be measured from video.

2. **Poh, McDuff, Picard (2010)**  
   *Non-contact, automated cardiac pulse measurements using video imaging and blind source separation*  
   - Introduced practical remote pulse estimation from facial video.

3. **de Haan and Jeanne (2013)**  
   *Robust pulse rate from chrominance-based rPPG*  
   - Basis for the **CHROM** method.

4. **Wang et al. (2017)**  
   *Algorithmic Principles of Remote PPG* / POS-related work  
   - Strong foundation for the **POS** method and rPPG signal separation.

5. **de Haan and van Leest (2014)**  
   *Improved motion robustness of remote-PPG by using the blood volume pulse signal from the skin*  
   - Useful for motion robustness ideas.

### Signal quality and reliability references
6. **Signal Quality Index / SQI literature for PPG**
   - Useful for building a confidence score beyond simple peak ratios.

7. **Welch (1967)**  
   *The use of the fast Fourier transform for the estimation of power spectra*  
   - Basis for Welch PSD.

8. **Kalman (1960)**  
   - Useful for temporal tracking and smoothing of BPM estimates.

### Practical search terms
If you continue researching, search for:
- “POS rPPG implementation”
- “CHROM remote photoplethysmography”
- “Welch PSD heart rate estimation”
- “remote PPG motion artifact rejection”
- “rPPG signal quality index”
- “Kalman filter heart rate tracking”

---

## 9. Recommended Final Direction for VitalSense

The best practical path is:

1. **Keep `lighting.py`**
   - It is valuable as a gatekeeper.

2. **Replace raw green averaging**
   - Use **POS** or **CHROM**.

3. **Improve BPM estimation**
   - Use **Welch PSD** instead of a single FFT peak.

4. **Add temporal smoothing**
   - Use a median filter or Kalman filter.

5. **Use quality-aware rejection**
   - Do not output BPM when the signal is not trustworthy.

This combination gives the best balance of:
- accuracy
- reliability
- implementation complexity
- maintainability

---

## 10. Summary

VitalSense currently has a solid baseline pipeline, but the biggest weakness is that it relies on a simple green-channel signal and a single FFT-based BPM estimate.

The most important improvements are:
- **better rPPG extraction**
- **better spectral estimation**
- **better quality scoring**
- **better temporal smoothing**

If those are implemented, VitalSense should become noticeably more reliable in real-world webcam conditions.
