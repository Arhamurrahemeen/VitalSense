This documentation serves as the official project repository file for **VitalSense V1**. It synthesizes the technical milestones, architectural decisions, and the software-engineering rigor applied from Stage 1 through the final optimization in Stage 6.

---

# VitalSense V1: Real-Time AI Health Monitor

**VitalSense** is a high-performance, webcam-based physiological monitor that extracts vital signs using remote Photoplethysmography (rPPG). [cite_start]Built as a modular "Skill-Based" system, it integrates 3D computer vision, digital signal processing (DSP), and Large Language Models (LLMs) to provide real-time heart rate, stress analysis, and health feedback. [cite: 46, 111, 142]

---

## 🏗️ 1. System Architecture
[cite_start]VitalSense follows a strict **Producer-Consumer** multithreaded architecture to maintain a consistent **30 FPS** webcam feed while performing heavy computation in the background. [cite: 41, 73]

* [cite_start]**Orchestrator (`main.py`):** Owns the webcam capture and MediaPipe face tracking. [cite: 41]
* [cite_start]**Signal Processor (`dsp_pipeline.py`):** Isolates the cardiac waveform using time-domain filters. [cite: 113, 121]
* [cite_start]**AI Feedback Layer (`ai_feedback.py`):** Communicates asynchronously with the Groq API for health insights. [cite: 42]
* [cite_start]**Live Dashboard (`dashboard.py`):** Renders a three-panel Matplotlib visualization of the raw, filtered, and spectral data. [cite: 74]

---

## 🛠️ 2. Core Technical Stages

### Stage 1: The Foundation (Extraction)
[cite_start]The system utilizes the **MediaPipe Tasks API (0.10.x)** for asynchronous face landmarking. [cite: 51, 52] 
* [cite_start]**Forehead ROI:** Tracks specific landmarks (IDs: 10, 109, 67, etc.) to isolate the most stable skin region. [cite: 58, 59]
* [cite_start]**G-Mean Extraction:** Calculates the average green-channel intensity ($G_{mean}$) because human blood absorbs green light most effectively. [cite: 60]

### Stage 2: Signal Quality Assurance (SQA)
[cite_start]Acts as a "Gatekeeper" to prevent "silent failures" where noisy data leads to incorrect readings. [cite: 83, 86]
* [cite_start]**Luminance Analysis:** Converts frames to **YCrCb** color space to isolate brightness (Y) from skin tone. [cite: 93, 94]
* [cite_start]**Lighting Classifiers:** Detects **Low Light** ($\mu < 60$), **Backlit** ($\sigma > 80$), and **Flicker** ($Var > 25$) to pause extraction if conditions are poor. [cite: 98, 99, 100]

### Stage 3 & 4: The Mathematical Engine (DSP)
[cite_start]Converts raw pixel fluctuations into a clean cardiac waveform. [cite: 112, 113]
* [cite_start]**SOS Butterworth Filter:** A 2nd-order bandpass filter ($0.7–4.0$ Hz) using **Second-Order Sections** to prevent numerical rounding errors. [cite: 123, 124]
* [cite_start]**Zero-Phase Filtering:** Employs `sosfiltfilt` to ensure the filtered peaks align perfectly with physical heartbeats without time delay. [cite: 125, 126]
* [cite_start]**Spectral Analysis:** Uses **Fast Fourier Transform (FFT)** to identify the dominant pulse frequency. [cite: 130, 131]
* [cite_start]**SNR Confidence Score:** Calculates a trust metric based on the Signal-to-Noise Ratio: $C = (\frac{P_{peak}}{P_{noise}} / 10) \times 100$. [cite: 133, 134]

### Stage 5: HRV & Stress Classification
[cite_start]Beyond average BPM, the system analyzes the variance between individual beats. [cite: 142]
* **Metrics:** Computes **RMSSD** (Root Mean Square of Successive Differences) and **SDNN**. 
* **Classifier:** A rule-based logic gate that labels user states as **Calm**, **Moderate Stress**, or **High Stress** based on RMSSD thresholds. 

### Stage 6: AI Feedback & Optimization
Integrates the **Groq API** (`llama-3.3-70b-versatile`) to provide health advice.
* **Async Daemon Threads:** Ensures API latency does not stutter the 30 FPS monitor.
* **Safety Guardrails:** Enforces a 60-word limit and prevents medical diagnosis.
* **Code Hardening:** Optimized signal padding (min 20 samples) and lazy-loading for the API client to ensure "buttery smooth" performance.

---

## 📊 3. Performance Metrics
* [cite_start]**Target Frame Rate:** 30 FPS (Locked). [cite: 67]
* [cite_start]**Sampling Requirement:** Minimum 8 Hz (Nyquist-compliant for heart rates up to 240 BPM). [cite: 66, 67]
* **API Throttle:** Every 30–60 seconds to respect Groq/Gemini RPM limits.

---

## 🚀 4. Future Roadmap (V2)
1.  **Enhanced UI:** Moving from OpenCV/Matplotlib to a custom-themed GUI.
2.  **Multimodal Vitals:** Researching SpO2 (oxygen saturation) and Respiratory Rate (RR) extraction.
3.  **Edge Deployment:** Further optimization for mobile and low-power hardware.

---

## 📝 5. Research & Citations
* **Verkruysse et al. (2008)[cite_start]:** Why the green channel is superior for rPPG. [cite: 60]
* **Poh et al. (2010)[cite_start]:** Foundations of FFT-based heart rate estimation. [cite: 130]
* **Zhao et al. (2023) / Fontes et al. (2024):** HRV and stress-level interpretation.

---
**Developed by:** Muhammad Arham  
**Affiliation:** Dawood University of Engineering & Technology 
