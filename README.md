# VitalSense

[cite_start]**VitalSense** is a high-performance implementation of remote PhotoPlethysmoGraphy (rPPG)[cite: 1, 22]. [cite_start]It enables camera-based measurement of the human blood volume pulse by detecting subtle changes in skin color that originate from blood pulsation[cite: 60, 137].

> [!CAUTION]
> [cite_start]This is a Computer Systems Engineering student project intended for demonstration purposes only[cite: 49, 65]. [cite_start]The provided code is not suitable for clinical use or medical decision-making[cite: 86, 104].

## Core Functionality
[cite_start]Unlike traditional monolithic rPPG scripts, VitalSense uses a modular architecture that decouples signal extraction from the user interface[cite: 41, 42]. [cite_start]The system leverages the **MediaPipe Tasks API (0.10.x)** for asynchronous face tracking, ensuring a stable **30 FPS** monitor even during heavy processing[cite: 51, 52, 73].

### Key Modules:
* [cite_start]**Signal Extraction**: Tracks anatomically stable forehead regions (ROI) to capture the raw green-channel signal ($G_{mean}$)[cite: 58, 59, 60].
* [cite_start]**DSP Pipeline**: Applies a 2nd-order SOS Butterworth bandpass filter ($0.7–4.0$ Hz) and Fast Fourier Transform (FFT) to isolate the heart rate[cite: 123, 124, 130].
* [cite_start]**AI Layer**: Integrates a **Groq-powered Llama 3.3** model to provide asynchronous, real-time health feedback and stress classification[cite: 10, 21].

## Installation & Usage
[cite_start]To run the VitalSense application, clone the repository and set up a virtual environment to ensure isolation[cite: 3, 10].

```bash
# Clone the repository
git clone https://github.com/Arhamurrahemeen/VitalSense.git
cd VitalSense

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Monitor
Launch the main orchestrator to start the webcam feed and HUD:
```bash
python main.py
```

## User Interfaces
* [cite_start]**OpenCV HUD**: The primary interface providing real-time overlays for BPM, Confidence, and AI-generated feedback[cite: 55, 89].
* **Scientific Dashboard**: A secondary Matplotlib window displaying live time-domain PPG waveforms and frequency-domain power spectra.

## Documentation
For a detailed breakdown of the engineering stages, mathematical computations, and research findings, see the full project documentation:
👉 **[VitalSense_V1.md](./VitalSense_V1.md)**

## Research & Footnotes
VitalSense is built upon established research in the field of digital signal processing and rPPG:
* **W. Verkruysse et al. (2008)[cite_start]**: Remote plethysmographic imaging using ambient light[cite: 1].
* **Poh et al. (2010)**: Foundations of webcam-based heart rate estimation.
* **Zhao et al. (2023)**: HRV feature extraction from rPPG signals.

---
[cite_start]**Developed by:** Muhammad Arham (Lead Developer & Solution Architect) [cite: 1]
