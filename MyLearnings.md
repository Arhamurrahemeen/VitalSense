This is a comprehensive draft for your `MyLearnings.md` file. It captures your transition from a Computer Systems Engineering student to a developer capable of handling real-time AI and signal processing.

---

# My Learnings: The VitalSense Journey

Developing **VitalSense** was more than just a coding project; it was a deep dive into the intersection of Digital Signal Processing (DSP), Computer Vision, and Artificial Intelligence. Below is a summary of the core technical and architectural skills I acquired during this journey.

## 1. Environment & Project Management
* [cite_start]**Miniconda & Virtual Environments:** I moved away from global Python installations to **Miniconda**[cite: 1, 3]. [cite_start]I learned how to create isolated environments (`.venv`) to prevent package conflicts, ensuring that VitalSense remains reproducible and stable[cite: 9, 10, 50].
* [cite_start]**Modular Architecture:** I learned to break a monolithic script into a clean, multi-file structure (e.g., `lighting.py`, `dsp_pipeline.py`, `hrv.py`), which made debugging and scaling the project much easier[cite: 38, 41, 42].

## 2. Computer Vision & Physiological Extraction
* [cite_start]**MediaPipe Tasks API (0.10.x):** I mastered the shift from the legacy `mp.solutions` to the modern **Tasks API**[cite: 51, 69]. [cite_start]I learned how to implement **asynchronous face tracking** so that AI landmark detection doesn't block the main video thread[cite: 52, 72].
* [cite_start]**Forehead ROI Logic:** I researched and identified specific **Forehead IDs**[cite: 58]. [cite_start]I learned that the forehead is anatomically stable and rich in capillaries, making it the ideal Region of Interest (ROI) for rPPG[cite: 59].
* [cite_start]**Color Space Mastery (YCrCb vs. RGB):** I learned that while RGB is standard, it mixes brightness into every channel[cite: 94]. [cite_start]By converting to **YCrCb**, I could isolate the **Luminance (Y)** channel to build a lighting validator that ignores skin tone and focuses strictly on environmental quality[cite: 93, 95].

## 3. Digital Signal Processing (DSP)
* [cite_start]**Butterworth Bandpass Filters:** I learned to design and implement a 2nd-order Butterworth filter restricted to the cardiac band (**0.7–4.0 Hz**)[cite: 123]. [cite_start]I specifically used **Second-Order Sections (SOS)** and `sosfiltfilt` to ensure numerical stability and zero-phase shift (perfectly aligned peaks)[cite: 124, 125, 126].
* [cite_start]**Fast Fourier Transform (FFT):** I learned to use `np.fft.rfft` to transform a 15-second time-domain signal into a frequency spectrum[cite: 130]. [cite_start]This allowed me to identify the dominant "pulse peak" and calculate the heart rate in BPM[cite: 131].
* [cite_start]**SNR Confidence Scoring:** I developed a mathematical formula to quantify "trust" in a reading by comparing the power of the heart rate peak to the surrounding noise floor ($C = \frac{P_{peak}}{P_{noise}}$)[cite: 132, 133].

## 4. Advanced Physiological Analysis
* **Heart Rate Variability (HRV):** By using `scipy.signal.find_peaks`, I learned to calculate the **Inter-Beat Interval (IBI)** in milliseconds. I implemented the **RMSSD** and **SDNN** formulas to derive stress levels from the timing variance between beats.
* **Oximetry Correlation:** Through my research, I learned how traditional pulse oximeters use Red and IR light absorption to calculate oxygen saturation and pulse. I used this knowledge to calibrate VitalSense, comparing my rPPG results against a physical oximeter for ground-truth validation.

## 5. High-Performance Multithreading
* **Concurrent Execution:** To achieve a "buttery smooth" 30 FPS, I learned to use Python’s `threading` module.
    * **Main Thread:** Handles the OpenCV camera feed and MediaPipe tracking.
    * **Dashboard Thread:** Runs the Matplotlib animation in a non-blocking way.
    * **AI Thread:** Handles the Groq/LLM API calls as daemon threads so the UI never freezes during network requests.
* **Shared State Synchronization:** I learned to use `threading.Lock()` to allow multiple threads to safely read and write to a central `shared_state` dictionary without memory corruption.

## 6. AI Integration & API Management
* **API Performance Metrics:** I learned the critical difference between **RPM (Requests Per Minute)** and **RPD (Requests Per Day)**.
* **The Groq Pivot:** When I encountered rate-limit bottlenecks with Gemini, I pivoted to **Groq (Llama 3.3-70B)**. This taught me to manage API usage by implementing a `FeedbackManager` that throttles calls and caches responses.
* **Asynchronous AI Logic:** I learned to handle "Rate Limit" and "Quota Exceeded" errors gracefully, providing fallback messages to the user instead of letting the application crash.

## 7. Research Findings (Highlights)
* [cite_start]**Green Channel Superiority:** My research (referencing **Verkruysse et al.**) confirmed that the green channel contains the strongest plethysmographic signal because oxyhemoglobin has a high absorption peak in that spectrum[cite: 60].
* [cite_start]**Environmental Impact:** I discovered that lighting flicker (50/60Hz) from artificial lights can alias into the cardiac band, creating "hallucinated" BPM readings—a finding that led to the development of my **Stage 2 Lighting Classifier**[cite: 101, 105].

---

> **Summary:** Through VitalSense, I evolved from writing simple Python scripts to architecting a real-time, multithreaded medical-vision system. I now have a solid foundation in **NumPy** for high-speed math, **SciPy** for signal processing, and **Groq/Gemini** for generative health feedback.
