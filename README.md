# sEMG-Gesture_Classification-and-Control
# sEMG Gesture Control

This project uses surface EMG signals to classify hand gestures.  
It includes signal processing, machine learning models, and hardware control (Arduino-based).

## 🧠 Project Scope
This project explores real-time gesture recognition using sEMG data with integrated muscle fatigue monitoring. The goal is to control a robotic arm (simulated) based on gestures, while adapting behaviour based on fatigue indicators (e.g. RMS and Median Frequency drift).

## Structure

- `data/`: Raw and processed sEMG data.
- `notebooks/`: Jupyter or MATLAB notebooks for preprocessing, feature extraction, and classification.
- `src/`: Standalone Python/MATLAB scripts.
- `hardware/`: Arduino code and hardware setup.
- `results/`: Model outputs like confusion matrices and plots.
- `models/`: Trained models (e.g., `.pkl`, `.mat`).
- `literature/`: Reference papers and notes.
- `figures/`: Diagrams for presentation/report.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

## 📊 Signal Processing Pipeline

- Loaded subject 37 from NinaPro DB2
- Applied 20–450 Hz bandpass + 50 Hz notch
- Segmented into 200 ms overlapping windows
- Extracted 3 features × 12 channels = 36D feature vectors
- Labels aligned per window using majority vote

✅ Data ready for machine learning (8822 windows)

## ▶️ Run

```bash
python main.py
