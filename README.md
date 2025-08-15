# Dual‑Function sEMG Pipeline: Gesture Classification + Fatigue Detection (SRL Proof‑of‑Concept)

## Scope
This repository implements a distinction‑grade, reproducible sEMG pipeline that:
- Performs **gesture classification** (3–5+ wrist/hand gestures) using NinaPro DB2 for model development and ablations.
- Computes **muscle fatigue** online features and a **composite Fatigue Index (FI)** (MPF/MDF ↓, RMS/iEMG trends ↑), with robust change detection (EWMA/CUSUM).
- Demonstrates **intent–capacity integration** via a **simulated SRL controller** that modulates assistance/impedance from logged predictions and FI.
- (Optional) Live **gesture‑only** demo with Myo Armband for latency validation.

**Out of scope:** Full hardware SRL integration and long‑term clinical validation.

## Datasets
- **NinaPro DB2 (offline)** — 12‑ch sEMG @2 kHz with labelled gestures for training/validation.
- **Myo Armband (optional live)** — 8‑ch sEMG @200 Hz (+ IMU) for real‑time demonstration.

## Pipeline (high level)
1. **Preprocess:** zero‑phase 20–450 Hz band‑pass + 50 Hz notch; rectification; (optional) wavelet denoising; per‑session normalisation.
2. **Segmentation:** 200 ms windows, 100 ms hop (50% overlap).
3. **Features:**
   - **Gesture:** TD (RMS, MAV, WL, ZC, SSC, iEMG, AR(4)); optional TF (STFT/CWT images).
   - **Fatigue:** FD (MDF, MPF, spectral slope, bandpowers) + amplitude trends; rolling slopes (≈30 s buffer).
4. **Models:** LDA / SVM‑RBF / RF baselines; optional 1D‑CNN or Spec‑CNN+BiLSTM.
5. **Fatigue detection:** Composite **FI ∈ [0,1]** + EWMA/CUSUM thresholds; optional short‑horizon forecasting of ΔMPF.
6. **Integration (sim):** Assistance/impedance gain scheduling from FI; playback on logged data with animation export.

## Success Metrics (pass/fail thresholds)
### Gesture classification
- **Within‑subject (3–5 gestures):**  
  - Top‑1 accuracy **≥ 90%**  
  - Macro‑F1 reported  
  - **MER** (Movement Error Rate) reported with 95% CI  
- **Latency (optional live):** End‑to‑end **< 120 ms** (filter → features → model → output)

### Fatigue detection
- **Physiological trends:** Significant **negative slope** in MPF/MDF across fatiguing segments (p < 0.05).  
- **Binary detection:** AUROC **≥ 0.80** for fatigue vs non‑fatigue episodes (labelled or stress‑tested via spectral drift injection).  
- **Stability:** False‑alarm rate **≤ 5%** on non‑fatiguing segments; mean detection delay reported.

### Integration (simulation)
- Assistance/impedance gain **monotonic** with FI; saturation bounds documented (e.g., α ∈ [0.2, 1.0]).  
- 20–60 s **demo video** (or GIF) showing assistance increase as FI rises.  
- Deterministic reproduction from `results/` logs via a single script.

## Reproducibility
- **Config‑first**: all experiments driven by YAML in `configs/`.  
- Fixed **random seeds**, version‑pinned deps, and saved artefacts (`results/`).  
- **Unit tests** for filters, features, MER, FI; latency profiling decorators on the real‑time path.  
- Optional DVC for dataset pointers (raw data not stored in repo).

## Folder Structure (summary)
