import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# Load filtered EMG + labels
emg = np.load("processed/emg_wavelet.npy")  # shape (N, C)
labels = np.load("data/labels.npy")         # shape (N,)
fs = 2000

# Pick one gesture (e.g. label 1)
gesture_label = 1
indices = np.where(labels == gesture_label)[0]

# Keep continuous block only (avoid scattered samples)
start = indices[0]
end = indices[np.where(np.diff(indices) > 1)[0][0]]  # first gap
emg_gesture = emg[start:end, :]

# Parameters
window_size = 400
step_size = 200

rms_vals = []
mf_vals = []

for i in range(0, len(emg_gesture) - window_size, step_size):
    window = emg_gesture[i:i+window_size, :]
    rms_window = np.sqrt(np.mean(window**2, axis=0))
    spectrum = np.abs(rfft(window, axis=0))
    freqs = rfftfreq(window_size, d=1/fs)
    mf_window = np.sum(freqs[:, None] * spectrum, axis=0) / (np.sum(spectrum, axis=0) + 1e-8)

    rms_vals.append(np.mean(rms_window))  # average across channels
    mf_vals.append(np.mean(mf_window))

# Time axis
time = np.arange(len(rms_vals)) * (step_size / fs)

# Plot RMS and MF drift
plt.figure(figsize=(10, 4))
plt.plot(time, rms_vals, label='RMS ↑ = Effort ↑')
plt.plot(time, mf_vals, label='MF ↓ = Fatigue ↑')
plt.title(f"Fatigue Tracking – Gesture {gesture_label}")
plt.xlabel("Time (s)")
plt.ylabel("Mean Value")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fatigue_rms_mf_gesture1.png", dpi=300)
plt.show()
