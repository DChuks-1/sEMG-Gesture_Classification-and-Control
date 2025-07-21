import numpy as np
import matplotlib.pyplot as plt

# Load
labels = np.load("processed/labels_windows.npy")
band = np.load("processed/features_bandpass.npy")
notch = np.load("processed/features_notch.npy")
wave = np.load("processed/features_wavelet.npy")

def plot_feature_trend(title, idx, ylabel, savefile):
    plt.figure(figsize=(12, 5))
    plt.plot(band[:, idx], label="Bandpass", alpha=0.6)
    plt.plot(notch[:, idx], label="Bandpass + Notch", alpha=0.6)
    plt.plot(wave[:, idx], label="+ Wavelet Denoised", alpha=0.9)
    plt.title(title)
    plt.xlabel("Window #")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{savefile}", dpi=300)
    print(f"Saved: figures/{savefile}")

# Plot RMS, MAV, WL for Ch 0
plot_feature_trend("RMS (Ch 1)", idx=0, ylabel="RMS (μV)", savefile="rms_ch1.png")
plot_feature_trend("MAV (Ch 1)", idx=1, ylabel="MAV (μV)", savefile="mav_ch1.png")
plot_feature_trend("WL (Ch 1)",  idx=2, ylabel="WL",       savefile="wl_ch1.png")

