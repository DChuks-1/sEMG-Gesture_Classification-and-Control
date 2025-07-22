import matplotlib.pyplot as plt
import numpy as np

from src.load_visualise import load_subject37, plot_emg_with_labels
from src.preprocessing import segment_emg
from src.load_visualise import load_subject37
from src.preprocessing import (
    bandpass_filter, notch_filter, wavelet_denoise,
    extract_features, align_labels
)
from src.preprocessing import align_labels

#  overlay raw vs filtered EMG for one channel so you can visually verify the filter’s effect.
# Choose a Time Window and Channel
fs = 2000
duration_sec = 3
samples = fs * duration_sec
channel = 0
t = np.arange(samples) / fs


# Load EMG data and gesture labels
emg, labels = load_subject37("./data/S37_E1_A1.mat")

# Plot first 4 channels with gesture overlays
plot_emg_with_labels(emg, labels, channels=[0, 1, 2, 3], save_path="./figures/emg_plot_s37_e1.png")

from src.preprocessing import bandpass_filter
filtered = bandpass_filter(emg)

# Plot Raw vs Filtered Overlay
plt.figure(figsize=(12, 4))
plt.plot(t, emg[:samples, channel], label="Raw", alpha=0.5)
plt.plot(t, filtered[:samples, channel], label="Filtered", linewidth=1.5)
plt.title("Raw vs Filtered EMG (Channel 1)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = "figures/emg_raw_vs_filtered_ch1.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")

filtered = bandpass_filter(emg)
print("After bandpass:", filtered.shape)

# Notch Filter
from src.preprocessing import notch_filter

notched = notch_filter(filtered) 


# Plot Bandpass vs Notched
plt.figure(figsize=(12, 4))
plt.plot(t, filtered[:samples, channel], label="Bandpass Only", alpha=0.7)
plt.plot(t, notched[:samples, channel], label="Bandpass + Notch", linewidth=1.5)
plt.title("Bandpass vs Notch Filtered EMG (Channel 1)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/emg_bandpass_vs_notch.png", dpi=300)
print("Saved: figures/emg_bandpass_vs_notch.png")

# Full Stack Plot (Raw vs Bandpass vs Notch)
plt.figure(figsize=(12, 6))
plt.plot(t, emg[:samples, channel], label="Raw", alpha=0.4)
plt.plot(t, filtered[:samples, channel], label="Bandpass", alpha=0.7)
plt.plot(t, notched[:samples, channel], label="Bandpass + Notch", linewidth=1.2)
plt.title("Raw vs Bandpass vs Notch (Channel 1)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/emg_all_filters_ch1.png", dpi=300)
print("Saved: figures/emg_all_filters_ch1.png")

notched = notch_filter(filtered)
print("After notch:", notched.shape)


# Segmenting
segments = segment_emg(notched)          # Use full-length filtered EMG
print("Segment shape:", segments.shape)

print("Filtered EMG shape:", notched.shape)

segments = segment_emg(notched)
print("Segments:", segments.shape)

# Feature Extraction
from src.preprocessing import extract_features

features = extract_features(segments)
print("Features shape:", features.shape)

# Aligning Labels
from src.preprocessing import align_labels

window_labels = align_labels(labels)
print("Labels shape:", window_labels.shape)

np.save("data/features.npy", features)
np.save("data/labels.npy", window_labels)
print("Saved: features.npy & labels.npy")

# Wavelet Transform

from src.preprocessing import wavelet_denoise

wavelet_clean = wavelet_denoise(notched)

# Plot comparison
plt.figure(figsize=(12, 5))
plt.plot(t, notched[:samples, channel], label="Bandpass + Notch", alpha=0.6)
plt.plot(t, wavelet_clean[:samples, channel], label="+ Wavelet Denoising", linewidth=1.5)
plt.title("Filtered vs Wavelet Denoised EMG (Channel 1)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/emg_wavelet_cleaned_ch1.png", dpi=300)
print("Saved: figures/emg_wavelet_cleaned_ch1.png")

#Visualise Segments
segment_idx = 30  # window index to visualise
plt.figure(figsize=(10, 4))
for ch in range(4):  # Plot 4 channels
    plt.plot(segments[segment_idx, :, ch], label=f'Ch {ch+1}')
plt.title("Segmented EMG Window (200ms)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude (μV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/sample_segment_window.png", dpi=300)
print("Saved: figures/sample_segment_window.png")

# Attempt to seperate different preprocessing techniques

# Bandpass
bandpass = bandpass_filter(emg)
segments_band = segment_emg(bandpass)
features_band = extract_features(segments_band)

# Notch
notched = notch_filter(bandpass)
segments_notch = segment_emg(notched)
features_notch = extract_features(segments_notch)

# Wavelet
wavelet = wavelet_denoise(notched)
segments_wave = segment_emg(wavelet)
# features_wave = extract_features(segments_wave)
features_fusion = extract_features(segments_wave)

# Align Labels Once
labels_windows = align_labels(labels)

# Save to processed/
np.save("processed/features_bandpass.npy", features_band)
np.save("processed/features_notch.npy", features_notch)
# np.save("processed/features_wavelet.npy", features_wave)
np.save("processed/features_fusion.npy", features_fusion)
np.save("processed/labels_windows.npy", labels_windows)

print("Features & labels saved to processed/")

# Align labels to sliding windows
window_labels = align_labels(labels)
np.save("processed/labels_windows.npy", window_labels)

