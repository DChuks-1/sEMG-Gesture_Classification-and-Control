import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt, iirnotch
from scipy.fft import rfft, rfftfreq


def bandpass_filter(emg, fs=2000, low=20, high=450):
    """
    Apply a Butterworth bandpass filter to multi-channel EMG data.

    Args:
        emg (ndarray): Raw EMG signal, shape (N, C)
        fs (int): Sampling frequency (Hz)
        low (float): Low cut-off frequency (Hz)
        high (float): High cut-off frequency (Hz)

    Returns:
        filtered_emg (ndarray): Bandpass-filtered EMG, shape (N, C)
    """

    # Design of a Butterworth Filter

    from scipy.signal import butter, filtfilt

    nyquist = fs / 2
    b, a = butter(4, [low / nyquist, high / nyquist], btype='bandpass')

    #  Apply the Filter to Each Channel
    filtered_emg = filtfilt(b, a, emg, axis=0)

    return filtered_emg

def notch_filter(emg, fs=2000, freq=50.0, Q=30.0):
    """
    Apply a notch filter to remove powerline interference.

    Args:
        emg (ndarray): EMG signal, shape (N, C)
        fs (int): Sampling rate
        freq (float): Frequency to notch out (default = 50 Hz)
        Q (float): Quality factor (narrower = higher Q)

    Returns:
        filtered_emg (ndarray): EMG with notch filter applied
    """
    # Design of a Notch Filter
    from scipy.signal import iirnotch, filtfilt

    b, a = iirnotch(w0=freq, Q=Q, fs=fs)

    filtered_emg = filtfilt(b, a, emg, axis=0)
    return filtered_emg

# Segmenting
def segment_emg(emg, window_size=400, step_size=200):
    """
    Segment EMG into overlapping windows.

    Args:
        emg (ndarray): Filtered EMG, shape (N, C)
        window_size (int): Samples per window (e.g., 200ms = 400 @ 2kHz)
        step_size (int): Step between windows (e.g., 100ms = 200)

    Returns:
        segments (ndarray): Shape (W, window_size, C)
    """

    segments = []
    N = emg.shape[0]

    for start in range(0, N - window_size + 1, step_size):
        end = start + window_size
        window = emg[start:end, :]
        segments.append(window)
        
    return np.array(segments)

# Adding extra features (ZC, SSC, IEMG)
def zero_crossings(signal, threshold=1e-3):
    return np.sum((signal[:-1] * signal[1:] < 0) & 
                  (np.abs(signal[:-1] - signal[1:]) >= threshold))

def slope_sign_changes(signal, threshold=1e-3):
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    return np.sum(((diff1[:-1] * diff1[1:] < 0) &
                   (np.abs(diff1[:-1] - diff1[1:]) >= threshold)))

def integrated_emg(signal):
    return np.sum(np.abs(signal))

# Feature Extraction
def extract_features(segments, fs=2000):
    """
    Extract features (RMS, MAV, WL, MF, ZC, SSC, IEMG) for each segment.

    Args:
        segments (ndarray): Shape (W, L, C)
        fs (int): Sampling rate in Hz

    Returns:
        features (ndarray): Shape (W, C x F)
    """
    W, L, C = segments.shape
    feature_list = []

    for window in segments:
        window_feats = []

        for ch in range(C):
            signal = window[:, ch]

            rms = np.sqrt(np.mean(signal ** 2))
            mav = np.mean(np.abs(signal))
            wl = np.sum(np.abs(np.diff(signal)))

            freqs = rfftfreq(len(signal), d=1/fs)
            spectrum = np.abs(rfft(signal))
            mf = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-8)

            zc = zero_crossings(signal)
            ssc = slope_sign_changes(signal)
            iemg = integrated_emg(signal)

            window_feats.extend([rms, mav, wl, mf, zc, ssc, iemg])

        feature_list.append(window_feats)

    return np.array(feature_list)

# Aligning Labels
def align_labels(labels, window_size=400, step_size=200):
    """
    Align gesture labels to each EMG window using majority voting.

    Args:
        labels (ndarray): Original sample-wise labels (N,)
        window_size (int): Size of each EMG segment in samples
        step_size (int): Step between windows

    Returns:
        window_labels (ndarray): One label per window
    """
    window_labels = []
    N = len(labels)

    for start in range(0, N - window_size + 1, step_size):
        end = start + window_size
        window = labels[start:end]
        most_common = np.bincount(window).argmax()
        window_labels.append(most_common)

    return np.array(window_labels)

def wavelet_denoise(signal, wavelet='db4', level=3):
    """
    Apply wavelet denoising to a multi-channel EMG signal.

    Args:
        signal (ndarray): EMG data, shape (N, C)
        wavelet (str): Wavelet type
        level (int): Number of decomposition levels

    Returns:
        denoised (ndarray): Wavelet-denoised signal
    """
    denoised = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        coeffs = pywt.wavedec(signal[:, ch], wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
        denoised[:, ch] = pywt.waverec(coeffs, wavelet)[:signal.shape[0]]
    return denoised



    









