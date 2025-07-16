import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

""" Load and visualises sEMG data from the NinaPro DB2 dataset
    Used for Subject 37, Exercise 1 """

def load_subject37(filepath):
    """ 
    Load sEMG and restimulus data from a NinaPro DB2 file
        Args:
            filepath (str): Path to the .mat file
             
        Returns:
            emg (ndarray): sEMG signals, shape (N, 12)
            restimulus (ndarray): Clean gesture labels, shape (N,)
    """
    # --- Load Data ---
    mat = scipy.io.loadmat('./data/S37_E1_A1.mat')
    print(mat.keys())

    # Extraction of Arrays, Gesture Labels
    emg = mat['emg']
    labels = mat['restimulus'].squeeze() #.squeeze removes the shape (N, 1 and flattens it to (N,))

    # Return the Data
    return emg, labels

""" Verification """
emg, labels = load_subject37('./data/S37_E1_A1.mat')
print(emg.shape, labels.shape)
print("EMG shape:", emg.shape)
print("Labels shape:", labels.shape)
print("Unique gesture labels:", np.unique(labels))

def plot_emg_with_labels(emg, labels, channels=[0,1,2,3], save_path=None):
    """ 
     Plot selected EMG channels with gesture label regions highlighted.
      
     Args:
        emg (ndarray): EMG dat, shape (N,12)
        labels (ndarray): Gesture label per sample, shape (N,)
        channels (list): List of channel indices to plot
    """
    # Get the time axis
    fs = 2000 # sampling frequency in Hz
    time = np.arange(emg.shape[0]) / fs # time is a NumPy array from 0 to N/fs — one timepoint per sample. Essential for the x-axis.

    # Setting up the Plotting Grid
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 8), sharex=True)

    for i, ch in enumerate(channels):
        axes[i].plot(time, emg[:, ch], label=f"Channel {ch+1}")
        axes[i].set_ylabel("μV")
        axes[i].legend(loc="upper right") #Plots each channels microvolts.
        for j in range(1, int(labels.max())+ 1): #Skips 0 (rest)
            mask = labels == j
            if np.any(mask):
                start = time[np.where(mask)[0][0]]
                end = time[np.where(mask)[0][-1]]
                axes[i].axvspan(start, end, color='red', alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()






