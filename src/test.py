import numpy as np

X = np.load("processed/features_fusion_expanded.npy")
print(X.shape)  # Should be (8822, 56)

y = np.load("processed/labels_windows.npy")
print(y.shape)  # Should be (8822,)
