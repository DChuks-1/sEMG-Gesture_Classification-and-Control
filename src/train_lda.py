import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load features and labels
# features = np.load("processed/features_wavelet.npy")  # or features_bandpass.npy or features_notch.npy

# # Each channel has 3 features: RMS, MAV, WL
# # We'll extract only RMS from all channels (every 3rd column starting at 0)
# X = features[:, ::3]  # Slice out RMS features only

# print("Shape of feature matrix (RMS only):", X.shape)
X = np.load("processed/features_wavelet.npy")  # Use entire feature vector (RMS + MAV + WL)
y = np.load("processed/labels_windows.npy")

# Split: 80% train, 20% test, stratified to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train LDA classifier
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Predict
y_pred = lda.predict(X_test)

# Accuracy and classification report
acc = accuracy_score(y_test, y_pred)
print(f"LDA Accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("LDA Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.savefig("figures/lda_confusion_matrix.png", dpi=300)
plt.show()
