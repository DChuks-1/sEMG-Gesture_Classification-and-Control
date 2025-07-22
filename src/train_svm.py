import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load full feature set
# X = np.load("processed/features_wavelet.npy")
X = np.load("processed/features_fusion.npy")
y = np.load("processed/labels_windows.npy")

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# 2.5 Standard Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Train SVM (RBF kernel)
svm = SVC(kernel='rbf', C=10, gamma='scale')  # C and gamma can be tuned
svm.fit(X_train, y_train)

# 4. Predict and evaluate
y_pred = svm.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred, zero_division=0))

# 5. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.savefig("figures/svm_confusion_matrix.png", dpi=300)
plt.show()


# C controls margin softness (high = less tolerant to errors)
# gamma controls kernel shape (low = smoother decision boundaries)

# Try: C = 1, 10, 100; gamma = 'scale' or 0.01, 0.1
