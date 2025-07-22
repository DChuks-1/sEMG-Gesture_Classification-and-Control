import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and split
X = np.load("processed/features_fusion.npy")
y = np.load("processed/labels_windows.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Train BPNN (MLP)
bpnn = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # two hidden layers
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)
bpnn.fit(X_train, y_train)

# 4. Predict & evaluate
y_pred = bpnn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"BPNN Accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred, zero_division=0))

# 5. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.title("BPNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("figures/bpnn_confusion_matrix.png", dpi=300)
plt.show()
