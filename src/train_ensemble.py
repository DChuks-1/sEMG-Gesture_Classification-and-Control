import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
X = np.load("processed/features_fusion_expanded.npy")
y = np.load("processed/labels_windows.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Define base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', max_iter=1000, random_state=42)

# 4. Ensemble (soft voting)
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)],
    voting='soft'
)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# 5. Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Ensemble Accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred, zero_division=0))

# 6. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="PuBu")
plt.title("Voting Ensemble Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("figures/ensemble_confusion_matrix.png", dpi=300)
plt.show()
