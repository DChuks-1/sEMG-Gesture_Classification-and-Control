import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load
# X = np.load("processed/features_fusion.npy")
X = np.load("processed/features_fusion_expanded.npy")
y = np.load("processed/labels_windows.npy")

# Pre-define classifiers
models = {
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": SVC(kernel='rbf', C=10, gamma='scale'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "BPNN": MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=300, random_state=42)
}

# CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scale once outside loop
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Run
print("\n5-Fold Cross-Validation Results:")
for name, model in models.items():
    scores = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f"{name}: {mean_acc:.4f} ± {std_acc:.4f}")
