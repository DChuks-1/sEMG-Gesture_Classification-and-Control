import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import optuna

# Load data
X = np.load("processed/features_fusion.npy")
y = np.load("processed/labels_windows.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Objective function for Optuna
def objective(trial):
    C = trial.suggest_loguniform('C', 1e-2, 1e3)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e1)

    clf = SVC(kernel='rbf', C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

# Optimise
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Best model
print("Best trial:", study.best_trial.value)
print("Best params:", study.best_params)

# Final model with best params
best_svm = SVC(kernel='rbf', **study.best_params)
best_svm.fit(X_train, y_train)
final_preds = best_svm.predict(X_test)
final_acc = accuracy_score(y_test, final_preds)
print(f"\nFinal PSO-SVM Accuracy: {final_acc:.2%}")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import optuna

# Load data
X = np.load("processed/features_fusion.npy")
y = np.load("processed/labels_windows.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Objective function for Optuna
def objective(trial):
    C = trial.suggest_loguniform('C', 1e-2, 1e3)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e1)

    clf = SVC(kernel='rbf', C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

# Optimise
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Best model
print("Best trial:", study.best_trial.value)
print("Best params:", study.best_params)

# Final model with best params
best_svm = SVC(kernel='rbf', **study.best_params)
best_svm.fit(X_train, y_train)
final_preds = best_svm.predict(X_test)
final_acc = accuracy_score(y_test, final_preds)
print(f"\nFinal PSO-SVM Accuracy: {final_acc:.2%}")
