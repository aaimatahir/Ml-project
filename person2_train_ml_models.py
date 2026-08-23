"""
Person 2 module: trains Logistic Regression, Decision Tree, Random Forest
and XGBoost on the preprocessed features from Person 1's pipeline, compares
them with accuracy/precision/recall/F1/confusion matrix, and saves each
model plus a summary of which one is best.
"""

import time
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(max_depth=20, random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, n_jobs=-1, random_state=42
    ),
    "xgboost": XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        eval_metric="logloss", n_jobs=-1, random_state=42
    ),
}

results = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"Trained in {time.time()-t0:.1f}s")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1-Score : {f1*100:.2f}%")
    print("Confusion matrix:\n", cm)

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    joblib.dump(model, f"models/{name}.joblib")
    print(f"Saved models/{name}.joblib")

print("\n==================================")
print("MODEL COMPARISON")
print("==================================")
best_name = max(results, key=lambda n: results[n]["f1"])
for name, r in results.items():
    marker = " <-- BEST (by F1)" if name == best_name else ""
    print(f"{name:22s} acc={r['accuracy']*100:.2f}% prec={r['precision']*100:.2f}% "
          f"rec={r['recall']*100:.2f}% f1={r['f1']*100:.2f}%{marker}")

joblib.dump(best_name, "models/best_ml_model_name.joblib")
print(f"\nBest ML model: {best_name}")
