import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
# ==============================
# 1. LOAD TEST DATA
# ==============================

print("Loading test data...")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Test data:", X_test.shape)

# ==============================
# 2. RESHAPE FOR CNN
# ==============================

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

print("CNN test shape:", X_test.shape)

# ==============================
# 3. LOAD BEST MODEL
# ==============================

print("\nLoading best CNN model...")

model = tf.keras.models.load_model(
    "models/best_dl_model.keras"
)

print("Model loaded successfully!")

# ==============================
# 4. PREDICTIONS
# ==============================

print("\nMaking predictions...")

probabilities = model.predict(
    X_test,
    batch_size=512,
    verbose=1
)

# Convert probability to class
y_pred = (probabilities >= 0.5).astype(int).flatten()

# ==============================
# 5. METRICS
# ==============================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

# ==============================
# 6. DISPLAY RESULTS
# ==============================

print("\n====================================")
print("     FINAL DEEP LEARNING RESULTS")
print("====================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

# ==============================
# 7. CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n====================================")
print("CONFUSION MATRIX")
print("====================================")

print(cm)

# ==============================
# 8. CLASSIFICATION REPORT
# ==============================

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)
