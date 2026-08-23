import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================
# 1. LOAD DATASET
# ==============================

DATASET = "feature_dataset_v2_augmented.csv"

print("Loading dataset...")

df = pd.read_csv(DATASET)

print("Dataset shape:", df.shape)


# ==============================
# 2. REMOVE INVALID DATA
# ==============================

df = df.dropna()

print("After removing missing values:", df.shape)


# ==============================
# 3. SELECT FEATURES
# ==============================

from feature_extraction import FEATURE_COLUMNS as feature_columns

X = df[feature_columns]
y = df["status"]
w = df["sample_weight"] if "sample_weight" in df.columns else pd.Series(1.0, index=df.index)

# ==============================
# 4. TRAIN / VALIDATION / TEST
# ==============================

X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
    X,
    y,
    w,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
    X_temp,
    y_temp,
    w_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# ==============================
# 5. SCALE FEATURES
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ==============================
# 6. SAVE DATA
# ==============================

import numpy as np

np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("X_test.npy", X_test)

np.save("y_train.npy", y_train.to_numpy())
np.save("y_val.npy", y_val.to_numpy())
np.save("y_test.npy", y_test.to_numpy())

np.save("w_train.npy", w_train.to_numpy())
np.save("w_val.npy", w_val.to_numpy())
np.save("w_test.npy", w_test.to_numpy())

joblib.dump(scaler, "models/scaler.pkl")


print("\nPreprocessing completed successfully!")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
