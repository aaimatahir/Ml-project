import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================
# 1. LOAD DATASET
# ==============================

DATASET = "feature_dataset_v2.csv"

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

feature_columns = [
    "url_length",
    "dot_count",
    "hyphen_count",
    "special_char_count",
    "has_https",
    "subdomain_count",
    "has_ip",
    "suspicious_keyword_count",
    "digit_count",
    "letter_count",
    "slash_count",
    "question_mark_count",
    "equals_count",
    "ampersand_count",
    "percent_count",
    "at_symbol_count",
    "colon_count",
    "double_slash_count",
    "domain_length",
    "path_length",
    "query_length",
    "digit_ratio",
    "special_char_ratio",
    "url_entropy"
]

X = df[feature_columns]
y = df["status"]

# ==============================
# 4. TRAIN / VALIDATION / TEST
# ==============================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
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

joblib.dump(scaler, "models/scaler.pkl")


print("\nPreprocessing completed successfully!")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
