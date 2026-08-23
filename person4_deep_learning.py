import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    BatchNormalization, Dense, Dropout, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

print("TensorFlow:", tf.__version__)

# ==============================
# 1. LOAD FULL DATA
# ==============================

X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")

y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)

# ==============================
# 2. RESHAPE FOR CNN
# ==============================

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_val = X_val.reshape(
    X_val.shape[0],
    X_val.shape[1],
    1
)

# ==============================
# 3. IMPROVED CNN
# ==============================

model = Sequential([

    Input(shape=(X_train.shape[1], 1)),

    Conv1D(64, 3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.20),

    Conv1D(128, 3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.25),

    Conv1D(256, 3, padding="same", activation="relu"),
    BatchNormalization(),
    Dropout(0.25),

    Flatten(),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.35),

    Dense(128, activation="relu"),
    Dropout(0.25),

    Dense(1, activation="sigmoid")
])

# ==============================
# 4. COMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# 5. MODEL FOLDER
# ==============================

os.makedirs("models", exist_ok=True)

# ==============================
# 6. CALLBACKS
# ==============================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max",
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/best_dl_model.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

# ==============================
# 7. TRAIN FULL DATASET
# ==============================

print("\nStarting FULL CNN training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=512,
    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ],
    verbose=1
)

# ==============================
# 8. SAVE FINAL MODEL
# ==============================

model.save(
    "models/final_dl_model.keras"
)

print("\n====================================")
print("FULL CNN TRAINING COMPLETED")
print("====================================")

print("Best model:")
print("models/best_dl_model.keras")

print("Final model:")
print("models/final_dl_model.keras")
