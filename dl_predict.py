"""
Person 4 module: loads the trained CNN and exposes a scam-probability
prediction from scaled URL features.

The CNN was trained with sigmoid output on `status` directly, where
status == 0 is scam and status == 1 is legitimate, so the raw sigmoid
output is P(legitimate); scam probability is 1 - that.
"""

import os

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_dl_model.keras")

_model = None


def _load():
    global _model
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(_MODEL_PATH)
    return _model


def predict_scam_probability(scaled_features):
    """scaled_features: 2D array, shape (1, 24), already StandardScaler-transformed."""
    model = _load()
    x = scaled_features.reshape(scaled_features.shape[0], scaled_features.shape[1], 1)
    prob_legit = float(model.predict(x, verbose=0)[0][0])
    return 1.0 - prob_legit
