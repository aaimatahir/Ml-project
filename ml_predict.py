"""
Person 2 module: loads the trained classical ML models and exposes a
scam-probability prediction from scaled URL features.

Label convention (confirmed from the dataset): status == 0 is scam/phishing,
status == 1 is legitimate.
"""

import os
import joblib

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_model_cache = {}


def _load(name):
    if name not in _model_cache:
        _model_cache[name] = joblib.load(os.path.join(_MODELS_DIR, f"{name}.joblib"))
    return _model_cache[name]


def best_model_name():
    path = os.path.join(_MODELS_DIR, "best_ml_model_name.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return "xgboost"


def predict_scam_probability(scaled_features, model_name=None):
    """scaled_features: 2D array, shape (1, 24), already StandardScaler-transformed."""
    name = model_name or best_model_name()
    model = _load(name)
    proba = model.predict_proba(scaled_features)[0]
    classes = list(model.classes_)
    scam_idx = classes.index(0)
    return float(proba[scam_idx]), name
