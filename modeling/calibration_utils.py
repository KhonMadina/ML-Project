#!/usr/bin/env python3
"""
Calibration utilities: Temperature scaling wrapper for scikit-learn classifiers.

- Works with multi-class classifiers that implement decision_function or predict_proba.
- Optimizes a single temperature parameter T to minimize negative log-likelihood on a validation set.
- Serializes with joblib (define in a stable module location).
"""
from __future__ import annotations

import math
from typing import Optional, List

try:
    import numpy as np  # type: ignore
except Exception as e:
    raise SystemExit(
        "Temperature scaling requires NumPy. Install with: pip install numpy\n"
        f"Underlying import error: {e}"
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    # logits: (n_samples, n_classes)
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _nll_from_logits(logits: np.ndarray, y_idx: np.ndarray) -> float:
    # logits: (n_samples, n_classes); y_idx: (n_samples,)
    probs = _softmax(logits)
    # Clip for stability
    probs = np.clip(probs, 1e-12, 1.0)
    chosen = probs[np.arange(len(y_idx)), y_idx]
    return float(-np.mean(np.log(chosen)))


class TemperatureScaledModel:
    """
    Wraps a fitted base estimator and applies temperature scaling at inference.

    - Exposes predict and predict_proba that operate on temperature-scaled logits.
    - Stores classes_ for downstream compatibility.
    """

    def __init__(self, base_estimator, temperature: float = 1.0):
        self.base_estimator = base_estimator
        self.temperature = float(max(1e-6, temperature))
        # Mirror classes_ if available for consumer code
        self.classes_ = getattr(base_estimator, "classes_", None)

    def _get_logits(self, X) -> np.ndarray:
        # Prefer decision_function; fallback to log of probabilities
        if hasattr(self.base_estimator, "decision_function"):
            logits = self.base_estimator.decision_function(X)
            # Ensure 2D shape
            if logits.ndim == 1:
                logits = np.stack([-logits, logits], axis=1)
            return logits.astype(float)
        elif hasattr(self.base_estimator, "predict_proba"):
            proba = self.base_estimator.predict_proba(X)
            proba = np.clip(proba, 1e-12, 1.0)
            return np.log(proba).astype(float)
        else:
            raise RuntimeError("Base estimator must implement decision_function or predict_proba for temperature scaling")

    def fit(self, X_val, y_val: List[str]) -> "TemperatureScaledModel":
        # Map y_val to indices based on base_estimator.classes_
        classes = list(getattr(self.base_estimator, "classes_", []))
        if not classes:
            raise RuntimeError("Base estimator must have classes_ attribute after fitting")
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx.get(str(y), None) for y in y_val], dtype=float)
        if np.any(np.isnan(y_idx)):
            raise ValueError("y_val contains labels not seen during training")
        y_idx = y_idx.astype(int)

        logits = self._get_logits(X_val)

        # Coarse-to-fine search for T that minimizes NLL
        best_T = 1.0
        best_loss = math.inf
        # Coarse grid
        for T in np.linspace(0.5, 5.0, num=91):  # step ~0.05
            loss = _nll_from_logits(logits / T, y_idx)
            if loss < best_loss:
                best_loss = loss
                best_T = float(T)
        # Fine search around best
        lo = max(0.1, best_T - 0.5)
        hi = best_T + 0.5
        for T in np.linspace(lo, hi, num=101):  # step ~0.01
            loss = _nll_from_logits(logits / T, y_idx)
            if loss < best_loss:
                best_loss = loss
                best_T = float(T)

        self.temperature = float(max(1e-6, best_T))
        return self

    def predict_proba(self, X):
        logits = self._get_logits(X) / self.temperature
        probs = _softmax(logits)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        if self.classes_ is not None:
            return np.array([self.classes_[i] for i in idx])
        return idx
