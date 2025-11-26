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


# -----------------------------
# Calibration metrics and plots
# -----------------------------

def probs_from_logits(logits: "np.ndarray") -> "np.ndarray":
    """Convert logits to probabilities via softmax.

    Parameters
    ----------
    logits : np.ndarray of shape (n_samples, n_classes)

    Returns
    -------
    np.ndarray of shape (n_samples, n_classes)
    """
    return _softmax(logits)


def compute_brier_score(probs: "np.ndarray", y_idx: "np.ndarray") -> float:
    """Compute multi-class Brier score.

    Parameters
    ----------
    probs : (n_samples, n_classes)
    y_idx : (n_samples,) integer labels
    """
    n = probs.shape[0]
    k = probs.shape[1]
    # Build one-hot
    oh = np.zeros_like(probs)
    oh[np.arange(n), y_idx] = 1.0
    diff = probs - oh
    brier = float(np.mean(np.sum(diff * diff, axis=1)))
    return brier


def reliability_bins(probs: "np.ndarray", y_idx: "np.ndarray", n_bins: int = 15):
    """Compute reliability bins for ECE/diagram using max-prob predictions.

    Returns a list of dicts with keys: bin_lower, bin_upper, count, accuracy, confidence.
    """
    preds = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (preds == y_idx).astype(float)

    # Bin edges in [0, 1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            acc = 0.0
            avg_conf = 0.0
        else:
            acc = float(np.mean(correct[mask]))
            avg_conf = float(np.mean(conf[mask]))
        out.append({
            "bin_lower": lo,
            "bin_upper": hi,
            "count": cnt,
            "accuracy": acc,
            "confidence": avg_conf,
        })
    return out


def compute_ece(probs: "np.ndarray", y_idx: "np.ndarray", n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) using max-prob confidence bins.
    Weighs bin gaps by bin frequency.
    """
    bins = reliability_bins(probs, y_idx, n_bins=n_bins)
    n = probs.shape[0]
    ece = 0.0
    for b in bins:
        if b["count"] <= 0:
            continue
        w = b["count"] / n
        ece += w * abs(b["accuracy"] - b["confidence"])
    return float(ece)


def save_reliability_bins(bins, json_path: str | None = None, csv_path: str | None = None) -> None:
    """Save reliability bin statistics to JSON and/or CSV."""
    if json_path:
        try:
            import json as _json
            with open(json_path, "w", encoding="utf-8") as f:
                _json.dump(bins, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    if csv_path:
        try:
            import csv as _csv
            keys = ["bin_lower", "bin_upper", "count", "accuracy", "confidence"]
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for b in bins:
                    w.writerow({k: b.get(k, "") for k in keys})
        except Exception:
            pass


def plot_reliability_diagram(bins, output_png: str, title: str | None = None) -> bool:
    """Plot reliability diagram. Returns True if saved successfully, else False."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    try:
        conf = [b["confidence"] for b in bins]
        acc = [b["accuracy"] for b in bins]
        width = [b["bin_upper"] - b["bin_lower"] for b in bins]
        centers = [b["bin_lower"] + w / 2 for b, w in zip(bins, width)]

        plt.figure(figsize=(5, 5), dpi=120)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        plt.bar(centers, acc, width=width, alpha=0.6, align="center", edgecolor="black", label="Empirical accuracy")
        plt.plot(centers, conf, color="C1", marker="o", label="Average confidence")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        if title:
            plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_png)
        plt.close()
        return True
    except Exception:
        return False


def compute_calibration_summary(
    probs: "np.ndarray" | None = None,
    logits: "np.ndarray" | None = None,
    y_idx: "np.ndarray" | None = None,
    n_bins: int = 15,
    diagram_png: str | None = None,
    bins_json: str | None = None,
    bins_csv: str | None = None,
    title: str | None = None,
):
    """Compute calibration metrics and optionally save reliability artifacts.

    Provide either `probs` or `logits`. Returns a dict with keys: ece, brier, n_samples, n_classes.
    """
    if probs is None and logits is None:
        raise ValueError("Provide either probs or logits")
    if y_idx is None:
        raise ValueError("Provide y_idx integer labels")
    if probs is None:
        probs = probs_from_logits(logits)  # type: ignore[arg-type]
    if probs is None:
        raise ValueError("Failed to derive probabilities")

    probs = np.asarray(probs, dtype=float)
    y_idx = np.asarray(y_idx, dtype=int)
    if probs.ndim != 2:
        raise ValueError("probs must be 2D (n_samples, n_classes)")
    if y_idx.ndim != 1 or y_idx.shape[0] != probs.shape[0]:
        raise ValueError("y_idx must be 1D and match n_samples in probs")

    ece = compute_ece(probs, y_idx, n_bins=n_bins)
    brier = compute_brier_score(probs, y_idx)
    bins = reliability_bins(probs, y_idx, n_bins=n_bins)

    if diagram_png:
        plot_reliability_diagram(bins, diagram_png, title=title)
    if bins_json or bins_csv:
        save_reliability_bins(bins, json_path=bins_json, csv_path=bins_csv)

    return {
        "ece": float(ece),
        "brier": float(brier),
        "n_samples": int(probs.shape[0]),
        "n_classes": int(probs.shape[1]),
        "n_bins": int(n_bins),
    }
