import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from modeling.error_analysis import compute_calibration_bins, expected_calibration_error, maximum_calibration_error


def test_calibration_bins_perfect_calibration():
    # 4 samples, predicted max probs equal to correctness
    y_true = ["POS", "NEG", "NEU", "POS"]
    y_pred = ["POS", "NEG", "NEU", "NEG"]  # last is wrong
    # Probabilities: confident and aligned with y_pred
    probs = np.array([
        [0.9, 0.05, 0.05],  # correct
        [0.1, 0.8, 0.1],    # correct
        [0.05, 0.05, 0.9],  # correct
        [0.7, 0.2, 0.1],    # incorrect (POS true, POS prob 0.7 but pred is POS? Wait: y_pred says NEG; fix to force pred NEG)
    ])
    # Adjust last row so class 1 (NEG) is max
    probs[-1] = np.array([0.2, 0.7, 0.1])

    bins = compute_calibration_bins(y_true, y_pred, probs, n_bins=5)
    assert isinstance(bins, list) and len(bins) == 5
    # ECE should be >= 0 but not NaN
    ece = expected_calibration_error(bins, total=len(y_true))
    mce = maximum_calibration_error(bins)
    assert ece >= 0.0
    assert mce >= 0.0


def test_calibration_bins_edge_cases():
    y_true = []
    y_pred = []
    probs = np.zeros((0, 3))
    bins = compute_calibration_bins(y_true, y_pred, probs, n_bins=10)
    assert bins == []
    assert expected_calibration_error(bins, total=0) == 0.0
    assert maximum_calibration_error(bins) == 0.0


def test_calibration_bins_uncalibrated_behavior():
    # Construct a simple case where confidence is higher than accuracy
    y_true = ["POS"] * 50 + ["NEG"] * 50
    y_pred = ["POS"] * 50 + ["NEG"] * 50
    # Make 20% of predictions wrong but keep high confidence
    # Flip 10 POS and 10 NEG labels in y_true to induce errors
    y_true[:10] = ["NEG"] * 10
    y_true[50:60] = ["POS"] * 10
    # Predict probabilities with high confidence 0.9 for predicted class
    probs = []
    for i in range(100):
        if y_pred[i] == "POS":
            probs.append([0.9, 0.05, 0.05])
        else:
            probs.append([0.05, 0.9, 0.05])
    probs = np.array(probs)

    bins = compute_calibration_bins(y_true, y_pred, probs, n_bins=10)
    ece = expected_calibration_error(bins, total=len(y_true))
    mce = maximum_calibration_error(bins)
    assert ece > 0.0
    assert mce > 0.0
