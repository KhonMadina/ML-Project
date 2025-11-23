from __future__ import annotations

"""Evaluation utilities for Khmer sentiment models.

This module centralizes common metric computations so that baseline and
transformer training scripts can share a consistent evaluation layer.

Initial version mirrors the metrics used in train_baseline.py.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

try:  # optional dependency
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )  # type: ignore
except Exception as e:  # pragma: no cover - surfaced at call time
    raise SystemExit(
        "Missing dependency for evaluation metrics. Install with: pip install scikit-learn\n"
        f"Underlying import error: {e}"
    )

LABEL_ORDER_DEFAULT: List[str] = ["POS", "NEG", "NEU"]


@dataclass
class SplitMetrics:
    """Container for metrics on a single split (val/test)."""

    accuracy: float | None
    f1_macro: float | None
    report: Dict[str, object]
    confusion: List[List[int]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "report": self.report,
            "confusion": self.confusion,
        }


def evaluate_classification(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> SplitMetrics:
    """Compute accuracy, macro-F1, report, and confusion matrix.

    This is a thin wrapper around scikit-learn metrics that matches the
    semantics previously embedded in train_baseline.py.
    """

    if labels is None:
        labels = LABEL_ORDER_DEFAULT

    if not y_true:
        # No data: mirror the previous behavior of returning Nones/zeros.
        return SplitMetrics(
            accuracy=None,
            f1_macro=None,
            report={},
            confusion=[[0 for _ in labels] for _ in labels],
        )

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels)).tolist()

    return SplitMetrics(accuracy=acc, f1_macro=f1m, report=report, confusion=cm)