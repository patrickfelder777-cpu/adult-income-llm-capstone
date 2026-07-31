"""Evaluation utilities for binary classification models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_probability: Any,
) -> dict[str, float]:
    """
    Calculate classification metrics for the positive income class.

    Parameters
    ----------
    y_true:
        Actual binary target values.
    y_pred:
        Predicted binary target values.
    y_probability:
        Predicted probabilities for the positive class.

    Returns
    -------
    dict[str, float]
        Accuracy, precision, recall, F1, and ROC-AUC scores.
    """
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    probabilities = np.asarray(y_probability)

    if true_values.shape[0] != predicted_values.shape[0]:
        raise ValueError(
            "y_true and y_pred must contain the same number of values."
        )

    if true_values.shape[0] != probabilities.shape[0]:
        raise ValueError(
            "y_true and y_probability must contain the same "
            "number of values."
        )

    metrics = {
        "accuracy": accuracy_score(
            true_values,
            predicted_values,
        ),
        "precision": precision_score(
            true_values,
            predicted_values,
            zero_division=0,
        ),
        "recall": recall_score(
            true_values,
            predicted_values,
            zero_division=0,
        ),
        "f1": f1_score(
            true_values,
            predicted_values,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            true_values,
            probabilities,
        ),
    }

    return {
        metric_name: float(metric_value)
        for metric_name, metric_value in metrics.items()
    }