import numpy as np
from library.config import set_seed


def calculate_f1_score(y_true, y_pred):
    """
    Calculates the Mean F1-Score (Samples-average) for multi-label classification.

    The metric is computed by treating each sample's tags as a set.
    F1 = 2 * (precision * recall) / (precision + recall)
    Precision = |intersection| / |predicted|
    Recall = |intersection| / |ground_truth|

    Args:
        y_true (list of str): A list of ground truth tag strings (space-delimited).
        y_pred (list of str): A list of predicted tag strings (space-delimited).

    Returns:
        float: The mean F1-score across all samples.
    """
    if not isinstance(y_true, list) or not isinstance(y_pred, list):
        raise ValueError("y_true and y_pred must be lists of strings.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)} elements."
        )

    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    total_f1 = 0.0

    for t_str, p_str in zip(y_true, y_pred):
        # Handle potential non-string inputs (e.g. None or NaN floats) safely
        if not isinstance(t_str, str):
            t_str = ""
        if not isinstance(p_str, str):
            p_str = ""

        # Convert space-delimited strings to sets
        true_tags = set(t_str.split())
        pred_tags = set(p_str.split())

        len_true = len(true_tags)
        len_pred = len(pred_tags)

        # Edge case: Both sets are empty -> Perfect match (F1 = 1.0)
        if len_true == 0 and len_pred == 0:
            total_f1 += 1.0
            continue

        # Edge case: One set is empty but not the other -> F1 = 0.0
        if len_true == 0 or len_pred == 0:
            total_f1 += 0.0
            continue

        # Calculate Intersection
        intersection = len(true_tags & pred_tags)

        # Calculate Precision and Recall
        precision = intersection / len_pred
        recall = intersection / len_true

        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            total_f1 += f1

    return total_f1 / n_samples
