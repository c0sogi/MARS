import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import seed_everything


def calculate_weighted_loss_metric(y_true, y_pred, epsilon=1e-15):
    """
    Calculates the weighted multi-label logarithmic loss for the RSNA Cervical Spine Fracture Detection task.

    The metric is calculated as the weighted average of the binary log loss for each target column.
    Weights are set to 1.0 for each vertebrae (C1-C7) and 7.0 for the 'patient_overall' label,
    reflecting the importance of the patient-level outcome.

    Args:
        y_true (pd.DataFrame or np.ndarray): True labels.
            Expected columns/order: ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'patient_overall'].
        y_pred (pd.DataFrame or np.ndarray): Predicted probabilities.
            Expected columns/order: ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'patient_overall'].
        epsilon (float): Clipping value for predictions to avoid log(0). Default is 1e-15.

    Returns:
        float: The calculated weighted log loss.
    """
    # Define weights based on the 1:7 ratio strategy
    weights = {
        "C1": 1.0,
        "C2": 1.0,
        "C3": 1.0,
        "C4": 1.0,
        "C5": 1.0,
        "C6": 1.0,
        "C7": 1.0,
        "patient_overall": 7.0,
    }

    # Target columns in order
    cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    # Standardize inputs to DataFrames
    if isinstance(y_true, np.ndarray):
        y_true = pd.DataFrame(y_true, columns=cols)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred, columns=cols)

    # Clip predictions to prevent NaN/Inf in log loss
    y_pred = y_pred.clip(epsilon, 1 - epsilon)

    total_loss = 0.0
    total_weight = 0.0

    for col in cols:
        # Check if column exists in both inputs
        if col not in y_true.columns or col not in y_pred.columns:
            continue

        # Calculate binary log loss for this specific class
        # log_loss returns the mean loss over all samples for this class
        class_loss = log_loss(y_true[col], y_pred[col], labels=[0, 1])

        w = weights.get(col, 1.0)
        total_loss += class_loss * w
        total_weight += w

    # Normalize by the sum of weights
    if total_weight == 0:
        return 0.0

    return total_loss / total_weight
