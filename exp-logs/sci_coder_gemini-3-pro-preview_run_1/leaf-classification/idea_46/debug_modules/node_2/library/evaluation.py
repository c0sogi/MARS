import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import ID_COL, FLOAT_PRECISION


def compute_log_loss(y_true, y_pred, classes):
    """
    Computes the Multi-class Log Loss metric with specific rescaling and clipping
    as defined in the task description.

    The metric specification states:
    1. Probabilities are rescaled (each row divided by row sum).
    2. Probabilities are clipped to [1e-15, 1 - 1e-15].

    Args:
        y_true (array-like): True class labels (strings or encoded integers).
        y_pred (array-like): Predicted probabilities, shape (n_samples, n_classes).
        classes (array-like): List of class names corresponding to the columns of y_pred.

    Returns:
        float: The computed log loss.
    """
    # Ensure predictions are high precision
    y_pred = np.array(y_pred, dtype=FLOAT_PRECISION)

    # 1. Rescale: Each row is divided by the row sum
    # "The submitted probabilities ... are rescaled prior to being scored"
    row_sums = y_pred.sum(axis=1, keepdims=True)

    # Handle potential zero sums to avoid NaN (safety check)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Calculate Log Loss
    # Manual clipping required as 'eps' parameter was removed in sklearn 1.5+
    y_pred_clipped = np.clip(y_pred_norm, 1e-15, 1 - 1e-15)

    # We pass labels to ensure correct alignment if y_true doesn't cover all classes.
    return log_loss(y_true, y_pred_clipped, labels=classes)


def generate_submission_file(ids, probs, classes, output_path):
    """
    Generates a submission CSV file in the required format.

    Args:
        ids (array-like): Vector of image IDs.
        probs (array-like): Matrix of predicted probabilities (n_samples, n_classes).
        classes (array-like): List of class names corresponding to probability columns.
        output_path (str): File path to save the submission.
    """
    # Ensure inputs are numpy arrays/lists and use high precision
    ids = np.array(ids)
    probs = np.array(probs, dtype=FLOAT_PRECISION)

    # Create DataFrame with class names as columns
    df = pd.DataFrame(probs, columns=classes)

    # Insert 'id' column at the start
    df.insert(0, ID_COL, ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV without index
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
