import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import SUBMISSION_PATH, ID_COL


def compute_log_loss(y_true, y_pred, classes):
    """
    Computes the multi-class log loss with specific normalization and clipping
    as defined in the task metric.

    Args:
        y_true (array-like): Ground truth labels (strings or integers).
        y_pred (array-like): Predicted probabilities, shape (n_samples, n_classes).
        classes (list): List of class names corresponding to the columns of y_pred.

    Returns:
        float: The calculated multi-class log loss.
    """
    # Ensure y_pred is a numpy array
    y_pred = np.array(y_pred)

    # 1. Rescale: The submitted probabilities are not required to sum to one
    # because they are rescaled prior to being scored (each row is divided by the row sum).
    row_sums = y_pred.sum(axis=1)
    # Handle potential division by zero if a row sums to 0 (unlikely but safe to handle)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums[:, np.newaxis]

    # 2. Clip: Predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, epsilon, 1 - epsilon)

    # 3. Compute Log Loss
    # We pass 'labels=classes' to ensure y_true is mapped correctly to the columns of y_pred.
    return log_loss(y_true, y_pred_clipped, labels=classes)


def save_submission(ids, y_pred, classes, filename=SUBMISSION_PATH):
    """
    Formats and saves the predictions to a CSV file.

    Args:
        ids (array-like): Sequence of image IDs.
        y_pred (array-like): Predicted probabilities matrix.
        classes (list): List of class names corresponding to columns of y_pred.
        filename (str): Path to save the submission CSV.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(y_pred, columns=classes)

    # Insert ID column at the start
    df.insert(0, ID_COL, ids)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
