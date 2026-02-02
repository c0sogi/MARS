import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library import config


def enforce_float64(data):
    """
    Strictly casts data structures to double precision (float64) to minimize numerical noise.

    Args:
        data: numpy array, pandas DataFrame, pandas Series, or list.

    Returns:
        The data cast to float64 type.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.astype("float64")
    elif isinstance(data, np.ndarray):
        return data.astype(np.float64)
    else:
        return np.array(data, dtype=np.float64)


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss matching the competition metric.

    The metric rescales probabilities to sum to 1 per row, then clips them
    to [1e-15, 1-1e-15] before calculating the negative log likelihood.

    Args:
        y_true: Ground truth labels (1D array of class labels or indices).
        y_pred: Predicted probabilities (2D array: n_samples x n_classes).

    Returns:
        float: The log loss value.
    """
    # Ensure float64 precision
    y_pred = enforce_float64(y_pred)

    # 1. Rescale rows to sum to 1
    # We add a safety check for row_sums being 0 to avoid NaNs, though unlikely in valid models
    row_sums = y_pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip probabilities to avoid log(0)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, epsilon, 1 - epsilon)

    # 3. Calculate Log Loss
    # sklearn's log_loss handles the label matching (string or int) automatically
    score = log_loss(y_true, y_pred_clipped)

    return score


def format_submission(
    test_ids, classes, predictions, output_path=config.SUBMISSION_PATH
):
    """
    Formats and saves the submission file.

    Args:
        test_ids: Array-like of image IDs.
        classes: List of class names (strings) corresponding to the columns of predictions.
        predictions: 2D array of probabilities (n_samples x n_classes).
        output_path: Path to save the CSV.
    """
    # Ensure float64 for precision
    predictions = enforce_float64(predictions)

    # Create DataFrame
    submission_df = pd.DataFrame(predictions, columns=classes)

    # Insert ID column at the beginning
    submission_df.insert(0, "id", test_ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
