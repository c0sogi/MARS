import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and os environments.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def to_float64(data):
    """
    Converts input data to float64 precision to minimize numerical noise.

    Args:
        data: Input data (list, numpy array, pandas Series, or DataFrame).

    Returns:
        Data converted to float64 type.
    """
    if isinstance(data, pd.DataFrame):
        return data.astype(np.float64)
    elif isinstance(data, pd.Series):
        return data.astype(np.float64)
    elif isinstance(data, np.ndarray):
        return data.astype(np.float64)
    else:
        return np.array(data, dtype=np.float64)


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss with specific normalization and clipping
    as defined in the competition metric.

    The probabilities are first rescaled so each row sums to 1.
    Then they are clipped to the range [1e-15, 1-1e-15].

    Args:
        y_true: Ground truth labels (1D array-like).
        y_pred: Predicted probabilities (2D array-like).

    Returns:
        float: The calculated log loss.
    """
    # Ensure float64 precision
    y_pred = to_float64(y_pred)

    # Rescale prior to scoring: each row is divided by the row sum
    row_sums = y_pred.sum(axis=1)
    # Avoid division by zero by replacing 0 sums with 1 (though 0 sum shouldn't happen in valid probs)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # Clip probabilities to avoid extremes of the log function
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate Log Loss
    return log_loss(y_true, y_pred)


def save_submission(ids, classes, probs, output_path):
    """
    Formats and saves the submission file in the required CSV format.

    Args:
        ids: 1D array-like of image ids.
        classes: List of class names corresponding to the columns of probs.
        probs: 2D array-like of predicted probabilities.
        output_path: Path to save the CSV file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(probs, columns=classes)

    # Insert id column at the beginning
    df.insert(0, "id", ids)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
