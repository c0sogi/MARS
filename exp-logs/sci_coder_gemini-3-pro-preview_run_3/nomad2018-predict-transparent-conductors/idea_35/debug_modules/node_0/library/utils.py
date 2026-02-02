import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from library.config import SUBMISSION_DIR, TARGET_COLS


def log1p_transform(y):
    """
    Applies the log1p transformation (log(1 + x)) to the input data.
    Useful for stabilizing variance in target variables like energy.

    Args:
        y (array-like): Input data.

    Returns:
        np.ndarray: Log-transformed data.
    """
    return np.log1p(y)


def expm1_transform(y):
    """
    Applies the expm1 transformation (exp(x) - 1) to the input data.
    Used to inverse the log1p transformation for final predictions.

    Args:
        y (array-like): Input data (log-scale).

    Returns:
        np.ndarray: Original scale data.
    """
    return np.expm1(y)


def calculate_rmsle(y_true, y_pred):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    The metric is defined as the average of the RMSLE calculated for each
    target column independently.

    Args:
        y_true (array-like): Ground truth values (original scale).
        y_pred (array-like): Predicted values (original scale).

    Returns:
        float: The mean RMSLE across all target columns.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to be non-negative to avoid errors in log1p
    # Energies cannot be negative in this context (or at least log is undefined for <= -1)
    # Assuming targets are formation energy (can be small/negative?) and bandgap (positive).
    # Based on EDA, targets are non-negative.
    y_pred = np.maximum(y_pred, 0)

    # Calculate Squared Logarithmic Error for each element
    squared_log_errors = (np.log1p(y_true) - np.log1p(y_pred)) ** 2

    # Calculate Mean Squared Logarithmic Error for each column
    msle_per_column = np.mean(squared_log_errors, axis=0)

    # Calculate Root Mean Squared Logarithmic Error for each column
    rmsle_per_column = np.sqrt(msle_per_column)

    # Return the average of the column-wise RMSLEs
    return np.mean(rmsle_per_column)


def save_submission(ids, predictions, filename="submission.csv"):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        ids (array-like): Sequence of IDs corresponding to the predictions.
        predictions (array-like): Matrix of predictions with shape (n_samples, n_targets).
                                  Columns must correspond to TARGET_COLS order.
        filename (str): Name of the output CSV file.
    """
    # Ensure the submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(predictions, columns=TARGET_COLS)

    # Insert ID column at the beginning
    submission_df.insert(0, "id", ids)

    # Construct full path
    output_path = os.path.join(SUBMISSION_DIR, filename)

    # Save to CSV without index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
