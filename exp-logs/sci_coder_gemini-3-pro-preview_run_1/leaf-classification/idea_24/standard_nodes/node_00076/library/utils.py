import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library import config


def enforce_float64(data):
    """
    Ensures the input data is in high-precision float64 format.

    Args:
        data: numpy array, pandas Series, or pandas DataFrame.

    Returns:
        Data converted to config.FLOAT_PRECISION (np.float64).
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.astype(config.FLOAT_PRECISION)
    elif isinstance(data, np.ndarray):
        return data.astype(config.FLOAT_PRECISION)
    else:
        # Fallback for lists or other iterables
        return np.array(data, dtype=config.FLOAT_PRECISION)


def calculate_log_loss(y_true, y_pred_probs, class_labels):
    """
    Calculates the Multi-class Log Loss metric with high precision.

    Mimics the competition metric:
    1. Rescales rows to sum to 1 (each row is divided by the row sum).
    2. Clips probabilities to [1e-15, 1-1e-15].
    3. Calculates log loss.

    Args:
        y_true: Ground truth labels (1D array-like of strings or integers).
        y_pred_probs: Predicted probabilities (2D array-like, N_samples x N_classes).
        class_labels: List of class names corresponding to the columns of y_pred_probs.

    Returns:
        float: The calculated log loss.
    """
    # Enforce float64 for precision
    y_pred_probs = enforce_float64(y_pred_probs)

    # 1. Rescale prior to being scored (each row is divided by the row sum)
    # Add epsilon to denominator to avoid division by zero if row sum is 0 (though unlikely)
    row_sums = y_pred_probs.sum(axis=1)
    # Handle potential zero sums to avoid NaN
    row_sums[row_sums == 0] = 1.0
    y_pred_probs = y_pred_probs / row_sums[:, np.newaxis]

    # 2. Predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)
    y_pred_probs = np.clip(y_pred_probs, config.PROB_CLIP_MIN, config.PROB_CLIP_MAX)

    # Calculate Log Loss
    # sklearn log_loss handles string labels in y_true if labels are provided
    loss = log_loss(y_true, y_pred_probs, labels=class_labels)

    # Print full precision for debugging/validation
    print(f"Validation Multi-class Log Loss: {loss:.20f}")

    return loss


def format_submission(
    test_ids, y_pred_probs, class_labels, output_path=config.SUBMISSION_FILE_PATH
):
    """
    Formats predictions into the required CSV format and saves to disk.

    Args:
        test_ids: 1D array-like of image IDs.
        y_pred_probs: 2D array-like of predicted probabilities.
        class_labels: List of class names (columns).
        output_path: Path to save the submission CSV.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    # Enforce float64
    y_pred_probs = enforce_float64(y_pred_probs)

    # Clip probabilities (Ensure submission is compliant with metric logic)
    y_pred_probs = np.clip(y_pred_probs, config.PROB_CLIP_MIN, config.PROB_CLIP_MAX)

    # Create DataFrame
    submission_df = pd.DataFrame(y_pred_probs, columns=class_labels)

    # Insert ID column at the beginning, ensuring integer type for IDs
    # Convert ids to numpy array first to ensure astype works
    ids_array = np.array(test_ids)
    submission_df.insert(0, config.ID_COL, ids_array.astype(int))

    # Ensure output directory exists
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to: {output_path}")

    return submission_df
