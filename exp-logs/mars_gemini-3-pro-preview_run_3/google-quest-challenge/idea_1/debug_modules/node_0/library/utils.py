import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_spearman_score(y_true, y_pred, target_cols):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        y_true: Ground truth target values (pd.DataFrame or np.ndarray).
        y_pred: Predicted target values (pd.DataFrame or np.ndarray).
        target_cols: List of target column names.

    Returns:
        float: The mean Spearman's correlation score.
    """
    # Convert inputs to numpy arrays if they are DataFrames
    if isinstance(y_true, pd.DataFrame):
        y_true_vals = y_true[target_cols].values
    else:
        y_true_vals = y_true

    if isinstance(y_pred, pd.DataFrame):
        y_pred_vals = y_pred[target_cols].values
    else:
        y_pred_vals = y_pred

    # Ensure shapes match
    if y_true_vals.shape != y_pred_vals.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_vals.shape} vs y_pred {y_pred_vals.shape}"
        )

    correlations = []
    num_targets = y_true_vals.shape[1]

    for i in range(num_targets):
        # Extract columns
        col_true = y_true_vals[:, i]
        col_pred = y_pred_vals[:, i]

        # Compute Spearman correlation
        # spearmanr returns a result object or tuple. accessing the statistic/correlation.
        # We handle the case where constant input might produce NaN.
        try:
            res = spearmanr(col_true, col_pred)
            # Check if res is a tuple or object (scipy version dependent)
            if hasattr(res, "statistic"):
                corr = res.statistic
            else:
                corr = res[0]
        except Exception:
            corr = np.nan

        correlations.append(corr)

    # Compute mean, ignoring NaNs (which occur if a target is constant)
    score = np.nanmean(correlations)

    return score


def save_submission(y_pred, test_ids, target_cols, output_path):
    """
    Formats and saves the predictions to a CSV file.

    Args:
        y_pred: Predicted probabilities (np.ndarray or pd.DataFrame).
        test_ids: Array-like of qa_ids corresponding to the predictions.
        target_cols: List of target column names.
        output_path: Path to save the submission CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare DataFrame
    if isinstance(y_pred, pd.DataFrame):
        # If it's already a DataFrame, ensure columns match target_cols
        # If columns are missing or different, we assume the order matches target_cols
        # to be safe, we reconstruct.
        df = pd.DataFrame(y_pred.values, columns=target_cols)
    else:
        df = pd.DataFrame(y_pred, columns=target_cols)

    # Insert qa_id at the first position
    # Flatten test_ids to ensure it matches index
    df.insert(0, "qa_id", np.array(test_ids).flatten())

    # Save to CSV
    df.to_csv(output_path, index=False)
