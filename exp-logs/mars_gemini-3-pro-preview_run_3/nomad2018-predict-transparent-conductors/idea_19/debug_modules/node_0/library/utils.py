import numpy as np
import pandas as pd
import os
from library.config import SUBMISSION_DIR


def log_transform(y):
    """
    Applies the natural logarithm transformation log(1 + y) to the input data.
    Useful for normalizing targets with long-tailed distributions or for
    optimizing RMSLE by training on log-transformed targets.

    Args:
        y (np.ndarray or pd.Series): Input array of target values.

    Returns:
        np.ndarray: Log-transformed values.
    """
    return np.log1p(y)


def inverse_log_transform(z):
    """
    Applies the inverse transformation exp(z) - 1 to revert data to the original scale.
    Used to convert model predictions back to physical units.

    Args:
        z (np.ndarray or pd.Series): Log-transformed values.

    Returns:
        np.ndarray: Values in the original scale.
    """
    return np.expm1(z)


def compute_rmsle(y_true, y_pred):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    This metric is calculated by taking the RMSE of the log-transformed (log(1+x))
    true and predicted values for each column, and then averaging these scores.

    Args:
        y_true (np.ndarray): Ground truth values (original scale).
        y_pred (np.ndarray): Predicted values (original scale).

    Returns:
        float: The mean RMSLE across all target columns.
    """
    # Clip predictions to be non-negative to avoid domain errors in log
    y_pred_clipped = np.maximum(y_pred, 0)

    # Calculate log(1+x) for both true and predicted values
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred_clipped)

    # Compute Mean Squared Error in log space for each column
    # axis=0 calculates the mean down the rows, resulting in a value per column
    mse_log = np.mean((log_true - log_pred) ** 2, axis=0)

    # Compute RMSE in log space (which is RMSLE) for each column
    rmsle_per_column = np.sqrt(mse_log)

    # Return the average of the column-wise RMSLEs
    return np.mean(rmsle_per_column)


def save_submission(ids, formation_energy, bandgap_energy, filename="submission.csv"):
    """
    Formats and saves the predictions to a CSV file in the directory specified by config.

    Args:
        ids (list or np.ndarray): List of test set IDs.
        formation_energy (list or np.ndarray): Predicted formation energies.
        bandgap_energy (list or np.ndarray): Predicted bandgap energies.
        filename (str): Name of the output CSV file. Defaults to "submission.csv".
    """
    # Create DataFrame with specified column names
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": formation_energy,
            "bandgap_energy_ev": bandgap_energy,
        }
    )

    # Construct full path using the submission directory from config
    output_path = os.path.join(SUBMISSION_DIR, filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV without index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")
