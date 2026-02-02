import os
import random
import numpy as np
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and OS environments.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def load_sensor_data(file_path, fill_na=True):
    """
    Loads a sensor data file from the given path.

    Args:
        file_path (str): The full path to the CSV file.
        fill_na (bool): If True, fills missing values with the column-wise mean
                        of the segment to preserve DC offsets.

    Returns:
        pd.DataFrame: The loaded sensor data with float32 precision.
    """
    # Load data with float32 precision to handle NaNs and optimize memory
    df = pd.read_csv(file_path, dtype="float32")

    if fill_na:
        # Calculate mean for each sensor column in this segment
        # This preserves the DC offset specific to this time window
        means = df.mean(axis=0)

        # Fill missing values with the calculated means
        df = df.fillna(means)

        # If any NaNs remain (e.g., if a whole column was NaN), fill with 0
        df = df.fillna(0.0)

    return df


def compute_mae(y_true, y_pred):
    """
    Computes the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: The computed MAE.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def save_submission(segment_ids, predictions, output_path=Config.SUBMISSION_PATH):
    """
    Formats and saves the submission file.

    Args:
        segment_ids (array-like): List or array of segment IDs.
        predictions (array-like): List or array of predicted time_to_eruption values.
        output_path (str): Path to save the CSV file. Defaults to Config.SUBMISSION_PATH.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame conforming to the submission format
    submission_df = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": predictions}
    )

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
