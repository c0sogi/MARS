import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from library.config import Config

# Set seeds for reproducibility
np.random.seed(Config.SEED)


def load_sensor_data(file_path):
    """
    Safely reads a sensor CSV file with float32 precision.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded sensor data. Returns an empty DataFrame on failure.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load data as float32 to handle potential nulls and reduce memory usage
        # The prompt notes that even normalized int16 data might need float32 due to nulls
        df = pd.read_csv(file_path, dtype="float32")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def compute_metric(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The MAE score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    return mae


def save_submission(segment_ids, predictions, filename="submission.csv"):
    """
    Formats and writes the final predictions to a CSV file.

    Args:
        segment_ids (array-like): List or array of segment IDs.
        predictions (array-like): List or array of predicted times.
        filename (str): Name of the output file. Defaults to "submission.csv".
    """
    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Create DataFrame with required columns
    submission_df = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": predictions}
    )

    # Construct full path
    save_path = os.path.join(Config.SUBMISSION_DIR, filename)

    # Save to CSV without index
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
