import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from library.config import Config


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: The RMSE value.
    """
    # Ensure inputs are numpy arrays to handle lists or pandas series
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.sqrt(mean_squared_error(y_true, y_pred))


def format_submission(keys, predictions, output_path=None):
    """
    Formats and saves the predictions to a CSV file in the required submission format.

    Args:
        keys (array-like): The 'key' column values from the test set.
        predictions (array-like): The predicted 'fare_amount' values.
        output_path (str, optional): Path to save the submission CSV.
                                     Defaults to Config.SUBMISSION_PATH.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create the submission DataFrame
    # The submission format requires 'key' and 'fare_amount' columns
    submission = pd.DataFrame({"key": keys, "fare_amount": predictions})

    # Save to CSV without the index, as required by the format
    submission.to_csv(output_path, index=False)
