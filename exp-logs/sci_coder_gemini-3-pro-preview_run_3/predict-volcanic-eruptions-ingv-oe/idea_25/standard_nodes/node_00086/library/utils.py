import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from library.config import Config

# Try importing torch to set seeds if available, as it is a common dependency
try:
    import torch
except ImportError:
    torch = None


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The MAE score.
    """
    return mean_absolute_error(y_true, y_pred)


def save_submission(predictions, test_df, output_path=Config.SUBMISSION_PATH):
    """
    Formats and saves the submission file according to the competition requirements.

    Args:
        predictions (array-like): Predicted time_to_eruption values.
                                  Must correspond to the order of segment_ids in test_df.
        test_df (pd.DataFrame): DataFrame containing the 'segment_id' column for the test set.
        output_path (str): Path to save the CSV file. Defaults to Config.SUBMISSION_PATH.
    """
    # Ensure the directory for the submission file exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create the submission DataFrame
    submission = pd.DataFrame()
    submission["segment_id"] = test_df["segment_id"]
    submission["time_to_eruption"] = predictions

    # Save to CSV without the index
    submission.to_csv(output_path, index=False)
    log_message(f"Submission saved to {output_path}")


def log_message(message):
    """
    Logs a message to the standard output.

    Args:
        message (str): The message string to print.
    """
    print(message)
