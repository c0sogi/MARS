import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations may impact performance, but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Checks for GPU availability and returns the appropriate PyTorch device.

    Returns:
        torch.device: The device object ('cuda' or 'cpu').
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def compute_rmse(y_true, y_pred) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The RMSE value.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return float(rmse)


def save_submission(keys, fare_amounts, output_path: str):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Format:
        key,fare_amount
        2015-01-27 13:08:24.0000002,11.00
        ...

    Args:
        keys (array-like): The unique key identifiers for the test set.
        fare_amounts (array-like): The predicted fare amounts.
        output_path (str): The file path where the submission CSV will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame({"key": keys, "fare_amount": fare_amounts})

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
