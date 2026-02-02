import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The mean absolute error.
    """
    return mean_absolute_error(y_true, y_pred)


def save_submission(segment_ids, predictions, save_path: str = Config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        segment_ids (array-like): List or array of segment IDs.
        predictions (array-like): List or array of predicted time_to_eruption values.
        save_path (str): File path to save the submission CSV. Defaults to Config.SUBMISSION_PATH.
    """
    # Ensure the directory exists
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Create DataFrame matching the sample submission format
    df = pd.DataFrame({"segment_id": segment_ids, "time_to_eruption": predictions})

    # Save to CSV without the index
    df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
