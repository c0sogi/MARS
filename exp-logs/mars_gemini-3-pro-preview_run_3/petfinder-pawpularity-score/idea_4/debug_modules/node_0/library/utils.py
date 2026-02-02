import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (np.ndarray): Array of ground truth values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
        float: The RMSE score.
    """
    # squared=False returns RMSE
    return mean_squared_error(y_true, y_pred, squared=False)


def save_submission(ids: list, predictions: list, path: str = None):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        ids (list): List of Pet Profile IDs.
        predictions (list): List of predicted Pawpularity scores.
        path (str, optional): File path to save the submission.
                              Defaults to Config.SUBMISSION_PATH.
    """
    if path is None:
        path = Config.SUBMISSION_PATH

    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame({"Id": ids, "Pawpularity": predictions})

    # Save to CSV
    submission_df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
