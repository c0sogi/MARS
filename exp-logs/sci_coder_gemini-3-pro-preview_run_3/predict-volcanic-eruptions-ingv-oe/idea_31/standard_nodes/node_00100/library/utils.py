import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame with float32 precision.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load with float32 to handle potential nulls and optimize memory usage
    return pd.read_csv(file_path, dtype="float32")


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (np.ndarray): Array of ground truth values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
        float: The Mean Absolute Error.
    """
    return mean_absolute_error(y_true, y_pred)
