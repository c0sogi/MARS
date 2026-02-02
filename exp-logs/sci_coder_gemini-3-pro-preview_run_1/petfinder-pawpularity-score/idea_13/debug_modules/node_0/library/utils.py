import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE).

    Args:
        y_true (np.ndarray): Array of true target values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
        float: The RMSE score.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def save_array(filename: str, array: np.ndarray) -> None:
    """
    Saves a numpy array to the configured cache directory.

    Args:
        filename (str): The name of the file (e.g., 'features.npy').
        array (np.ndarray): The numpy array to save.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    file_path = os.path.join(Config.CACHE_DIR, filename)
    np.save(file_path, array)


def load_array(filename: str) -> np.ndarray:
    """
    Loads a numpy array from the configured cache directory.

    Args:
        filename (str): The name of the file to load (e.g., 'features.npy').

    Returns:
        np.ndarray: The loaded numpy array.

    Raises:
        FileNotFoundError: If the file does not exist in the cache.
    """
    file_path = os.path.join(Config.CACHE_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cached file not found: {file_path}")

    return np.load(file_path)


def check_cache_exists(filename: str) -> bool:
    """
    Checks if a file exists in the cache directory.

    Args:
        filename (str): The name of the file to check.

    Returns:
        bool: True if file exists, False otherwise.
    """
    file_path = os.path.join(Config.CACHE_DIR, filename)
    return os.path.exists(file_path)


def create_submission(ids: np.ndarray, predictions: np.ndarray) -> None:
    """
    Creates the submission DataFrame and saves it to the configured path.

    Args:
        ids (np.ndarray): Array of Pet Profile IDs.
        predictions (np.ndarray): Array of predicted Pawpularity scores.
    """
    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame({Config.ID_COL: ids, Config.TARGET_COL: predictions})

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
