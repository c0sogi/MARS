import os
import random
import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error
from library import config


def seed_everything(seed: int = config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: The RMSE value.
    """
    return root_mean_squared_error(y_true, y_pred)


def save_numpy_array(path: str, array: np.ndarray):
    """
    Saves a numpy array to the specified path, ensuring the directory exists.

    Args:
        path (str): The full file path to save the .npy file.
        array (np.ndarray): The numpy array to save.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(path, array)


def load_numpy_array(path: str) -> np.ndarray:
    """
    Loads a numpy array from the specified path.

    Args:
        path (str): The full file path to load the .npy file from.

    Returns:
        np.ndarray: The loaded numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found at {path}")
    return np.load(path, allow_pickle=True)
