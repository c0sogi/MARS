import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_metric(y_true, y_pred) -> float:
    """
    Computes the Multi Class Log Loss.

    Args:
        y_true: Array-like of shape (n_samples,) containing true class indices.
        y_pred: Array-like of shape (n_samples, n_classes) containing predicted probabilities.

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate Log Loss
    # sklearn handles clipping internally (eps=1e-15 by default)
    return log_loss(y_true, y_pred)


def save_array(filename: str, array: np.ndarray) -> None:
    """
    Saves a numpy array to the working directory defined in Config.
    Automatically creates the directory if it does not exist.

    Args:
        filename (str): The name of the file (e.g., 'train_embeddings.npy').
        array (np.ndarray): The numpy array to save.
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    file_path = os.path.join(Config.WORKING_DIR, filename)
    np.save(file_path, array)


def load_array(filename: str):
    """
    Loads a numpy array from the working directory defined in Config.

    Args:
        filename (str): The name of the file to load.

    Returns:
        np.ndarray or None: The loaded array if the file exists, else None.
    """
    file_path = os.path.join(Config.WORKING_DIR, filename)

    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)

    return None


def check_files_exist(filenames: list) -> bool:
    """
    Checks if a list of filenames exists in the working directory.

    Args:
        filenames (list): List of filenames to check.

    Returns:
        bool: True if all files exist, False otherwise.
    """
    for fname in filenames:
        file_path = os.path.join(Config.WORKING_DIR, fname)
        if not os.path.exists(file_path):
            return False
    return True
