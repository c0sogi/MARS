import os
import random
import numpy as np
import torch
from library import config


def seed_everything(seed: int = config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
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


def save_array(array: np.ndarray, file_path: str):
    """
    Saves a numpy array to the specified path. Creates directories if they don't exist.

    Args:
        array (np.ndarray): The numpy array to save.
        file_path (str): The destination file path.
    """
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(file_path, array)


def load_array(file_path: str) -> np.ndarray:
    """
    Loads a numpy array from the specified path.

    Args:
        file_path (str): The path to the numpy file.

    Returns:
        np.ndarray: The loaded numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return np.load(file_path)
