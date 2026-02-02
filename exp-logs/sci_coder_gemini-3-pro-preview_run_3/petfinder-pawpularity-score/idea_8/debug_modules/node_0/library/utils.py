import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def get_device() -> torch.device:
    """
    Returns the appropriate torch device (CUDA if available, else CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_cache(data: np.ndarray, filename: str):
    """
    Saves a numpy array to the working directory defined in Config.

    Args:
        data (np.ndarray): The data to save.
        filename (str): The name of the file (e.g., 'features.npy').
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    file_path = os.path.join(Config.WORKING_DIR, filename)
    np.save(file_path, data)


def load_cache(filename: str):
    """
    Attempts to load a numpy array from the working directory.

    Args:
        filename (str): The name of the file to load.

    Returns:
        np.ndarray or None: The loaded data if the file exists, else None.
    """
    file_path = os.path.join(Config.WORKING_DIR, filename)

    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)

    return None
