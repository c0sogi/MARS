import os
import random
import numpy as np
import torch
import joblib
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # OS environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_object(obj, filepath):
    """
    Saves a Python object (e.g., model, pipeline) to a file using joblib.
    Automatically creates the parent directory if it doesn't exist.

    Args:
        obj: The Python object to save.
        filepath (str): The destination file path.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(obj, filepath)


def load_object(filepath):
    """
    Loads a Python object from a file using joblib.

    Args:
        filepath (str): The path to the file to load.

    Returns:
        The loaded Python object.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Object file not found at: {filepath}")

    return joblib.load(filepath)
