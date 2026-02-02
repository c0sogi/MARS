import os
import random
import numpy as np
import joblib
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch might not be installed or needed for all components,
        # but we handle it if present as per requirements.
        pass


def save_model(model, path: str) -> None:
    """
    Saves a model to the specified path using joblib.
    Automatically creates the parent directory if it does not exist.

    Args:
        model: The model object to save.
        path (str): The file path where the model should be saved.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(model, path)


def load_model(path: str):
    """
    Loads a model from the specified path using joblib.

    Args:
        path (str): The file path to load the model from.

    Returns:
        The loaded model object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    return joblib.load(path)


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """
    Prints validation metrics with full precision.

    Args:
        metrics (dict): A dictionary of metric names and values.
        prefix (str): An optional prefix for the log message.
    """
    prefix_str = f"[{prefix}] " if prefix else ""
    for key, value in metrics.items():
        print(f"{prefix_str}{key}: {value}")
