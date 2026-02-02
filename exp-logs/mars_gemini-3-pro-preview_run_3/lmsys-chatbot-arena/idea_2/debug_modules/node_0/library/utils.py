import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, filepath: str):
    """
    Saves a model checkpoint (state dictionary) to the specified filepath.
    Creates the directory if it does not exist.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        filepath (str): The path where the checkpoint will be saved.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, device: str = Config.DEVICE):
    """
    Loads a model checkpoint from the specified filepath.

    Args:
        filepath (str): The path to the checkpoint file.
        device (str): The device to map the location to ('cpu' or 'cuda').

    Returns:
        dict: The loaded state dictionary, or None if the file does not exist.
    """
    if not os.path.exists(filepath):
        return None
    return torch.load(filepath, map_location=device)


def print_metrics(metrics: dict):
    """
    Prints metric values with full precision.

    Args:
        metrics (dict): A dictionary where keys are metric names and values are the scores.
    """
    for key, value in metrics.items():
        print(f"{key}: {value}")
